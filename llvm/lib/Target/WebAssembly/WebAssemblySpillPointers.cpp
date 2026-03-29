//===-- WebAssemblySpillPointers.cpp - Spill pointers to shadow stack ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a pass that spills pointer-typed virtual registers to
/// the shadow stack (linear memory) before function calls. This allows
/// conservative garbage collectors like Boehm GC to scan the shadow stack and
/// find all live pointers.
///
/// Similar to binaryen's SpillPointers pass, this ensures GC correctness by
/// making sure all pointer values that are live across calls are visible in
/// linear memory where the GC can find them.
///
/// Instead of conservatively spilling all I32/I64 registers, this pass performs
/// a dataflow analysis to identify which virtual registers actually hold
/// potential pointer values. Seed pointers are identified from call results,
/// memory loads, function arguments, and global gets. Pointer-ness is then
/// propagated through a blocklist approach: any I32/I64-producing instruction
/// whose input includes a potential pointer will propagate pointer-ness to its
/// result, UNLESS the instruction is on a blocklist of operations that
/// definitely destroy pointer structure. This blocklist includes MUL, DIV, REM,
/// SHL, SHR, ROT, CLZ, CTZ, POPCNT, XOR, and comparisons. Operations like
/// ADD, SUB, AND, OR, SELECT, COPY, PHI, and type conversions (WRAP, EXTEND)
/// propagate pointer-ness because they can preserve recognizable pointer values
/// (e.g., AND for alignment, OR for tagging, ADD/SUB for GEP offsets).
///
/// The blocklist approach is safer than an allowlist for conservative GC: any
/// unknown or newly-added instruction defaults to propagating pointer-ness,
/// which may cause some unnecessary spills but will never miss a real pointer.
///
/// The pass runs after register allocation but before ExplicitLocals, so it
/// can work with virtual registers and insert machine instructions.
///
/// This pass is enabled by default at optimization levels above -O0.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LowLevelTypeImpl.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-spill-pointers"

namespace {

/// Returns true if a call instruction is known to be safe for GC purposes,
/// meaning it cannot trigger a garbage collection and therefore does not
/// require pointer spills. This includes:
///
/// - **readnone nounwind calls**: In LLVM IR, function attributes describe
///   a function's behavior. `readnone` (also called `doesNotAccessMemory`)
///   means the function does not read from or write to any memory -- it is a
///   pure computation (e.g., a math function like sin/cos, or a simple
///   integer computation). `nounwind` (also called `doesNotThrow`) means the
///   function will never throw an exception or invoke an unwinder. A function
///   with BOTH attributes cannot possibly trigger garbage collection, because
///   GC requires either memory allocation (which involves writing memory) or
///   a safepoint mechanism (which involves throwing/unwinding). Therefore,
///   pointer values that are live across such calls do not need to be spilled
///   to the shadow stack.
///   Example from test: `declare i32 @readnone_callee() readnone nounwind`
///
/// - **Known safe library intrinsics**: memcpy, memmove, memset only copy or
///   set existing memory and do not allocate new memory, so they cannot
///   trigger GC. These are frequently emitted by LLVM as external symbol
///   calls (MO_ExternalSymbol) rather than global function references.
static bool isCallSafeForGC(const MachineInstr &MI) {
  assert(MI.isCall());

  for (unsigned I = 0, E = MI.getNumOperands(); I < E; ++I) {
    const MachineOperand &MO = MI.getOperand(I);

    if (MO.isGlobal()) {
      if (auto *Callee = dyn_cast<Function>(MO.getGlobal())) {
        // doesNotAccessMemory() checks for the `readnone` attribute.
        // doesNotThrow() checks for the `nounwind` attribute.
        // Together they guarantee the call is a pure computation that cannot
        // trigger GC. Example: `declare i32 @readnone_callee() readnone nounwind`
        if (Callee->doesNotAccessMemory() && Callee->doesNotThrow())
          return true;

        // Known-safe library functions that only read/write memory but never
        // allocate, and therefore cannot trigger GC.
        StringRef Name = Callee->getName();
        if (Name == "memcpy" || Name == "memmove" || Name == "memset")
          return true;
      }
      break;
    }

    // memcpy/memmove/memset are often lowered as external symbol calls
    // (MO_ExternalSymbol) rather than global address calls.
    if (MO.isSymbol()) {
      StringRef Name = MO.getSymbolName();
      if (Name == "memcpy" || Name == "memmove" || Name == "memset")
        return true;
      break;
    }
  }

  return false;
}

/// Returns true if a load instruction is loading a value that could be a
/// pointer. We check the MachineMemOperand's type info to distinguish pointer
/// loads from non-pointer loads.
///
/// **What is LLT (Low Level Type)?**
/// LLVM uses LLT to represent types at the machine instruction level. Unlike
/// LLVM IR types (i32, i64, ptr, float, etc.), LLT is a simplified type system
/// used after instruction selection. The key types relevant here are:
///
/// - `s32` (scalar 32-bit): Represents a plain 32-bit integer value. This is
///   what you get from `load i32, ptr %p` in LLVM IR. The "s" stands for
///   "scalar" and "32" is the bit width. Similarly, `s64` is a 64-bit integer.
///
/// - `p0` (pointer in address space 0): Represents a pointer value. This is
///   what you get from `load ptr, ptr %p` in LLVM IR. The "p" stands for
///   "pointer" and "0" is the address space number.
///
/// **Why does this distinction matter?**
/// In WebAssembly, both pointers and integers are represented as i32 (or i64
/// on wasm64). At the machine instruction level, `load i32` and `load ptr`
/// both produce an I32 register. However, the MachineMemOperand attached to
/// the load instruction preserves the original type from LLVM IR:
///   - `load i32, ptr @count` → MachineMemOperand type = `s32` (NOT a pointer)
///   - `load ptr, ptr @table`  → MachineMemOperand type = `p0` (IS a pointer)
///
/// By checking MMO->getType().isPointer() vs isScalar(), we can avoid
/// unnecessarily treating integer loads as pointer values, which would cause
/// false-positive spills.
///
/// If we cannot determine the type (no MachineMemOperand available), we
/// conservatively assume the loaded value could be a pointer to ensure GC
/// safety.
static bool isLoadOfPotentialPointer(const MachineInstr &MI, bool Is64Bit) {
  assert(MI.mayLoad());

  // Check MachineMemOperand type info
  for (auto *MMO : MI.memoperands()) {
    LLT MemType = MMO->getType();
    if (MemType.isValid()) {
      // LLT preserves pointer vs scalar distinction from LLVM IR.
      // If the memory type is a pointer, the loaded value could be a pointer.
      // If it's a scalar (e.g., s32 for `load i32`), it's not a pointer.
      if (MemType.isPointer())
        return true;
      if (MemType.isScalar())
        return false;
    }
  }

  // If no memory operand info available, conservatively assume pointer.
  return true;
}

/// Returns true if an instruction's I32/I64 result is definitely not a pointer,
/// even if some of its inputs are pointers. This blocklist approach means any
/// instruction NOT listed here will conservatively propagate pointer-ness from
/// its inputs to its output, ensuring no real pointers are missed.
static bool isDefinitelyNotPointerResult(unsigned Opc) {
  switch (Opc) {
  // Multiplication destroys pointer structure
  case WebAssembly::MUL_I32:
  case WebAssembly::MUL_I32_S:
  case WebAssembly::MUL_I64:
  case WebAssembly::MUL_I64_S:
  // Division destroys pointer structure
  case WebAssembly::DIV_S_I32:
  case WebAssembly::DIV_S_I32_S:
  case WebAssembly::DIV_U_I32:
  case WebAssembly::DIV_U_I32_S:
  case WebAssembly::DIV_S_I64:
  case WebAssembly::DIV_S_I64_S:
  case WebAssembly::DIV_U_I64:
  case WebAssembly::DIV_U_I64_S:
  // Remainder is not a pointer
  case WebAssembly::REM_S_I32:
  case WebAssembly::REM_S_I32_S:
  case WebAssembly::REM_U_I32:
  case WebAssembly::REM_U_I32_S:
  case WebAssembly::REM_S_I64:
  case WebAssembly::REM_S_I64_S:
  case WebAssembly::REM_U_I64:
  case WebAssembly::REM_U_I64_S:
  // Shifts destroy pointer structure
  case WebAssembly::SHL_I32:
  case WebAssembly::SHL_I32_S:
  case WebAssembly::SHL_I64:
  case WebAssembly::SHL_I64_S:
  case WebAssembly::SHR_S_I32:
  case WebAssembly::SHR_S_I32_S:
  case WebAssembly::SHR_U_I32:
  case WebAssembly::SHR_U_I32_S:
  case WebAssembly::SHR_S_I64:
  case WebAssembly::SHR_S_I64_S:
  case WebAssembly::SHR_U_I64:
  case WebAssembly::SHR_U_I64_S:
  // Rotations destroy pointer structure
  case WebAssembly::ROTL_I32:
  case WebAssembly::ROTL_I32_S:
  case WebAssembly::ROTL_I64:
  case WebAssembly::ROTL_I64_S:
  case WebAssembly::ROTR_I32:
  case WebAssembly::ROTR_I32_S:
  case WebAssembly::ROTR_I64:
  case WebAssembly::ROTR_I64_S:
  // Bit counting produces small integers, not pointers
  case WebAssembly::CLZ_I32:
  case WebAssembly::CLZ_I32_S:
  case WebAssembly::CLZ_I64:
  case WebAssembly::CLZ_I64_S:
  case WebAssembly::CTZ_I32:
  case WebAssembly::CTZ_I32_S:
  case WebAssembly::CTZ_I64:
  case WebAssembly::CTZ_I64_S:
  case WebAssembly::POPCNT_I32:
  case WebAssembly::POPCNT_I32_S:
  case WebAssembly::POPCNT_I64:
  case WebAssembly::POPCNT_I64_S:
  // XOR completely transforms the value; GC cannot trace XOR'd pointers
  case WebAssembly::XOR_I32:
  case WebAssembly::XOR_I32_S:
  case WebAssembly::XOR_I64:
  case WebAssembly::XOR_I64_S:
  // Comparisons produce boolean (0 or 1), not pointers
  case WebAssembly::EQ_I32:
  case WebAssembly::EQ_I32_S:
  case WebAssembly::EQ_I64:
  case WebAssembly::EQ_I64_S:
  case WebAssembly::NE_I32:
  case WebAssembly::NE_I32_S:
  case WebAssembly::NE_I64:
  case WebAssembly::NE_I64_S:
  case WebAssembly::LT_S_I32:
  case WebAssembly::LT_S_I32_S:
  case WebAssembly::LT_U_I32:
  case WebAssembly::LT_U_I32_S:
  case WebAssembly::LT_S_I64:
  case WebAssembly::LT_S_I64_S:
  case WebAssembly::LT_U_I64:
  case WebAssembly::LT_U_I64_S:
  case WebAssembly::GT_S_I32:
  case WebAssembly::GT_S_I32_S:
  case WebAssembly::GT_U_I32:
  case WebAssembly::GT_U_I32_S:
  case WebAssembly::GT_S_I64:
  case WebAssembly::GT_S_I64_S:
  case WebAssembly::GT_U_I64:
  case WebAssembly::GT_U_I64_S:
  case WebAssembly::LE_S_I32:
  case WebAssembly::LE_S_I32_S:
  case WebAssembly::LE_U_I32:
  case WebAssembly::LE_U_I32_S:
  case WebAssembly::LE_S_I64:
  case WebAssembly::LE_S_I64_S:
  case WebAssembly::LE_U_I64:
  case WebAssembly::LE_U_I64_S:
  case WebAssembly::GE_S_I32:
  case WebAssembly::GE_S_I32_S:
  case WebAssembly::GE_U_I32:
  case WebAssembly::GE_U_I32_S:
  case WebAssembly::GE_S_I64:
  case WebAssembly::GE_S_I64_S:
  case WebAssembly::GE_U_I64:
  case WebAssembly::GE_U_I64_S:
  case WebAssembly::EQZ_I32:
  case WebAssembly::EQZ_I32_S:
  case WebAssembly::EQZ_I64:
  case WebAssembly::EQZ_I64_S:
    return true;
  default:
    return false;
  }
}

class WebAssemblySpillPointers final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Spill Pointers to Shadow Stack";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervals>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID;
  WebAssemblySpillPointers() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblySpillPointers::ID = 0;
INITIALIZE_PASS(WebAssemblySpillPointers, DEBUG_TYPE,
                "Spill pointer values to shadow stack for GC", false, false)

FunctionPass *llvm::createWebAssemblySpillPointers() {
  return new WebAssemblySpillPointers();
}

bool WebAssemblySpillPointers::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG({
    dbgs() << "********** Spilling Pointers **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  const auto *TII = MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  auto &MRI = MF.getRegInfo();
  auto &LIS = getAnalysis<LiveIntervals>();
  auto &MFI = MF.getFrameInfo();
  bool Changed = false;

  // Determine if we're in 32-bit or 64-bit mode
  bool Is64Bit = MF.getSubtarget<WebAssemblySubtarget>().hasAddr64();
  int PtrSize = Is64Bit ? 8 : 4;

  // --- Pointer classification via dataflow analysis ---
  // Phase 1: Identify "seed" potential pointers from instructions that can
  // produce pointer values.
  DenseSet<Register> PotentialPointers;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      for (unsigned I = 0, E = MI.getNumOperands(); I < E; ++I) {
        const MachineOperand &MO = MI.getOperand(I);
        if (!MO.isReg() || !MO.isDef() || !MO.getReg().isVirtual())
          continue;

        Register DefReg = MO.getReg();
        if (MRI.reg_nodbg_empty(DefReg))
          continue;

        const auto *RC = MRI.getRegClass(DefReg);
        if (RC != &WebAssembly::I32RegClass &&
            RC != &WebAssembly::I64RegClass)
          continue;

        unsigned Opc = MI.getOpcode();

        // Call results: for direct calls, only treat as potential pointers if the
        // callee's return type is a pointer (e.g., malloc, GC_malloc). For
        // indirect calls, conservatively treat all I32/I64 results as pointers.
        if (MI.isCall()) {
          bool MayBePointer = true;
          for (unsigned J = 0, JE = MI.getNumOperands(); J < JE; ++J) {
            const MachineOperand &CalleeOp = MI.getOperand(J);
            if (CalleeOp.isGlobal()) {
              if (auto *Callee =
                      dyn_cast<Function>(CalleeOp.getGlobal())) {
                Type *RetTy = Callee->getReturnType();
                // For multi-value returns (struct returns), check all elements
                if (auto *STy = dyn_cast<StructType>(RetTy)) {
                  // Find which element this def corresponds to
                  // For simplicity, if any element is a pointer, mark as pointer
                  MayBePointer = false;
                  for (unsigned K = 0; K < STy->getNumElements(); ++K) {
                    if (STy->getElementType(K)->isPointerTy()) {
                      MayBePointer = true;
                      break;
                    }
                  }
                } else {
                  MayBePointer = RetTy->isPointerTy();
                }
              }
              break;
            }
          }
          if (MayBePointer)
            PotentialPointers.insert(DefReg);
          continue;
        }

        // Values loaded from memory could be pointers, but only if the loaded
        // type is a pointer type. Loading a plain i32 (e.g., `load i32, ptr @count`)
        // should not be treated as a pointer.
        if (MI.mayLoad()) {
          if (isLoadOfPotentialPointer(MI, Is64Bit))
            PotentialPointers.insert(DefReg);
          continue;
        }

        // Function arguments: only treat as potential pointers if the original
        // LLVM IR parameter type is a pointer. Pure integer arguments (e.g.,
        // sizes, offsets, flags) should not be treated as pointers.
        if (WebAssembly::isArgument(Opc)) {
          unsigned ArgIdx = MI.getOperand(1).getImm();
          const Function &F = MF.getFunction();
          if (ArgIdx < F.arg_size() &&
              F.getArg(ArgIdx)->getType()->isPointerTy()) {
            PotentialPointers.insert(DefReg);
          }
          continue;
        }
      }
    }
  }

  // Phase 2: Propagate pointer-ness through dataflow.
  // Use a blocklist approach: any I32/I64-producing instruction that has a
  // potential pointer input will propagate pointer-ness to its result, UNLESS
  // the instruction is on the blocklist of operations that definitely destroy
  // pointer structure. This ensures safety: unknown instructions default to
  // propagating, which may cause extra spills but won't miss real pointers.
  //
  // Propagated: ADD, SUB, AND (alignment), OR (tagging), SELECT, COPY, PHI,
  //             WRAP, EXTEND, and any unlisted instruction.
  // Blocked: MUL, DIV, REM, SHL, SHR, ROT, CLZ, CTZ, POPCNT, XOR,
  //          comparisons (EQ, NE, LT, GT, LE, GE, EQZ).
  bool Propagated = true;
  while (Propagated) {
    Propagated = false;
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        for (unsigned I = 0, E = MI.getNumOperands(); I < E; ++I) {
          const MachineOperand &MO = MI.getOperand(I);
          if (!MO.isReg() || !MO.isDef() || !MO.getReg().isVirtual())
            continue;

          Register DefReg = MO.getReg();
          if (PotentialPointers.count(DefReg))
            continue;

          const auto *RC = MRI.getRegClass(DefReg);
          if (RC != &WebAssembly::I32RegClass &&
              RC != &WebAssembly::I64RegClass)
            continue;

          unsigned Opc = MI.getOpcode();

          // Skip instructions whose results are definitely not pointers.
          if (isDefinitelyNotPointerResult(Opc))
            continue;

          // Skip loads: a load takes a pointer address as input but produces
          // the loaded value, which is NOT necessarily a pointer. Loads are
          // already handled in Phase 1 with proper type checking via
          // isLoadOfPotentialPointer(). We must not propagate pointer-ness
          // from the address operand to the loaded result here.
          // if (MI.mayLoad())
          //   continue;

          // Similarly, skip calls: call results are already classified in
          // Phase 1 based on the callee's return type. We must not propagate
          // pointer-ness from call arguments to the call result.
          // if (MI.isCall())
          //   continue;

          // Check if any register use operand is a potential pointer.
          for (unsigned J = 0, JE = MI.getNumOperands(); J < JE; ++J) {
            const MachineOperand &UseOp = MI.getOperand(J);
            if (!UseOp.isReg() || UseOp.isDef() || !UseOp.getReg().isVirtual())
              continue;
            if (PotentialPointers.count(UseOp.getReg())) {
              PotentialPointers.insert(DefReg);
              Propagated = true;
              break;
            }
          }
        }
      }
    }
  }

  // Build the final list of pointer registers
  SmallVector<Register, 16> PointerRegs;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (PotentialPointers.count(Reg))
      PointerRegs.push_back(Reg);
  }

  if (PointerRegs.empty()) {
    LLVM_DEBUG(dbgs() << "  No pointer registers, skipping\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  Found " << PointerRegs.size()
                    << " potential pointer registers\n");

  // First pass: determine which potential pointer vregs are live across any
  // call. Only allocate spill slots for those that actually need spilling to
  // avoid creating unnecessary stack frame space.
  //
  // A register is live "across" a call if it is live both before AND after the
  // call instruction:
  //   - liveAt(CallIdx): the register was defined before the call and its value
  //     exists at the call's base slot. This excludes call results (which are
  //     defined by the call, so their live range starts at getRegSlot()).
  //   - liveAt(CallIdx.getRegSlot()): the register is still live at the call's
  //     def slot. This excludes values that are merely consumed as call
  //     arguments (their live range ends at getRegSlot()).
  DenseSet<Register> NeedSpill;
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (!MI.isCall())
        continue;
      // Skip calls that are known to be safe for GC (can't trigger collection)
      // if (isCallSafeForGC(MI))
      //   continue;
      SlotIndex CallIdx = LIS.getInstructionIndex(MI);
      for (Register Reg : PointerRegs) {
        if (NeedSpill.count(Reg))
          continue;
        if (LIS.hasInterval(Reg)) {
          auto &LI = LIS.getInterval(Reg);
          if (LI.liveAt(CallIdx) && LI.liveAt(CallIdx.getRegSlot()))
            NeedSpill.insert(Reg);
        }
      }
    }
  }

  if (NeedSpill.empty()) {
    LLVM_DEBUG(dbgs() << "  No pointers live across calls, skipping\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  " << NeedSpill.size()
                    << " pointer registers need spilling\n");

  // Allocate frame index slots only for vregs that need spilling
  DenseMap<Register, int> SpillSlots;
  for (Register Reg : PointerRegs) {
    if (!NeedSpill.count(Reg))
      continue;
    int FI = MFI.CreateSpillStackObject(PtrSize, Align(PtrSize));
    SpillSlots[Reg] = FI;
    LLVM_DEBUG(dbgs() << "  Allocated frame slot " << FI << " for "
                      << printReg(Reg) << "\n");
  }

  // Second pass: insert spill stores before each call
  for (auto &MBB : MF) {
    for (auto MII = MBB.begin(); MII != MBB.end(); ++MII) {
      MachineInstr &MI = *MII;

      if (!MI.isCall())
        continue;

      // Skip calls that are known to be safe for GC (can't trigger collection)
      // if (isCallSafeForGC(MI))
      //   continue;

      LLVM_DEBUG(dbgs() << "  Found call at: " << MI);

      // Find which pointer registers are live across this call
      SlotIndex CallIdx = LIS.getInstructionIndex(MI);
      SmallVector<Register, 8> LivePointers;
      for (Register Reg : PointerRegs) {
        if (!NeedSpill.count(Reg))
          continue;
        if (LIS.hasInterval(Reg)) {
          auto &LI = LIS.getInterval(Reg);
          if (LI.liveAt(CallIdx) && LI.liveAt(CallIdx.getRegSlot()))
            LivePointers.push_back(Reg);
        }
      }

      if (LivePointers.empty()) {
        LLVM_DEBUG(dbgs() << "    No live pointers at this call\n");
        continue;
      }

      LLVM_DEBUG(dbgs() << "    Spilling " << LivePointers.size()
                        << " live pointers\n");

      // Insert stores before the call for each live pointer
      for (Register Reg : LivePointers) {
        int FI = SpillSlots[Reg];
        const auto *RC = MRI.getRegClass(Reg);

        unsigned StoreOpcode;
        if (Is64Bit && RC == &WebAssembly::I64RegClass) {
          StoreOpcode = WebAssembly::STORE_I64_A64;
        } else if (!Is64Bit && RC == &WebAssembly::I32RegClass) {
          StoreOpcode = WebAssembly::STORE_I32_A32;
        } else if (Is64Bit && RC == &WebAssembly::I32RegClass) {
          StoreOpcode = WebAssembly::STORE_I32_A64;
        } else {
          StoreOpcode = WebAssembly::STORE_I64_A32;
        }

        BuildMI(MBB, MII, MI.getDebugLoc(), TII->get(StoreOpcode))
            .addImm(0)            // p2align
            .addImm(0)            // offset
            .addFrameIndex(FI)    // address (frame index)
            .addReg(Reg);         // value to store

        LLVM_DEBUG(dbgs() << "      Spilled "
                          << printReg(Reg)
                          << " to frame slot " << FI << "\n");
        Changed = true;
      }
    }
  }

  return Changed;
}
