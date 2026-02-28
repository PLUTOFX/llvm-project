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
/// propagated through ADD, SUB, SELECT, COPY, and PHI instructions. Registers
/// defined by pure arithmetic (MUL, DIV, REM), bitwise, comparison, constant,
/// or conversion instructions are not treated as pointers.
///
/// The pass runs after register allocation but before ExplicitLocals, so it
/// can work with virtual registers and insert machine instructions.
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
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-spill-pointers"

namespace {

/// Returns true if the opcode is an ADD or SUB that could represent pointer
/// arithmetic (e.g., base + offset from GEP lowering).
static bool isPointerArithmeticOpcode(unsigned Opc) {
  switch (Opc) {
  case WebAssembly::ADD_I32:
  case WebAssembly::ADD_I32_S:
  case WebAssembly::ADD_I64:
  case WebAssembly::ADD_I64_S:
  case WebAssembly::SUB_I32:
  case WebAssembly::SUB_I32_S:
  case WebAssembly::SUB_I64:
  case WebAssembly::SUB_I64_S:
    return true;
  default:
    return false;
  }
}

/// Returns true if the opcode is a SELECT that could propagate pointer values.
static bool isSelectOpcode(unsigned Opc) {
  switch (Opc) {
  case WebAssembly::SELECT_I32:
  case WebAssembly::SELECT_I32_S:
  case WebAssembly::SELECT_I64:
  case WebAssembly::SELECT_I64_S:
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
  const auto *TRI = MF.getSubtarget<WebAssemblySubtarget>().getRegisterInfo();
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

        // Call results could be pointers (e.g., malloc, GC_malloc).
        if (MI.isCall()) {
          PotentialPointers.insert(DefReg);
          continue;
        }

        // Values loaded from memory could be pointers (LOAD, GLOBAL_GET, etc.)
        if (MI.mayLoad()) {
          PotentialPointers.insert(DefReg);
          continue;
        }

        // Function arguments could be pointers.
        if (WebAssembly::isArgument(Opc)) {
          PotentialPointers.insert(DefReg);
          continue;
        }
      }
    }
  }

  // Phase 2: Propagate pointer-ness through dataflow.
  // Instructions that can propagate a pointer value from an operand to their
  // result: ADD/SUB (pointer arithmetic), SELECT, COPY, PHI.
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

          // Only propagate through pointer-preserving instructions.
          if (!isPointerArithmeticOpcode(Opc) && !isSelectOpcode(Opc) &&
              !MI.isCopy() && !MI.isPHI())
            continue;

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

  // Allocate frame index slots for spilling pointers
  DenseMap<Register, int> SpillSlots;
  for (Register Reg : PointerRegs) {
    int FI = MFI.CreateSpillStackObject(PtrSize, Align(PtrSize));
    SpillSlots[Reg] = FI;
    LLVM_DEBUG(dbgs() << "  Allocated frame slot " << FI << " for "
                      << printReg(Reg, TRI) << "\n");
  }

  // Scan for function calls and insert spills before them
  for (auto &MBB : MF) {
    for (auto MII = MBB.begin(); MII != MBB.end(); ++MII) {
      MachineInstr &MI = *MII;

      if (!MI.isCall())
        continue;

      LLVM_DEBUG(dbgs() << "  Found call at: " << MI);

      // Find which pointer registers are live at this call
      SmallVector<Register, 8> LivePointers;
      for (Register Reg : PointerRegs) {
        if (LIS.hasInterval(Reg)) {
          auto &LI = LIS.getInterval(Reg);
          SlotIndex CallIdx = LIS.getInstructionIndex(MI);
          if (LI.liveAt(CallIdx)) {
            LivePointers.push_back(Reg);
          }
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
                          << printReg(Reg, TRI)
                          << " to frame slot " << FI << "\n");
        Changed = true;
      }
    }
  }

  return Changed;
}
