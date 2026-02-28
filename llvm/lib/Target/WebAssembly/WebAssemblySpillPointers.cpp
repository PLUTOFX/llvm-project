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
/// The pass runs after register allocation but before ExplicitLocals, so it
/// can work with virtual registers and insert machine instructions.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
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
  
  // Collect pointer-typed virtual registers
  // In WebAssembly, pointers are i32 or i64 depending on address size
  SmallVector<Register, 16> PointerRegs;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!MRI.reg_nodbg_empty(Reg)) {
      const auto *RC = MRI.getRegClass(Reg);
      // Treat all I32/I64 registers as potential pointers
      // (conservative but safe for GC)
      if (RC == &WebAssembly::I32RegClass || RC == &WebAssembly::I64RegClass) {
        PointerRegs.push_back(Reg);
      }
    }
  }

  if (PointerRegs.empty()) {
    LLVM_DEBUG(dbgs() << "  No pointer registers, skipping\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  Found " << PointerRegs.size() << " pointer registers\n");

  // Allocate frame index slots for spilling pointers
  DenseMap<Register, int> SpillSlots;
  for (Register Reg : PointerRegs) {
    // Create a stack slot for this pointer
    int FI = MFI.CreateSpillStackObject(PtrSize, Align(PtrSize));
    SpillSlots[Reg] = FI;
    LLVM_DEBUG(dbgs() << "  Allocated frame slot " << FI << " for "
                      << printReg(Reg, TII->getRegisterInfo()) << "\n");
  }

  // Scan for function calls and insert spills before them
  for (auto &MBB : MF) {
    for (auto MII = MBB.begin(); MII != MBB.end(); ++MII) {
      MachineInstr &MI = *MII;
      
      // Check if this is a function call
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
        
        // Determine the appropriate store opcode
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

        // Build: STORE p2align, offset, addr, val
        // This stores val to memory at [addr + offset]
        // We use the frame index as the address
        BuildMI(MBB, MII, MI.getDebugLoc(), TII->get(StoreOpcode))
            .addImm(0)            // p2align
            .addImm(0)            // offset  
            .addFrameIndex(FI)    // address (frame index)
            .addReg(Reg);         // value to store

        LLVM_DEBUG(dbgs() << "      Spilled " 
                          << printReg(Reg, TII->getRegisterInfo())
                          << " to frame slot " << FI << "\n");
        Changed = true;
      }
    }
  }

  return Changed;
}
