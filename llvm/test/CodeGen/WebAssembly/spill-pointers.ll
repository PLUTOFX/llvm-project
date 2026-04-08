; RUN: llc < %s -O2 -asm-verbose=false -wasm-keep-registers | FileCheck %s

; Test that the SpillPointers pass spills all potential pointer values that are
; live at any call site (including values used only as call arguments) to the
; shadow stack. Conservative GCs like Boehm GC scan the shadow stack during
; calls, so every potential pointer that is alive when a call executes must be
; visible there.

target triple = "wasm32-unknown-unknown"

declare ptr @GC_malloc(i32)
declare void @GC_gcollect()
declare void @use_int(i32)
declare void @use_ptr(ptr)

; Test: pointer from malloc should be spilled before a call.
;
; CHECK-LABEL: test_spill_pointer_not_int:
; The pointer from GC_malloc should be stored to shadow stack before GC_gcollect.
; CHECK: call {{.*}}GC_malloc
; CHECK: i32.store
; CHECK: call {{.*}}GC_gcollect
define ptr @test_spill_pointer_not_int(i32 %x) {
entry:
  %ptr = call ptr @GC_malloc(i32 8)
  store i32 %x, ptr %ptr
  call void @GC_gcollect()
  ret ptr %ptr
}

; Test: pointer and derived pointer should be spilled, but pure integer
; multiplication should not affect pointer classification.
;
; CHECK-LABEL: test_pointer_arithmetic:
; The pointer value from GC_malloc should be stored before GC_gcollect.
; CHECK: call {{.*}}GC_malloc
; CHECK: i32.store
; CHECK: call {{.*}}GC_gcollect
define void @test_pointer_arithmetic(i32 %idx) {
entry:
  %size = mul i32 %idx, 4
  %ptr = call ptr @GC_malloc(i32 %size)
  store i32 42, ptr %ptr
  call void @GC_gcollect()
  call void @use_ptr(ptr %ptr)
  ret void
}

; Test: pointer should be spilled before each call where it is live.
;
; CHECK-LABEL: test_multi_call:
; CHECK: call {{.*}}GC_malloc
; CHECK: i32.store
; CHECK: call {{.*}}GC_gcollect
define ptr @test_multi_call() {
entry:
  %p1 = call ptr @GC_malloc(i32 16)
  call void @GC_gcollect()
  ret ptr %p1
}

; Test: integer-only function should NOT generate shadow stack frame or spills.
; Although all i32/i64 arguments are treated as potential pointer seeds, %a and
; %b are dead before the call to use_int (last used in the add/mul sequence),
; so no potential pointer is live at the call site and NeedSpill is empty.
;
; CHECK-LABEL: test_int_only:
; CHECK-NOT: __stack_pointer
; CHECK-NOT: i32.store
; CHECK: i32.add
; CHECK: i32.mul
; CHECK: call {{.*}}use_int
; CHECK-NOT: __stack_pointer
; CHECK: end_function
define void @test_int_only(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  %prod = mul i32 %sum, 3
  call void @use_int(i32 %prod)
  ret void
}

; Test: AND-based pointer alignment should be spilled.
; ptr & ~0xF aligns a pointer down - the result is still a valid pointer that
; the GC must be able to find.
;
; CHECK-LABEL: test_and_alignment:
; CHECK: call {{.*}}GC_malloc
; CHECK: i32.and
; CHECK: i32.store
; CHECK: call {{.*}}GC_gcollect
define ptr @test_and_alignment() {
entry:
  %p = call ptr @GC_malloc(i32 32)
  %i = ptrtoint ptr %p to i32
  %aligned = and i32 %i, -16
  %q = inttoptr i32 %aligned to ptr
  call void @GC_gcollect()
  ret ptr %q
}

; Test: OR-based pointer tagging should be spilled.
; ptr | 1 sets a tag bit but the value is still recognizable as a pointer.
;
; CHECK-LABEL: test_or_tagging:
; CHECK: call {{.*}}GC_malloc
; CHECK: i32.or
; CHECK: i32.store
; CHECK: call {{.*}}GC_gcollect
define ptr @test_or_tagging() {
entry:
  %p = call ptr @GC_malloc(i32 16)
  %i = ptrtoint ptr %p to i32
  %tagged = or i32 %i, 1
  %q = inttoptr i32 %tagged to ptr
  call void @GC_gcollect()
  ret ptr %q
}

; Test: a pointer that is only used as a call argument (consumed by the call,
; not needed after it returns) must still be spilled before the call.
; During the call, GC may run and needs to find this pointer in the shadow
; stack to keep the referenced object alive. Interior pointers derived from GC
; heap objects (e.g., vtable+offset) commonly exhibit this pattern.
;
; CHECK-LABEL: test_ptr_consumed_by_call:
; CHECK: call {{.*}}GC_malloc
; CHECK: i32.store
; CHECK: call {{.*}}use_ptr
define void @test_ptr_consumed_by_call() {
entry:
  %ptr = call ptr @GC_malloc(i32 8)
  ; %ptr is only used as an argument to use_ptr and is not needed after.
  ; It must still be spilled so GC can find the object during the call.
  call void @use_ptr(ptr %ptr)
  ret void
}
