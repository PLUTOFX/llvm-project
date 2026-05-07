; RUN: llc -mtriple=wasm32 -O2 %s -o - | FileCheck %s

declare ptr @GC_malloc(i32)
declare void @GC_gcollect()

define ptr @cons(i32 %x, ptr %xs) {
entry:
  %n = alloca ptr, align 4
  %tmp = call ptr @GC_malloc(i32 8)
  store ptr %tmp, ptr %n, align 4
  call void @GC_gcollect()
  %val = load ptr, ptr %n, align 4
  ret ptr %val
}

; CHECK: call $GC_malloc
; CHECK: i32.store
; CHECK: call $GC_gcollect
