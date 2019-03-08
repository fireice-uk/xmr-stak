#ifdef __APPLE__
#   define ALIGN(x) .align 6
#else
#   define ALIGN(x) .align x
#endif
.intel_syntax noprefix
#ifdef __APPLE__
#   define FN_PREFIX(fn) _ ## fn
.text
#else
#   define FN_PREFIX(fn) fn
.section .text
#endif
.global FN_PREFIX(cryptonight_v8_mainloop_ivybridge_asm)
.global FN_PREFIX(cryptonight_v8_mainloop_ryzen_asm)
.global FN_PREFIX(cryptonight_v8_double_mainloop_sandybridge_asm)

ALIGN(64)
FN_PREFIX(cryptonight_v8_mainloop_ivybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cryptonight_v8_main_loop_ivybridge_linux.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cryptonight_v8_mainloop_ryzen_asm):
	sub rsp, 48
	mov rcx, rdi
	#include "cryptonight_v8_main_loop_ryzen_linux.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054

ALIGN(64)
FN_PREFIX(cryptonight_v8_double_mainloop_sandybridge_asm):
	sub rsp, 48
	mov rcx, rdi
	mov rdx, rsi
	#include "cryptonight_v8_double_main_loop_sandybridge_linux.inc"
	add rsp, 48
	ret 0
	mov eax, 3735929054