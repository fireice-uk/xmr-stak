#define ALIGN .align
.intel_syntax noprefix
.section .text
.global cryptonigh_v8_mainloop_ivybridge_asm
.global cryptonigh_v8_mainloop_ryzen_asm

ALIGN 64
cryptonigh_v8_mainloop_ivybridge_asm:
	sub rsp, 48
	mov rcx, rdi
	#include "cryptonigh_v8_main_loop_ivybridge.inc"
	add rsp, 48
	ret 0

ALIGN 64
cryptonigh_v8_mainloop_ryzen_asm:
	sub rsp, 48
	mov rcx, rdi
	#include "cryptonigh_v8_main_loop_ryzen.inc"
	add rsp, 48
	ret 0
