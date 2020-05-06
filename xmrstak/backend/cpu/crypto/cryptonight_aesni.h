/*
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
  *
  */
#pragma once

#include "../../miner_work.hpp"
#include "cryptonight.h"
#include "xmrstak/backend/cryptonight.hpp"
#include "xmrstak/backend/cpu/jconf.hpp"
#include "xmrstak/backend/cpu/cpuType.hpp"
#include <cfenv>
#include <memory.h>
#include <stdio.h>
#include <utility>
#include "xmrstak/backend/cpu/crypto/randomx/randomx.h"
#include "xmrstak/backend/globalStates.hpp"

#ifdef _WIN64
#include <winsock2.h>
// this comment disable clang include reordering
#include <ntsecapi.h>
#include <tchar.h>
// this comment disable clang include reordering for windows.h
#include <windows.h>
#else
#include <sys/mman.h>
#endif

template <size_t N>
struct Cryptonight_hash;

template <size_t N>
struct RandomX_hash
{
	template <xmrstak_algo_id ALGO, bool SOFT_AES>
	static void hash(const void* input, size_t len, void* output, cryptonight_ctx** ctx, const xmrstak_algo& algo)
	{
		for(size_t i = 0u; i < N; ++i)
			randomx_calculate_hash(ctx[i]->m_rx_vm, (char*)input + len * i, len, (char*)output + 32 * i);
	}
};

namespace
{

void* allocateExecutableMemory(size_t size)
{

#ifdef _WIN64
	return VirtualAlloc(0, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
#if defined(__APPLE__)
	return mmap(0, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
#else
	return mmap(0, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
#endif
}

void protectExecutableMemory(void* p, size_t size)
{
#ifdef _WIN64
	DWORD oldProtect;
	VirtualProtect(p, size, PAGE_EXECUTE_READ, &oldProtect);
#else
	mprotect(p, size, PROT_READ | PROT_EXEC);
#endif
}

void unprotectExecutableMemory(void* p, size_t size)
{
#ifdef _WIN64
	DWORD oldProtect;
	VirtualProtect(p, size, PAGE_EXECUTE_READWRITE, &oldProtect);
#else
	mprotect(p, size, PROT_WRITE | PROT_EXEC);
#endif
}

void flushInstructionCache(void* p, size_t size)
{
#ifdef _WIN64
	::FlushInstructionCache(GetCurrentProcess(), p, size);
#else
#ifndef __FreeBSD__
	__builtin___clear_cache(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + size);
#endif
#endif
}

}

template <size_t N>
struct RandomX_generator
{
	template <xmrstak_algo_id ALGO>
	static void cn_on_new_job(const xmrstak::miner_work& work, cryptonight_ctx** ctx)
	{

		bool algorithm_switched = ctx[0]->last_algo != POW(ALGO);

		if(ctx[0]->m_rx_vm == nullptr)
		{
			int flags = RANDOMX_FLAG_LARGE_PAGES | RANDOMX_FLAG_FULL_MEM | RANDOMX_FLAG_JIT;
			auto cpu_model = xmrstak::cpu::getModel();

			if(cpu_model.aes)
				flags |= RANDOMX_FLAG_HARD_AES;

			for(size_t i = 0; i < N; i++)
			{
				printer::inst()->print_msg(LDEBUG,"%s create vm", POW(ALGO).Name().c_str());
				ctx[i]->m_rx_vm = randomx_create_vm(static_cast<randomx_flags>(flags), nullptr, randomX_global_ctx::inst().getDataset(ctx[i]->numa), ctx[i]->long_state);
				if (!ctx[i]->m_rx_vm)
					ctx[i]->m_rx_vm = randomx_create_vm(static_cast<randomx_flags>(flags - RANDOMX_FLAG_LARGE_PAGES), nullptr, randomX_global_ctx::inst().getDataset(ctx[i]->numa), ctx[i]->long_state);
			}
		}
		else if(algorithm_switched)
		{
			printer::inst()->print_msg(LDEBUG,"%s switched to %s",ctx[0]->last_algo.Name().c_str(), POW(ALGO).Name().c_str());
			// remove old vm and re-initialize the full randomx context
			for(size_t i = 0; i < N; i++)
			{
				randomx_destroy_vm(ctx[i]->m_rx_vm);
				ctx[i]->m_rx_vm = nullptr;
			}
			cn_on_new_job<ALGO>(work, ctx);
			return;
		}

		if(algorithm_switched)
		{
			if(ALGO == randomX)
				randomx_apply_config(RandomX_MoneroConfig);
			else if(ALGO == randomX_loki)
				randomx_apply_config(RandomX_LokiConfig);
			else if(ALGO == randomX_wow)
				randomx_apply_config(RandomX_WowneroConfig);
			else if(ALGO == randomX_arqma)
				randomx_apply_config(RandomX_ArqmaConfig);
			else if(ALGO == randomX_safex)
				randomx_apply_config(RandomX_SafexConfig);
			else if(ALGO == randomX_keva)
				randomx_apply_config(RandomX_KevaConfig);
		}

		for(size_t i = 0; i < N; i++)
			ctx[i]->last_algo = POW(ALGO);

		printer::inst()->print_msg(LDEBUG,"%s check for update dataset with %u threads", POW(ALGO).Name().c_str(), xmrstak::globalStates::inst().iThreadCount);
		randomX_global_ctx::inst().updateDataset(work.seed_hash, xmrstak::globalStates::inst().iThreadCount);
	}
};
