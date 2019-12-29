/* XMRig
 * Copyright 2010      Jeff Garzik <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler      <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466    <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee   <jayddee246@gmail.com>
 * Copyright 2017-2018 XMR-Stak    <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
 * Copyright 2018      Lee Clagett <https://github.com/vtnerd>
 * Copyright 2018-2019 SChernykh   <https://github.com/SChernykh>
 * Copyright 2018-2019 tevador     <tevador@gmail.com>
 * Copyright 2016-2019 XMRig       <https://github.com/xmrig>, <support@xmrig.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */


#include <stdlib.h>
#include <sys/mman.h>


#include "xmrstak/backend/cpu/crypto/common/portable/mm_malloc.h"
#include "xmrstak/backend/cpu/crypto/common/VirtualMemory.h"


#if defined(__APPLE__)
#   include <mach/vm_statistics.h>
#endif

#if defined(__linux__) && !defined(MAP_HUGE_SHIFT)
#	include <asm-generic/mman-common.h>
#endif

#include "xmrstak/misc/console.hpp"

int xmrstak::VirtualMemory::m_globalFlags = 0;


xmrstak::VirtualMemory::VirtualMemory(size_t size, bool hugePages, size_t align) :
    m_size(VirtualMemory::align(size))
{
    if (hugePages) {
        m_scratchpad = static_cast<uint8_t*>(allocateLargePagesMemory(m_size));
        if (m_scratchpad) {
            m_flags |= HUGEPAGES;

            madvise(m_scratchpad, size, MADV_RANDOM | MADV_WILLNEED);

            if (mlock(m_scratchpad, m_size) == 0) {
                m_flags |= LOCK;
            }

            return;
        }
    }

    m_scratchpad = static_cast<uint8_t*>(_mm_malloc(m_size, align));
}


xmrstak::VirtualMemory::~VirtualMemory()
{
    if (!m_scratchpad) {
        return;
    }

    if (isHugePages()) {
        if (m_flags & LOCK) {
            munlock(m_scratchpad, m_size);
        }

        freeLargePagesMemory(m_scratchpad, m_size);
    }
    else {
        _mm_free(m_scratchpad);
    }
}



void *xmrstak::VirtualMemory::allocateExecutableMemory(size_t size)
{
#   if defined(__APPLE__)
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
#   else
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#   endif

    return mem == MAP_FAILED ? nullptr : mem;
}

void *xmrstak::VirtualMemory::allocateLargePagesMemory(size_t size, size_t page_size)
{
#   if defined(__APPLE__)
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0);
#   elif defined(__FreeBSD__)
    void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_ALIGNED_SUPER | MAP_PREFAULT_READ, -1, 0);
#   else
	constexpr int MAP_HUGE_2MB = (21 << MAP_HUGE_SHIFT);
	constexpr int MAP_HUGE_1GB = (30 << MAP_HUGE_SHIFT);

	int page_size_flags = 0;
	if(page_size == 2u)
		page_size_flags |= MAP_HUGE_2MB;
	else if(page_size == 1024u)
		page_size_flags |= MAP_HUGE_1GB;

    void *mem = mmap(0, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE | page_size_flags, 0, 0);
#   endif

    return mem == MAP_FAILED ? nullptr : mem;
}


void xmrstak::VirtualMemory::flushInstructionCache(void *p, size_t size)
{
#   ifdef HAVE_BUILTIN_CLEAR_CACHE
    __builtin___clear_cache(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + size);
#   endif
}


void xmrstak::VirtualMemory::freeLargePagesMemory(void *p, size_t size)
{
    if(munmap(p, size) != 0)
	{
		printer::inst()->print_msg(LDEBUG,"munmap failed %llu", (uint64_t)size);
		size_t page3gib = 3llu*1024*1024*1024;
		printer::inst()->print_msg(LDEBUG,"try to unmap ", page3gib);
		if(munmap(p, page3gib) != 0)
		{
			printer::inst()->print_msg(LDEBUG,"munmap failed %llu", (uint64_t)page3gib);
		}
	}
}


void xmrstak::VirtualMemory::init(bool hugePages)
{
    if (hugePages) {
        m_globalFlags = HUGEPAGES | HUGEPAGES_AVAILABLE;
    }
}


void xmrstak::VirtualMemory::protectExecutableMemory(void *p, size_t size)
{
    mprotect(p, size, PROT_READ | PROT_EXEC);
}


void xmrstak::VirtualMemory::unprotectExecutableMemory(void *p, size_t size)
{
    mprotect(p, size, PROT_WRITE | PROT_EXEC);
}
