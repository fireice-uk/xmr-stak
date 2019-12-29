/* XMRig
 * Copyright 2010      Jeff Garzik              <jgarzik@pobox.com>
 * Copyright 2012-2014 pooler                   <pooler@litecoinpool.org>
 * Copyright 2014      Lucas Jones              <https://github.com/lucasjones>
 * Copyright 2014-2016 Wolf9466                 <https://github.com/OhGodAPet>
 * Copyright 2016      Jay D Dee                <jayddee246@gmail.com>
 * Copyright 2018      Lee Clagett              <https://github.com/vtnerd>
 * Copyright 2018-2019 tevador                  <tevador@gmail.com>
 * Copyright 2018-2019 SChernykh                <https://github.com/SChernykh>
 * Copyright 2000      Transmeta Corporation    <https://github.com/intel/msr-tools>
 * Copyright 2004-2008 H. Peter Anvin           <https://github.com/intel/msr-tools>
 * Copyright 2016-2019 XMRig                    <https://github.com/xmrig>, <support@xmrig.com>
 * Copyright 2017-2019 XMR-Stak                 <https://github.com/fireice-uk>, <https://github.com/psychocrypt>
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

#ifdef _WIN32

#include "xmrstak/misc/console.hpp"
#include "win_msr.hpp"

#include <windows.h>
#include <string>
#include <thread>

#define SERVICE_NAME L"WinRing0_1_2_0"

static SC_HANDLE hManager;
static SC_HANDLE hService;

static bool uninstall_driver()
{
    bool result = true;
    DWORD err;
    SERVICE_STATUS serviceStatus;
    if (!ControlService(hService, SERVICE_CONTROL_STOP, &serviceStatus)) {
        err = GetLastError();
		printer::inst()->print_msg(L0, "Failed to stop WinRing0 driver, error %u", err);
        result = false;
    }
    if (!DeleteService(hService)) {
        err = GetLastError();
		printer::inst()->print_msg(L0, "Failed to remove WinRing0 driver, error %u", err);
        result = false;
    }
    return result;
}

static HANDLE install_driver()
{
    DWORD err = 0;

    hManager = OpenSCManager(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!hManager) {
        err = GetLastError();
		printer::inst()->print_msg(L0, "Failed to open service control manager, error %u", err);
        return 0;
    }

    std::vector<wchar_t> dir;
    dir.resize(MAX_PATH);
    do {
        dir.resize(dir.size() * 2);
        DWORD len = GetModuleFileNameW(NULL, dir.data(), dir.size());
        err = GetLastError();
    } while (err == ERROR_INSUFFICIENT_BUFFER);

    if (err != ERROR_SUCCESS) {
		printer::inst()->print_msg(L0, "Failed to get path to driver, error %u", err);
        return 0;
    }

    for (auto it = dir.end(); it != dir.begin(); --it) {
        if ((*it == L'\\') || (*it == L'/')) {
            ++it;
            *it = L'\0';
            break;
        }
    }

    std::wstring driverPath = dir.data();
    driverPath += L"WinRing0x64.sys";

    hService = OpenServiceW(hManager, SERVICE_NAME, SERVICE_ALL_ACCESS);
    if (hService) {
        if (!uninstall_driver()) {
            return 0;
        }
        CloseServiceHandle(hService);
        hService = 0;
    }
    else {
        err = GetLastError();
        if (err != ERROR_SERVICE_DOES_NOT_EXIST) {
			printer::inst()->print_msg(L0, "Failed to open WinRing0 driver, error %u", err);
            return 0;
        }
    }

    hService = CreateServiceW(hManager, SERVICE_NAME, SERVICE_NAME, SERVICE_ALL_ACCESS, SERVICE_KERNEL_DRIVER, SERVICE_DEMAND_START, SERVICE_ERROR_NORMAL, driverPath.c_str(), nullptr, nullptr, nullptr, nullptr, nullptr);
    if (!hService) {
		printer::inst()->print_msg(L0, "Failed to install WinRing0 driver, error %u", err);
    }

    if (!StartService(hService, 0, nullptr)) {
        err = GetLastError();
        if (err != ERROR_SERVICE_ALREADY_RUNNING) {
			printer::inst()->print_msg(L0, "Failed to start WinRing0 driver, error %u", err);
            return 0;
        }
    }

    HANDLE hDriver = CreateFileW(L"\\\\.\\" SERVICE_NAME, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (!hDriver) {
        err = GetLastError();
 		printer::inst()->print_msg(L0, "Failed to connect to WinRing0 driver, error %u", err);
        return 0;
    }

    return hDriver;
}


#define IOCTL_WRITE_MSR CTL_CODE(40000, 0x822, METHOD_BUFFERED, FILE_ANY_ACCESS)

static bool wrmsr(HANDLE hDriver, uint32_t reg, uint64_t value) {
    struct {
        uint32_t reg;
        uint32_t value[2];
    } input;
    static_assert(sizeof(input) == 12, "Invalid struct size for WinRing0 driver");

    input.reg = reg;
    *((uint64_t*)input.value) = value;

    DWORD output;
    DWORD k;
    if (!DeviceIoControl(hDriver, IOCTL_WRITE_MSR, &input, sizeof(input), &output, sizeof(output), &k, nullptr)) {
        const DWORD err = GetLastError();
		printer::inst()->print_msg(L0, "Setting MSR %x to %llx failed.", reg, int_port(value));
        return false;
    }

    return true;
}

void load_win_msrs(const std::vector<msr_reg>& regs)
{
	printer::inst()->print_msg(L0, "MSR mod: loading WinRing0 driver");

    HANDLE hDriver = install_driver();
    if (!hDriver) {
        if (hService) {
            uninstall_driver();
            CloseServiceHandle(hService);
        }
        if (hManager) {
            CloseServiceHandle(hManager);
        }
        return;
    }

    printer::inst()->print_msg(L0, "MSR mod: setting MSR register values");

	std::thread wrmsr_thread([hDriver, &regs]() {
        for (uint64_t i = 0, n = std::thread::hardware_concurrency(); i < n; ++i) {
			SetThreadAffinityMask(GetCurrentThread(), 1ULL << i);
			for(const msr_reg& r : regs)
				wrmsr(hDriver, r.addr, r.val);
        }
    });
    wrmsr_thread.join();

    CloseHandle(hDriver);

    uninstall_driver();

    CloseServiceHandle(hService);
    CloseServiceHandle(hManager);

	printer::inst()->print_msg(L0, "MSR mod: all done, WinRing0 driver unloaded");
}

#endif
