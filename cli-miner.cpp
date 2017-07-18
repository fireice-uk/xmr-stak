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
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
  *
  */

#include "executor.h"
#include "minethd.h"
#include "jconf.h"
#include "console.h"
#include "donate-level.h"
#ifndef CONF_NO_HWLOC
#   include "autoAdjustHwloc.hpp"
#else
#   include "autoAdjust.hpp"
#endif
#include "version.h"

#ifndef CONF_NO_HTTPD
#	include "httpd.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <time.h>

#ifndef CONF_NO_TLS
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif

//Do a press any key for the windows folk. *insert any key joke here*
#ifdef _WIN32
void win_exit()
{
	printer::inst()->print_str("Press any key to exit.");
	get_key();
	return;
}

#define strcasecmp _stricmp

#else
void win_exit() { return; }
#endif // _WIN32

void do_benchmark();

int main(int argc, char *argv[])
{
#ifndef CONF_NO_TLS
	SSL_library_init();
	SSL_load_error_strings();
	ERR_load_BIO_strings();
	ERR_load_crypto_strings();
	SSL_load_error_strings();
	OpenSSL_add_all_digests();
#endif

	srand(time(0));

	const char* sFilename = "config.txt";
	bool benchmark_mode = false;

	if(argc >= 2)
	{
		if(strcmp(argv[1], "-h") == 0)
		{
			printer::inst()->print_msg(L0, "Usage %s [CONFIG FILE]", argv[0]);
			win_exit();
			return 0;
		}

		if(argc >= 3 && strcasecmp(argv[1], "-c") == 0)
		{
			sFilename = argv[2];
		}
		else if(argc >= 3 && strcasecmp(argv[1], "benchmark_mode") == 0)
		{
			sFilename = argv[2];
			benchmark_mode = true;
		}
		else
			sFilename = argv[1];
	}

	if(!jconf::inst()->parse_config(sFilename))
	{
		win_exit();
		return 0;
	}

	if(jconf::inst()->NeedsAutoconf())
	{
		autoAdjust adjust;
		adjust.printConfig();
		win_exit();
		return 0;
	}

	if (!minethd::self_test())
	{
		win_exit();
		return 0;
	}

	if(benchmark_mode)
	{
		do_benchmark();
		win_exit();
		return 0;
	}

#ifndef CONF_NO_HTTPD
	if(jconf::inst()->GetHttpdPort() != 0)
	{
		if (!httpd::inst()->start_daemon())
		{
			win_exit();
			return 0;
		}
	}
#endif

	printer::inst()->print_str("-------------------------------------------------------------------\n");
	printer::inst()->print_str( XMR_STAK_NAME" " XMR_STAK_VERSION " mining software, CPU Version.\n");
	printer::inst()->print_str("Based on CPU mining code by wolf9466 (heavily optimized by fireice_uk).\n");
	printer::inst()->print_str("Brought to you by fireice_uk and psychocrypt under GPLv3.\n\n");
	char buffer[64];
	snprintf(buffer, sizeof(buffer), "Configurable dev donation level is set to %.1f %%\n\n", fDevDonationLevel * 100.0);
	printer::inst()->print_str(buffer);
	printer::inst()->print_str("You can use following keys to display reports:\n");
	printer::inst()->print_str("'h' - hashrate\n");
	printer::inst()->print_str("'r' - results\n");
	printer::inst()->print_str("'c' - connection\n");
	printer::inst()->print_str("-------------------------------------------------------------------\n");

	if(strlen(jconf::inst()->GetOutputFile()) != 0)
		printer::inst()->open_logfile(jconf::inst()->GetOutputFile());

	executor::inst()->ex_start(jconf::inst()->DaemonMode());

	using namespace std::chrono;
	uint64_t lastTime = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();

	int key;
	while(true)
	{
		key = get_key();

		switch(key)
		{
		case 'h':
			executor::inst()->push_event(ex_event(EV_USR_HASHRATE));
			break;
		case 'r':
			executor::inst()->push_event(ex_event(EV_USR_RESULTS));
			break;
		case 'c':
			executor::inst()->push_event(ex_event(EV_USR_CONNSTAT));
			break;
		default:
			break;
		}

		uint64_t currentTime = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();

		/* Hard guard to make sure we never get called more than twice per second */
		if( currentTime - lastTime < 500)
			std::this_thread::sleep_for(std::chrono::milliseconds(500 - (currentTime - lastTime)));
		lastTime = currentTime;
	}

	return 0;
}

void do_benchmark()
{
	using namespace std::chrono;
	std::vector<minethd*>* pvThreads;

	printer::inst()->print_msg(L0, "Running a 60 second benchmark...");

	uint8_t work[76] = {0};
	minethd::miner_work oWork = minethd::miner_work("", work, sizeof(work), 0, 0, false, 0);
	pvThreads = minethd::thread_starter(oWork);

	uint64_t iStartStamp = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();

	std::this_thread::sleep_for(std::chrono::seconds(60));

	oWork = minethd::miner_work();
	minethd::switch_work(oWork);

	double fTotalHps = 0.0;
	for (uint32_t i = 0; i < pvThreads->size(); i++)
	{
		double fHps = pvThreads->at(i)->iHashCount;
		fHps /= (pvThreads->at(i)->iTimestamp - iStartStamp) / 1000.0;

		printer::inst()->print_msg(L0, "Thread %u: %.1f H/S", i, fHps);
		fTotalHps += fHps;
	}

	printer::inst()->print_msg(L0, "Total: %.1f H/S", fTotalHps);
}
