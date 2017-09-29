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

#include "xmrstak/misc/executor.hpp"
#include "xmrstak/backend/miner_work.hpp"
#include "xmrstak/backend/globalStates.hpp"
#include "xmrstak/backend/backendConnector.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/donate-level.hpp"
#include "xmrstak/params.hpp"
#include "xmrstak/misc/configEditor.hpp"
#include "xmrstak/version.hpp"

#ifndef CONF_NO_HTTPD
#	include "xmrstak/http//httpd.hpp"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>

#ifndef CONF_NO_TLS
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif


#ifdef _WIN32
#	define strcasecmp _stricmp
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

	using namespace xmrstak;

	std::string pathWithName(argv[0]);
	auto pos = pathWithName.rfind("/");
	if(pos == std::string::npos)
	{
		// try windows "\"
		pos = pathWithName.rfind("\\");
	}
	
	Params::inst().executablePrefix = std::string(pathWithName, 0, pos);

	for(int i = 1; i < argc; ++i)
	{
		std::string opName(argv[i]);
		if(opName.compare("-h") == 0 || opName.compare("--help") == 0)
		{
			printer::inst()->print_msg(L0, "Usage: %s [--help|-h] [--benchmark] [-c CONFIGFILE] [CONFIG FILE]", argv[0]);
			win_exit();
			return 0;
		}
		else if(opName.compare("--noCPU") == 0)
		{
			Params::inst().useCPU = false;
		}
		else if(opName.compare("--noAMD") == 0)
		{
			Params::inst().useAMD = false;
		}
		else if(opName.compare("--noAMD") == 0)
		{
			Params::inst().useNVIDIA = false;
		}
		else if(opName.compare("--cpu") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for opName '--cpu' given");
				win_exit();
				return 1;
			}
			Params::inst().configFileCPU = argv[i];
		}
		else if(opName.compare("--amd") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for opName '--amd' given");
				win_exit();
				return 1;
			}
			Params::inst().configFileAMD = argv[i];
		}
		else if(opName.compare("--nvidia") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for opName '--nvidia' given");
				win_exit();
				return 1;
			}
			Params::inst().configFileNVIDIA = argv[i];
		}
		else if(opName.compare("-o") == 0 || opName.compare("--url") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for opName '-o/--url' given");
				win_exit();
				return 1;
			}
			Params::inst().poolURL = argv[i];
		}
		else if(opName.compare("-u") == 0 || opName.compare("--user") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for opName '-u/--user' given");
				win_exit();
				return 1;
			}
			Params::inst().poolUsername = argv[i];
		}
		else if(opName.compare("-p") == 0 || opName.compare("--pass") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for opName '-p/--pass' given");
				win_exit();
				return 1;
			}
			Params::inst().poolPasswd = argv[i];
		}
		else
			Params::inst().configFile = argv[i];
	}

	// check if we need a guided start
	if(!ConfigEditor::file_exist(Params::inst().configFile))
	{
		// load the template of the backend config into a char variable
		const char *tpl =
			#include "../config.tpl"
		;
		ConfigEditor configTpl{};
		configTpl.set(std::string(tpl));
		auto& pool = Params::inst().poolURL;
		if(pool.empty())
		{
			std::cout<<"Please enter:\n- pool address: e.g. pool.usxmrpool.com:3333"<<std::endl;
			std::cin >> pool;
		}
		auto& userName = Params::inst().poolUsername;
		if(userName.empty())
		{
			std::cout<<"- user name (wallet address or pool login):"<<std::endl;
			std::cin >> userName;
		}
		auto& passwd = Params::inst().poolPasswd;
		if(passwd.empty())
		{
			// clear everything from stdin to allow an empty password
			std::cin.clear(); std::cin.ignore(INT_MAX,'\n');
			std::cout<<"- password (mostly empty or x):"<<std::endl;	
			getline(std::cin, passwd);
		}
		configTpl.replace("POOLURL", pool);
		configTpl.replace("POOLUSER", userName);
		configTpl.replace("POOLPASSWD", passwd);
		configTpl.write(Params::inst().configFile);
		std::cout<<"Configuration stored in file '"<<Params::inst().configFile<<"'"<<std::endl;
	}

	if(!jconf::inst()->parse_config(Params::inst().configFile.c_str()))
	{
		win_exit();
		return 0;
	}

	if (!BackendConnector::self_test())
	{
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
	printer::inst()->print_str( XMR_STAK_NAME" " XMR_STAK_VERSION " mining software.\n");
	printer::inst()->print_str("Based on CPU mining code by wolf9466 (heavily optimized by fireice_uk).\n");
#ifndef CONF_NO_CUDA
	printer::inst()->print_str("NVIDIA mining code was written by KlausT and psychocrypt.\n");
#endif
#ifndef CONF_NO_OPENCL
	printer::inst()->print_str("AMD mining code was written by wolf9466.\n");
#endif
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
	std::vector<xmrstak::IBackend*>* pvThreads;

	printer::inst()->print_msg(L0, "Running a 60 second benchmark...");

	uint8_t work[76] = {0};
	xmrstak::miner_work oWork = xmrstak::miner_work("", work, sizeof(work), 0, 0, 0);
	pvThreads = xmrstak::BackendConnector::thread_starter(oWork);

	uint64_t iStartStamp = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();

	std::this_thread::sleep_for(std::chrono::seconds(60));

	oWork = xmrstak::miner_work();
	xmrstak::GlobalStates::inst().switch_work(oWork);

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
