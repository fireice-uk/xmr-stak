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
#include "xmrstak/misc/utility.hpp"

#ifndef CONF_NO_HTTPD
#	include "xmrstak/http/httpd.hpp"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <time.h>
#include <iostream>

#ifndef CONF_NO_TLS
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif

#ifdef _WIN32
#	define strcasecmp _stricmp
#endif // _WIN32

void do_benchmark();

void help()
{
	using namespace std;
	using namespace xmrstak;

	cout<<"Usage: "<<params::inst().binaryName<<" [OPTION]..."<<endl;
	cout<<" "<<endl;
	cout<<"  -h, --help            show this help"<<endl;
	cout<<"  -v, --version         show version number"<<endl;
	cout<<"  -V, --version-long    show long version number"<<endl;
	cout<<"  -c, --config FILE     common miner configuration file"<<endl;
#if (!defined(CONF_NO_AEON)) && (!defined(CONF_NO_MONERO))
	cout<<"  --currency NAME       currency to mine: monero or aeon"<<endl;
#endif
#ifndef CONF_NO_CPU
	cout<<"  --noCPU               disable the CPU miner backend"<<endl;
	cout<<"  --cpu FILE            CPU backend miner config file"<<endl;
#endif
#ifndef CONF_NO_OPENCL
	cout<<"  --noAMD               disable the AMD miner backend"<<endl;
	cout<<"  --amd FILE            AMD backend miner config file"<<endl;
#endif
#ifndef CONF_NO_CUDA
	cout<<"  --noNVIDIA            disable the NVIDIA miner backend"<<endl;
	cout<<"  --nvidia FILE         NVIDIA backend miner config file"<<endl;
#endif
	cout<<" "<<endl;
	cout<<"The Following options temporary overwrites the config file settings:"<<endl;
	cout<<"  -o, --url URL         pool url and port, e.g. pool.usxmrpool.com:3333"<<endl;
	cout<<"  -u, --user USERNAME   pool user name or wallet address"<<endl;
	cout<<"  -p, --pass PASSWD     pool password, in the most cases x or empty \"\""<<endl;
	cout<<" \n"<<endl;
	cout<< "Version: " << get_version_str_short() << endl;
	cout<<"Brought to by fireice_uk and psychocrypt under GPLv3."<<endl;
}

bool read_yes_no(const char* str)
{
	std::string tmp;
	do
	{
		std::cout << str << std::endl;
		std::cin >> tmp;
		std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
	}
	while(tmp != "y" && tmp != "n" && tmp != "yes" && tmp != "no");

	return tmp == "y" || tmp == "yes";
}

inline const char* bool_to_str(bool v)
{
	return v ? "true" : "false";
}

std::string get_multipool_entry(bool& final)
{
	std::cout<<std::endl<<"- Next Pool:"<<std::endl<<std::endl;

	std::string pool;
	if(xmrstak::params::inst().currency == "monero")
		std::cout<<"- Pool address: e.g. pool.usxmrpool.com:3333"<<std::endl;
	else
		std::cout<<"- Pool address: e.g. mine.aeon-pool.com:5555"<<std::endl;
	std::cin >> pool;

	std::string userName;
	std::cout<<"- Username (wallet address or pool login):"<<std::endl;
	std::cin >> userName;

	std::string passwd;
	std::cin.clear(); std::cin.ignore(INT_MAX,'\n');
	std::cout<<"- Password (mostly empty or x):"<<std::endl;
	getline(std::cin, passwd);

#ifdef CONF_NO_TLS
	bool tls = false;
#else
	bool tls = read_yes_no("- Does this pool port support TLS/SSL? Use no if unknown. (y/N)");
#endif
	bool nicehash = read_yes_no("- Do you want to use nicehash on this pool? (y/n)");

	int64_t pool_weight;
	std::cout << "- Please enter a weight for this pool: "<<std::endl;
	while(!(std::cin >> pool_weight) || pool_weight <= 0)
	{
		std::cin.clear();
		std::cin.ignore(INT_MAX, '\n');
		std::cout << "Invalid weight.  Try 1, 10, 100, etc:" << std::endl;
	}

	final = !read_yes_no("- Do you want to add another pool? (y/n)");

	return "\t{\"pool_address\" : \"" + pool +"\", \"wallet_address\" : \"" + userName +  "\", \"pool_password\" : \"" + 
		passwd + "\", \"use_nicehash\" : " + bool_to_str(nicehash) + ", \"use_tls\" : " + bool_to_str(tls) + 
		", \"tls_fingerprint\" : \"\", \"pool_weight\" : " + std::to_string(pool_weight) + " },\n";
}

void do_guided_config(bool userSetPasswd)
{
	using namespace xmrstak;

	// load the template of the backend config into a char variable
	const char *tpl =
		#include "../config.tpl"
	;

	configEditor configTpl{};
	configTpl.set(std::string(tpl));
	std::cout<<"Please enter:"<<std::endl;
	auto& currency = params::inst().currency;
	if(currency.empty())
	{
		std::string tmp;
#if defined(CONF_NO_AEON)
		tmp = "monero";
#elif defined(CONF_NO_MONERO)
		tmp = "aeon";
#endif
		while(tmp != "monero" && tmp != "aeon")
		{
			std::cout<<"- Currency: 'monero' or 'aeon'"<<std::endl;
			std::cin >> tmp;
			std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
		} 
		currency = tmp;
	}

	auto& pool = params::inst().poolURL;
	bool userSetPool = true;
	if(pool.empty())
	{
		userSetPool = false;
		if(currency == "monero")
			std::cout<<"- Pool address: e.g. pool.usxmrpool.com:3333"<<std::endl;
		else
			std::cout<<"- Pool address: e.g. mine.aeon-pool.com:5555"<<std::endl;
		std::cin >> pool;
	}

	auto& userName = params::inst().poolUsername;
	if(userName.empty())
	{
		std::cout<<"- Username (wallet address or pool login):"<<std::endl;
		std::cin >> userName;
	}

	auto& passwd = params::inst().poolPasswd;
	if(passwd.empty() && (!userSetPasswd))
	{
		// clear everything from stdin to allow an empty password
		std::cin.clear(); std::cin.ignore(INT_MAX,'\n');
		std::cout<<"- Password (mostly empty or x):"<<std::endl;
		getline(std::cin, passwd);
	}

#ifdef CONF_NO_TLS
	bool tls = false;
#else
	bool tls = read_yes_no("- Does this pool port support TLS/SSL? Use no if unknown. (y/N)");
#endif
	bool nicehash = read_yes_no("- Do you want to use nicehash on this pool? (y/n)");

	bool multipool;
	if(!userSetPool)
		multipool = read_yes_no("- Do you want to use multiple pools? (y/n)");
	else
		multipool = false;

	int64_t pool_weight;
	if(multipool)
	{
		std::cout << "Pool weight is a number telling the miner how important the pool is." << std::endl;
		std::cout << "Miner will mine mostly at the pool with the highest weight, unless the pool fails." << std::endl;
		std::cout << "Weight must be an integer larger than 0." << std::endl;
		std::cout << "- Please enter a weight for this pool: "<<std::endl;
		
		while(!(std::cin >> pool_weight) || pool_weight <= 0)
		{
			std::cin.clear();
			std::cin.ignore(INT_MAX, '\n');
			std::cout << "Invalid weight.  Try 1, 10, 100, etc:" << std::endl;
		}
	}
	else
		pool_weight = 1;

	std::string pool_table;
	pool_table += "\t{\"pool_address\" : \"" + pool +"\", \"wallet_address\" : \"" + userName +  "\", \"pool_password\" : \"" + 
		passwd + "\", \"use_nicehash\" : " + bool_to_str(nicehash) + ", \"use_tls\" : " + bool_to_str(tls) + 
		", \"tls_fingerprint\" : \"\", \"pool_weight\" : " + std::to_string(pool_weight) + " },\n";

	if(multipool)
	{
		bool final;
		do
		{
			pool_table += get_multipool_entry(final);
		}
		while(!final);
	}

	configTpl.replace("POOLCONF", pool_table);
	configTpl.replace("CURRENCY", currency);
	configTpl.write(params::inst().configFile);
	std::cout<<"Configuration stored in file '"<<params::inst().configFile<<"'"<<std::endl;
}

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
	std::string seperator("/");
	auto pos = pathWithName.rfind(seperator);

	if(pos == std::string::npos)
	{
		// try windows "\"
		seperator = "\\";
		pos = pathWithName.rfind(seperator);
	}
	params::inst().binaryName = std::string(pathWithName, pos + 1, std::string::npos);
	if(params::inst().binaryName.compare(pathWithName) != 0)
	{
		params::inst().executablePrefix = std::string(pathWithName, 0, pos);
		params::inst().executablePrefix += seperator;
	}

	bool userSetPasswd = false;
	for(int i = 1; i < argc; ++i)
	{
		std::string opName(argv[i]);
		if(opName.compare("-h") == 0 || opName.compare("--help") == 0)
		{
			help();
			win_exit(0);
			return 0;
		}
		if(opName.compare("-v") == 0 || opName.compare("--version") == 0)
		{
			std::cout<< "Version: " << get_version_str_short() << std::endl;
			win_exit();
			return 0;
		}
		else if(opName.compare("-V") == 0 || opName.compare("--version-long") == 0)
		{
			std::cout<< "Version: " << get_version_str() << std::endl;
			win_exit();
			return 0;
		}
		else if(opName.compare("--noCPU") == 0)
		{
			params::inst().useCPU = false;
		}
		else if(opName.compare("--noAMD") == 0)
		{
			params::inst().useAMD = false;
		}
		else if(opName.compare("--noNVIDIA") == 0)
		{
			params::inst().useNVIDIA = false;
		}
		else if(opName.compare("--cpu") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for parameter '--cpu' given");
				win_exit();
				return 1;
			}
			params::inst().configFileCPU = argv[i];
		}
		else if(opName.compare("--amd") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for parameter '--amd' given");
				win_exit();
				return 1;
			}
			params::inst().configFileAMD = argv[i];
		}
		else if(opName.compare("--nvidia") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for parameter '--nvidia' given");
				win_exit();
				return 1;
			}
			params::inst().configFileNVIDIA = argv[i];
		}
		else if(opName.compare("--currency") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for parameter '--currency' given");
				win_exit();
				return 1;
			}
			params::inst().currency = argv[i];
		}
		else if(opName.compare("-o") == 0 || opName.compare("--url") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for parameter '-o/--url' given");
				win_exit();
				return 1;
			}
			params::inst().poolURL = argv[i];
		}
		else if(opName.compare("-u") == 0 || opName.compare("--user") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for parameter '-u/--user' given");
				win_exit();
				return 1;
			}
			params::inst().poolUsername = argv[i];
		}
		else if(opName.compare("-p") == 0 || opName.compare("--pass") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for parameter '-p/--pass' given");
				win_exit();
				return 1;
			}
			userSetPasswd = true;
			params::inst().poolPasswd = argv[i];
		}
		else if(opName.compare("-c") == 0 || opName.compare("--config") == 0)
		{
			++i;
			if( i >=argc )
			{
				printer::inst()->print_msg(L0, "No argument for parameter '-c/--config' given");
				win_exit();
				return 1;
			}
			params::inst().configFile = argv[i];
		}
		else
		{
			printer::inst()->print_msg(L0, "Parameter unknown '%s'",argv[i]);
			win_exit();
			return 1;
		}
	}

	// check if we need a guided start
	if(!configEditor::file_exist(params::inst().configFile))
		do_guided_config(userSetPasswd);

	if(!jconf::inst()->parse_config(params::inst().configFile.c_str()))
	{
		win_exit();
		return 1;
	}

	if (!BackendConnector::self_test())
	{
		win_exit();
		return 1;
	}

#ifndef CONF_NO_HTTPD
	if(jconf::inst()->GetHttpdPort() != 0)
	{
		if (!httpd::inst()->start_daemon())
		{
			win_exit();
			return 1;
		}
	}
#endif

	printer::inst()->print_str("-------------------------------------------------------------------\n");
	printer::inst()->print_str(get_version_str_short().c_str());
	printer::inst()->print_str("\n\n");
	printer::inst()->print_str("Brought to you by fireice_uk and psychocrypt under GPLv3.\n");
	printer::inst()->print_str("Based on CPU mining code by wolf9466 (heavily optimized by fireice_uk).\n");
#ifndef CONF_NO_CUDA
	printer::inst()->print_str("Based on NVIDIA mining code by KlausT and psychocrypt.\n");
#endif
#ifndef CONF_NO_OPENCL
	printer::inst()->print_str("Based on OpenCL mining code by wolf9466.\n");
#endif
	char buffer[64];
	snprintf(buffer, sizeof(buffer), "\nConfigurable dev donation level is set to %.1f%%\n\n", fDevDonationLevel * 100.0);
	printer::inst()->print_str(buffer);
	printer::inst()->print_str("You can use following keys to display reports:\n");
	printer::inst()->print_str("'h' - hashrate\n");
	printer::inst()->print_str("'r' - results\n");
	printer::inst()->print_str("'c' - connection\n");
	printer::inst()->print_str("-------------------------------------------------------------------\n");
	if(::jconf::inst()->IsCurrencyMonero())
		printer::inst()->print_msg(L0,"Start mining: MONERO");
	else
		printer::inst()->print_msg(L0,"Start mining: AEON");

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
	std::vector<xmrstak::iBackend*>* pvThreads;

	printer::inst()->print_msg(L0, "Running a 60 second benchmark...");

	uint8_t work[76] = {0};
	xmrstak::miner_work oWork = xmrstak::miner_work("", work, sizeof(work), 0, false, 0);
	pvThreads = xmrstak::BackendConnector::thread_starter(oWork);

	uint64_t iStartStamp = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();

	std::this_thread::sleep_for(std::chrono::seconds(60));

	oWork = xmrstak::miner_work();
	xmrstak::pool_data dat;
	xmrstak::globalStates::inst().switch_work(oWork, dat);

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
