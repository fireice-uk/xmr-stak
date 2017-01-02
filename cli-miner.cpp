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
  */

#include "executor.h"
#include "minethd.h"
#include "jconf.h"
#include "console.h"
#include "donate-level.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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

int main(int argc, char *argv[])
{
	const char* sFilename = "config.txt";

	if(argc >= 2)
	{
		if(strcmp(argv[1], "-h") == 0)
		{
			printer::inst()->print_msg(L0, "Usage %s [CONFIG FILE]", argv[0]);
			win_exit();
			return 0;
		}

		if(argc >= 3 && strcasecmp(argv[1], "-c") == 0)
			sFilename = argv[2];
		else
			sFilename = argv[1];
	}

	if(!jconf::inst()->parse_config(sFilename))
	{
		win_exit();
		return 0;
	}

	printer::inst()->print_str("-------------------------------------------------------------------\n");
	printer::inst()->print_str("XMR-Stak-CPU mining software, CPU Version.\n");
	printer::inst()->print_str("Based on CPU mining code by wolf9466 (heavily optimized by myself).\n");
	printer::inst()->print_str("Brought to you by fireice_uk under GPLv3.\n\n");
	char buffer[64];
	snprintf(buffer, sizeof(buffer), "Configurable dev donation level is set to %.1f %%\n\n", fDevDonationLevel * 100.0);
	printer::inst()->print_str(buffer);
	printer::inst()->print_str("You can use following keys to display reports:\n");
	printer::inst()->print_str("'h' - hashrate\n");
	printer::inst()->print_str("'r' - results\n");
	printer::inst()->print_str("'c' - connection\n");
	printer::inst()->print_str("-------------------------------------------------------------------\n");

	if (!minethd::self_test())
	{
		win_exit();
		return 0;
	}

	executor::inst()->ex_start();

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
	}

	return 0;
}
