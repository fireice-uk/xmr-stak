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

#include <thread>
#include <string>
#include <cmath>
#include <algorithm>
#include <assert.h>
#include <time.h>
#include "executor.h"
#include "jpsock.h"
#include "minethd.h"
#include "jconf.h"
#include "console.h"
#include "donate-level.h"
#include "webdesign.h"

#ifdef _WIN32
#define strncasecmp _strnicmp
#endif // _WIN32

executor* executor::oInst = NULL;

executor::executor()
{
}

void executor::push_timed_event(ex_event&& ev, size_t sec)
{
	std::unique_lock<std::mutex> lck(timed_event_mutex);
	lTimedEvents.emplace_back(std::move(ev), sec_to_ticks(sec));
}

void executor::ex_clock_thd()
{
	size_t iSwitchPeriod = sec_to_ticks(iDevDonatePeriod);
	size_t iDevPortion = (size_t)floor(((double)iSwitchPeriod) * fDevDonationLevel);

	//No point in bothering with less than 10 sec
	if(iDevPortion < sec_to_ticks(10))
		iDevPortion = 0;

	//Add 2 seconds to compensate for connect
	if(iDevPortion != 0)
		iDevPortion += sec_to_ticks(2);

	while (true)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(size_t(iTickTime)));

		push_event(ex_event(EV_PERF_TICK));

		// Service timed events
		std::unique_lock<std::mutex> lck(timed_event_mutex);
		std::list<timed_event>::iterator ev = lTimedEvents.begin();
		while (ev != lTimedEvents.end())
		{
			ev->ticks_left--;
			if(ev->ticks_left == 0)
			{
				push_event(std::move(ev->event));
				ev = lTimedEvents.erase(ev);
			}
			else
				ev++;
		}
		lck.unlock();

		if(iDevPortion == 0)
			continue;

		iSwitchPeriod--;
		if(iSwitchPeriod == 0)
		{
			push_event(ex_event(EV_SWITCH_POOL, usr_pool_id));
			iSwitchPeriod = sec_to_ticks(iDevDonatePeriod);
		}
		else if(iSwitchPeriod == iDevPortion)
		{
			push_event(ex_event(EV_SWITCH_POOL, dev_pool_id));
		}
	}
}

void executor::sched_reconnect()
{
	iReconnectAttempts++;
	size_t iLimit = jconf::inst()->GetGiveUpLimit();
	if(iLimit != 0 && iReconnectAttempts > iLimit)
	{
		printer::inst()->print_msg(L0, "Give up limit reached. Exitting.");
		exit(0);
	}

	long long unsigned int rt = jconf::inst()->GetNetRetry();
	printer::inst()->print_msg(L1, "Pool connection lost. Waiting %lld s before retry (attempt %llu).",
		rt, int_port(iReconnectAttempts));

	auto work = minethd::miner_work();
	minethd::switch_work(work);

	push_timed_event(ex_event(EV_RECONNECT, usr_pool_id), rt);
}

void executor::log_socket_error(std::string&& sError)
{
	vSocketLog.emplace_back(std::move(sError));
	printer::inst()->print_msg(L1, "SOCKET ERROR - %s", vSocketLog.back().msg.c_str());
}

void executor::log_result_error(std::string&& sError)
{
	size_t i = 1, ln = vMineResults.size();
	for(; i < ln; i++)
	{
		if(vMineResults[i].compare(sError))
		{
			vMineResults[i].increment();
			break;
		}
	}

	if(i == ln) //Not found
		vMineResults.emplace_back(std::move(sError));
	else
		sError.clear();
}

void executor::log_result_ok(uint64_t iActualDiff)
{
	iPoolHashes += iPoolDiff;

	size_t ln = iTopDiff.size() - 1;
	if(iActualDiff > iTopDiff[ln])
	{
		iTopDiff[ln] = iActualDiff;
		std::sort(iTopDiff.rbegin(), iTopDiff.rend());
	}

	vMineResults[0].increment();
}

jpsock* executor::pick_pool_by_id(size_t pool_id)
{
	assert(pool_id != invalid_pool_id);

	if(pool_id == dev_pool_id)
		return dev_pool;
	else
		return usr_pool;
}

void executor::on_sock_ready(size_t pool_id)
{
	jpsock* pool = pick_pool_by_id(pool_id);

	if(pool_id == dev_pool_id)
	{
		if(!pool->cmd_login("", ""))
			pool->disconnect();

		current_pool_id = dev_pool_id;
		printer::inst()->print_msg(L1, "Dev pool logged in. Switching work.");
		return;
	}

	printer::inst()->print_msg(L1, "Connected. Logging in...");

	if (!pool->cmd_login(jconf::inst()->GetWalletAddress(), jconf::inst()->GetPoolPwd()))
	{
		if(!pool->have_sock_error())
		{
			log_socket_error(pool->get_call_error());
			pool->disconnect();
		}
	}
	else
	{
		iReconnectAttempts = 0;
		reset_stats();
	}
}

void executor::on_sock_error(size_t pool_id, std::string&& sError)
{
	jpsock* pool = pick_pool_by_id(pool_id);

	if(pool_id == dev_pool_id)
	{
		pool->disconnect();

		if(current_pool_id != dev_pool_id)
			return;

		printer::inst()->print_msg(L1, "Dev pool connection error. Switching work.");
		on_switch_pool(usr_pool_id);
		return;
	}

	log_socket_error(std::move(sError));
	pool->disconnect();
	sched_reconnect();
}

void executor::on_pool_have_job(size_t pool_id, pool_job& oPoolJob)
{
	if(pool_id != current_pool_id)
		return;

	jpsock* pool = pick_pool_by_id(pool_id);

	minethd::miner_work oWork(oPoolJob.sJobID, oPoolJob.bWorkBlob,
		oPoolJob.iWorkLen, oPoolJob.iResumeCnt, oPoolJob.iTarget,
		pool_id != dev_pool_id && jconf::inst()->NiceHashMode(),
		pool_id);

	minethd::switch_work(oWork);

	if(pool_id == dev_pool_id)
		return;

	if(iPoolDiff != pool->get_current_diff())
	{
		iPoolDiff = pool->get_current_diff();
		printer::inst()->print_msg(L2, "Difficulty changed. Now: %llu.", int_port(iPoolDiff));
	}

	printer::inst()->print_msg(L3, "New block detected.");
}

void executor::on_miner_result(size_t pool_id, job_result& oResult)
{
	jpsock* pool = pick_pool_by_id(pool_id);

	if(pool_id == dev_pool_id)
	{
		//Ignore errors silently
		if(pool->is_running() && pool->is_logged_in())
			pool->cmd_submit(oResult.sJobID, oResult.iNonce, oResult.bResult);

		return;
	}

	if (!pool->is_running() || !pool->is_logged_in())
	{
		log_result_error("[NETWORK ERROR]");
		return;
	}

	using namespace std::chrono;
	size_t t_start = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count();
	bool bResult = pool->cmd_submit(oResult.sJobID, oResult.iNonce, oResult.bResult);
	size_t t_len = time_point_cast<milliseconds>(high_resolution_clock::now()).time_since_epoch().count() - t_start;

	if(t_len > 0xFFFF)
		t_len = 0xFFFF;
	iPoolCallTimes.push_back((uint16_t)t_len);

	if(bResult)
	{
		uint64_t* targets = (uint64_t*)oResult.bResult;
		log_result_ok(jpsock::t64_to_diff(targets[3]));
		printer::inst()->print_msg(L3, "Result accepted by the pool.");
	}
	else
	{
		if(!pool->have_sock_error())
		{
			printer::inst()->print_msg(L3, "Result rejected by the pool.");

			std::string error = pool->get_call_error();

			if(strncasecmp(error.c_str(), "Unauthenticated", 15) == 0)
			{
				printer::inst()->print_msg(L2, "Your miner was unable to find a share in time. Either the pool difficulty is too high, or the pool timeout is too low.");
				pool->disconnect();
			}

			log_result_error(std::move(error));
		}
		else
			log_result_error("[NETWORK ERROR]");
	}
}

void executor::on_reconnect(size_t pool_id)
{
	jpsock* pool = pick_pool_by_id(pool_id);

	std::string error;
	if(pool_id == dev_pool_id)
		return;

	printer::inst()->print_msg(L1, "Connecting to pool %s ...", jconf::inst()->GetPoolAddress());

	if(!pool->connect(jconf::inst()->GetPoolAddress(), error))
	{
		log_socket_error(std::move(error));
		sched_reconnect();
	}
}

void executor::on_switch_pool(size_t pool_id)
{
	if(pool_id == current_pool_id)
		return;

	jpsock* pool = pick_pool_by_id(pool_id);
	if(pool_id == dev_pool_id)
	{
		std::string error;

		// If it fails, it fails, we carry on on the usr pool
		// as we never receive further events
		printer::inst()->print_msg(L1, "Connecting to dev pool...");
		const char* dev_pool_addr = jconf::inst()->GetTlsSetting() ? "donate.xmr-stak.net:6666" : "donate.xmr-stak.net:3333";
		if(!pool->connect(dev_pool_addr, error))
			printer::inst()->print_msg(L1, "Error connecting to dev pool. Staying with user pool.");
	}
	else
	{
		printer::inst()->print_msg(L1, "Switching back to user pool.");

		current_pool_id = pool_id;
		pool_job oPoolJob;

		if(!pool->get_current_job(oPoolJob))
		{
			pool->disconnect();
			return;
		}

		minethd::miner_work oWork(oPoolJob.sJobID, oPoolJob.bWorkBlob,
			oPoolJob.iWorkLen, oPoolJob.iResumeCnt, oPoolJob.iTarget,
			jconf::inst()->NiceHashMode(), pool_id);

		minethd::switch_work(oWork);

		if(dev_pool->is_running())
			push_timed_event(ex_event(EV_DEV_POOL_EXIT), 5);
	}
}

void executor::ex_main()
{
	assert(1000 % iTickTime == 0);

	minethd::miner_work oWork = minethd::miner_work();
	pvThreads = minethd::thread_starter(oWork);
	telem = new telemetry(pvThreads->size());

	current_pool_id = usr_pool_id;
	usr_pool = new jpsock(usr_pool_id, jconf::inst()->GetTlsSetting());
	dev_pool = new jpsock(dev_pool_id, jconf::inst()->GetTlsSetting());

	ex_event ev;
	std::thread clock_thd(&executor::ex_clock_thd, this);

	//This will connect us to the pool for the first time
	push_event(ex_event(EV_RECONNECT, usr_pool_id));

	// Place the default success result at position 0, it needs to
	// be here even if our first result is a failure
	vMineResults.emplace_back();

	// If the user requested it, start the autohash printer
	if(jconf::inst()->GetVerboseLevel() >= 4)
		push_timed_event(ex_event(EV_HASHRATE_LOOP), jconf::inst()->GetAutohashTime());

	size_t cnt = 0, i;
	while (true)
	{
		ev = oEventQ.pop();
		switch (ev.iName)
		{
		case EV_SOCK_READY:
			on_sock_ready(ev.iPoolId);
			break;

		case EV_SOCK_ERROR:
			on_sock_error(ev.iPoolId, std::move(ev.sSocketError));
			break;

		case EV_POOL_HAVE_JOB:
			on_pool_have_job(ev.iPoolId, ev.oPoolJob);
			break;

		case EV_MINER_HAVE_RESULT:
			on_miner_result(ev.iPoolId, ev.oJobResult);
			break;

		case EV_RECONNECT:
			on_reconnect(ev.iPoolId);
			break;

		case EV_SWITCH_POOL:
			on_switch_pool(ev.iPoolId);
			break;

		case EV_DEV_POOL_EXIT:
			dev_pool->disconnect();
			break;

		case EV_PERF_TICK:
			for (i = 0; i < pvThreads->size(); i++)
				telem->push_perf_value(i, pvThreads->at(i)->iHashCount.load(std::memory_order_relaxed),
				pvThreads->at(i)->iTimestamp.load(std::memory_order_relaxed));

			if((cnt++ & 0xF) == 0) //Every 16 ticks
			{
				double fHps = 0.0;
				double fTelem;
				bool normal = true;

				for (i = 0; i < pvThreads->size(); i++)
				{
					fTelem = telem->calc_telemetry_data(2500, i);
					if(std::isnormal(fTelem))
					{
						fHps += fTelem;
					}
					else
					{
						normal = false;
						break;
					}
				}

				if(normal && fHighestHps < fHps)
					fHighestHps = fHps;
			}
		break;

		case EV_USR_HASHRATE:
		case EV_USR_RESULTS:
		case EV_USR_CONNSTAT:
			print_report(ev.iName);
			break;

		case EV_HTML_HASHRATE:
		case EV_HTML_RESULTS:
		case EV_HTML_CONNSTAT:
			http_report(ev.iName);
			break;

		case EV_HASHRATE_LOOP:
			print_report(EV_USR_HASHRATE);
			push_timed_event(ex_event(EV_HASHRATE_LOOP), jconf::inst()->GetAutohashTime());
			break;

		case EV_INVALID_VAL:
		default:
			assert(false);
			break;
		}
	}
}

inline const char* hps_format(double h, char* buf, size_t l)
{
	if(std::isnormal(h) || h == 0.0)
	{
		snprintf(buf, l, " %03.1f", h);
		return buf;
	}
	else
		return " (na)";
}

void executor::hashrate_report(std::string& out)
{
	char num[32];
	size_t nthd = pvThreads->size();

	out.reserve(256 + nthd * 64);

	double fTotal[3] = { 0.0, 0.0, 0.0};
	size_t i;

	out.append("HASHRATE REPORT\n");
	out.append("| ID | 2.5s |  60s |  15m |");
	if(nthd != 1)
		out.append(" ID | 2.5s |  60s |  15m |\n");
	else
		out.append(1, '\n');

	for (i = 0; i < nthd; i++)
	{
		double fHps[3];

		fHps[0] = telem->calc_telemetry_data(2500, i);
		fHps[1] = telem->calc_telemetry_data(60000, i);
		fHps[2] = telem->calc_telemetry_data(900000, i);

		snprintf(num, sizeof(num), "| %2u |", (unsigned int)i);
		out.append(num);
		out.append(hps_format(fHps[0], num, sizeof(num))).append(" |");
		out.append(hps_format(fHps[1], num, sizeof(num))).append(" |");
		out.append(hps_format(fHps[2], num, sizeof(num))).append(1, ' ');

		fTotal[0] += fHps[0];
		fTotal[1] += fHps[1];
		fTotal[2] += fHps[2];

		if((i & 0x1) == 1) //Odd i's
			out.append("|\n");
	}

	if((i & 0x1) == 1) //We had odd number of threads
		out.append("|\n");

	if(nthd != 1)
		out.append("-----------------------------------------------------\n");
	else
		out.append("---------------------------\n");

	out.append("Totals:  ");
	out.append(hps_format(fTotal[0], num, sizeof(num)));
	out.append(hps_format(fTotal[1], num, sizeof(num)));
	out.append(hps_format(fTotal[2], num, sizeof(num)));
	out.append(" H/s\nHighest: ");
	out.append(hps_format(fHighestHps, num, sizeof(num)));
	out.append(" H/s\n");
}

char* time_format(char* buf, size_t len, std::chrono::system_clock::time_point time)
{
	time_t ctime = std::chrono::system_clock::to_time_t(time);
	tm stime;

	/*
	 * Oh for god's sake... this feels like we are back to the 90's...
	 * and don't get me started on lack strcpy_s because NIH - use non-standard strlcpy...
	 * And of course C++ implements unsafe version because... reasons
	 */

#ifdef _WIN32
	localtime_s(&stime, &ctime);
#else
	localtime_r(&ctime, &stime);
#endif // __WIN32
	strftime(buf, len, "%F %T", &stime);

	return buf;
}

void executor::result_report(std::string& out)
{
	char num[128];
	char date[32];

	out.reserve(1024);

	size_t iGoodRes = vMineResults[0].count, iTotalRes = iGoodRes;
	size_t ln = vMineResults.size();

	for(size_t i=1; i < ln; i++)
		iTotalRes += vMineResults[i].count;

	out.append("RESULT REPORT\n");
	if(iTotalRes == 0)
	{
		out.append("You haven't found any results yet.\n");
		return;
	}

	double dConnSec;
	{
		using namespace std::chrono;
		dConnSec = (double)duration_cast<seconds>(system_clock::now() - tPoolConnTime).count();
	}

	snprintf(num, sizeof(num), " (%.1f %%)\n", 100.0 * iGoodRes / iTotalRes);

	out.append("Difficulty       : ").append(std::to_string(iPoolDiff)).append(1, '\n');
	out.append("Good results     : ").append(std::to_string(iGoodRes)).append(" / ").
		append(std::to_string(iTotalRes)).append(num);

	if(iPoolCallTimes.size() != 0)
	{
		// Here we use iPoolCallTimes since it also gets reset when we disconnect
		snprintf(num, sizeof(num), "%.1f sec\n", dConnSec / iPoolCallTimes.size());
		out.append("Avg result time  : ").append(num);
	}
	out.append("Pool-side hashes : ").append(std::to_string(iPoolHashes)).append(2, '\n');
	out.append("Top 10 best results found:\n");

	for(size_t i=0; i < 10; i += 2)
	{
		snprintf(num, sizeof(num), "| %2llu | %16llu | %2llu | %16llu |\n",
			int_port(i), int_port(iTopDiff[i]), int_port(i+1), int_port(iTopDiff[i+1]));
		out.append(num);
	}

	out.append("\nError details:\n");
	if(ln > 1)
	{
		out.append("| Count | Error text                       | Last seen           |\n");
		for(size_t i=1; i < ln; i++)
		{
			snprintf(num, sizeof(num), "| %5llu | %-32.32s | %s |\n", int_port(vMineResults[i].count),
				vMineResults[i].msg.c_str(), time_format(date, sizeof(date), vMineResults[i].time));
			out.append(num);
		}
	}
	else
		out.append("Yay! No errors.\n");
}

void executor::connection_report(std::string& out)
{
	char num[128];
	char date[32];

	out.reserve(512);

	jpsock* pool = pick_pool_by_id(dev_pool_id + 1);

	out.append("CONNECTION REPORT\n");
	out.append("Pool address    : ").append(jconf::inst()->GetPoolAddress()).append(1, '\n');
	if (pool->is_running() && pool->is_logged_in())
		out.append("Connected since : ").append(time_format(date, sizeof(date), tPoolConnTime)).append(1, '\n');
	else
		out.append("Connected since : <not connected>\n");

	size_t n_calls = iPoolCallTimes.size();
	if (n_calls > 1)
	{
		//Not-really-but-good-enough median
		std::nth_element(iPoolCallTimes.begin(), iPoolCallTimes.begin() + n_calls/2, iPoolCallTimes.end());
		out.append("Pool ping time  : ").append(std::to_string(iPoolCallTimes[n_calls/2])).append(" ms\n");
	}
	else
		out.append("Pool ping time  : (n/a)\n");

	out.append("\nNetwork error log:\n");
	size_t ln = vSocketLog.size();
	if(ln > 0)
	{
		out.append("| Date                | Error text                                             |\n");
		for(size_t i=0; i < ln; i++)
		{
			snprintf(num, sizeof(num), "| %s | %-54.54s |\n",
				time_format(date, sizeof(date), vSocketLog[i].time), vSocketLog[i].msg.c_str());
			out.append(num);
		}
	}
	else
		out.append("Yay! No errors.\n");
}

void executor::print_report(ex_event_name ev)
{
	std::string out;
	switch(ev)
	{
	case EV_USR_HASHRATE:
		hashrate_report(out);
		break;

	case EV_USR_RESULTS:
		result_report(out);
		break;

	case EV_USR_CONNSTAT:
		connection_report(out);
		break;
	default:
		assert(false);
		break;
	}

	printer::inst()->print_str(out.c_str());
}

void executor::http_hashrate_report(std::string& out)
{
	char num_a[32], num_b[32], num_c[32], num_d[32];
	char buffer[4096];
	size_t nthd = pvThreads->size();

	out.reserve(4096);

	snprintf(buffer, sizeof(buffer), sHtmlCommonHeader, "Hashrate Report", "Hashrate Report");
	out.append(buffer);

	snprintf(buffer, sizeof(buffer), sHtmlHashrateBodyHigh, (unsigned int)nthd + 3);
	out.append(buffer);

	double fTotal[3] = { 0.0, 0.0, 0.0};
	for(size_t i=0; i < nthd; i++)
	{
		double fHps[3];

		fHps[0] = telem->calc_telemetry_data(2500, i);
		fHps[1] = telem->calc_telemetry_data(60000, i);
		fHps[2] = telem->calc_telemetry_data(900000, i);

		num_a[0] = num_b[0] = num_c[0] ='\0';
		hps_format(fHps[0], num_a, sizeof(num_a));
		hps_format(fHps[1], num_b, sizeof(num_b));
		hps_format(fHps[2], num_c, sizeof(num_c));

		fTotal[0] += fHps[0];
		fTotal[1] += fHps[1];
		fTotal[2] += fHps[2];

		snprintf(buffer, sizeof(buffer), sHtmlHashrateTableRow, (unsigned int)i, num_a, num_b, num_c);
		out.append(buffer);
	}

	num_a[0] = num_b[0] = num_c[0] = num_d[0] ='\0';
	hps_format(fTotal[0], num_a, sizeof(num_a));
	hps_format(fTotal[1], num_b, sizeof(num_b));
	hps_format(fTotal[2], num_c, sizeof(num_c));
	hps_format(fHighestHps, num_d, sizeof(num_d));

	snprintf(buffer, sizeof(buffer), sHtmlHashrateBodyLow, num_a, num_b, num_c, num_d);
	out.append(buffer);
}

void executor::http_result_report(std::string& out)
{
	char date[128];
	char buffer[4096];

	out.reserve(4096);

	snprintf(buffer, sizeof(buffer), sHtmlCommonHeader, "Result Report", "Result Report");
	out.append(buffer);

	size_t iGoodRes = vMineResults[0].count, iTotalRes = iGoodRes;
	size_t ln = vMineResults.size();

	for(size_t i=1; i < ln; i++)
		iTotalRes += vMineResults[i].count;

	double fGoodResPrc = 0.0;
	if(iTotalRes > 0)
		fGoodResPrc = 100.0 * iGoodRes / iTotalRes;

	double fAvgResTime = 0.0;
	if(iPoolCallTimes.size() > 0)
	{
		using namespace std::chrono;
		fAvgResTime = ((double)duration_cast<seconds>(system_clock::now() - tPoolConnTime).count())
			/ iPoolCallTimes.size();
	}

	snprintf(buffer, sizeof(buffer), sHtmlResultBodyHigh,
		iPoolDiff, iGoodRes, iTotalRes, fGoodResPrc, fAvgResTime, iPoolHashes,
		int_port(iTopDiff[0]), int_port(iTopDiff[1]), int_port(iTopDiff[2]), int_port(iTopDiff[3]),
		int_port(iTopDiff[4]), int_port(iTopDiff[5]), int_port(iTopDiff[6]), int_port(iTopDiff[7]),
		int_port(iTopDiff[8]), int_port(iTopDiff[9]));

	out.append(buffer);

	for(size_t i=1; i < vMineResults.size(); i++)
	{
		snprintf(buffer, sizeof(buffer), sHtmlResultTableRow, vMineResults[i].msg.c_str(),
			int_port(vMineResults[i].count), time_format(date, sizeof(date), vMineResults[i].time));
		out.append(buffer);
	}

	out.append(sHtmlResultBodyLow);
}

void executor::http_connection_report(std::string& out)
{
	char date[128];
	char buffer[4096];

	out.reserve(4096);

	snprintf(buffer, sizeof(buffer), sHtmlCommonHeader, "Connection Report", "Connection Report");
	out.append(buffer);

	jpsock* pool = pick_pool_by_id(dev_pool_id + 1);
	const char* cdate = "not connected";
	if (pool->is_running() && pool->is_logged_in())
		cdate = time_format(date, sizeof(date), tPoolConnTime);

	size_t n_calls = iPoolCallTimes.size();
	unsigned int ping_time = 0;
	if (n_calls > 1)
	{
		//Not-really-but-good-enough median
		std::nth_element(iPoolCallTimes.begin(), iPoolCallTimes.begin() + n_calls/2, iPoolCallTimes.end());
		ping_time = iPoolCallTimes[n_calls/2];
	}

	snprintf(buffer, sizeof(buffer), sHtmlConnectionBodyHigh,
		jconf::inst()->GetPoolAddress(),
		cdate, ping_time);
	out.append(buffer);


	for(size_t i=0; i < vSocketLog.size(); i++)
	{
		snprintf(buffer, sizeof(buffer), sHtmlConnectionTableRow,
			time_format(date, sizeof(date), vSocketLog[i].time), vSocketLog[i].msg.c_str());
		out.append(buffer);
	}

	out.append(sHtmlConnectionBodyLow);
}

void executor::http_report(ex_event_name ev)
{
	assert(pHttpString != nullptr);

	switch(ev)
	{
	case EV_HTML_HASHRATE:
		http_hashrate_report(*pHttpString);
		break;

	case EV_HTML_RESULTS:
		http_result_report(*pHttpString);
		break;

	case EV_HTML_CONNSTAT:
		http_connection_report(*pHttpString);
		break;
	default:
		assert(false);
		break;
	}

	httpReady.set_value();
}

void executor::get_http_report(ex_event_name ev_id, std::string& data)
{
	std::lock_guard<std::mutex> lck(httpMutex);

	assert(pHttpString == nullptr);
	assert(ev_id == EV_HTML_HASHRATE || ev_id == EV_HTML_RESULTS || ev_id == EV_HTML_CONNSTAT);

	pHttpString = &data;
	httpReady = std::promise<void>();
	std::future<void> ready = httpReady.get_future();

	push_event(ex_event(ev_id));

	ready.wait();
	pHttpString = nullptr;
}
