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

executor* executor::oInst = NULL;

executor::executor()
{
	my_thd = nullptr;
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

		if(iReconnectCountdown != 0)
		{
			iReconnectCountdown--;
			if(iReconnectCountdown == 0)
				push_event(ex_event(EV_RECONNECT, usr_pool_id));
		}

		if(iDevDisconnectCountdown != 0)
		{
			iDevDisconnectCountdown--;
			if(iDevDisconnectCountdown == 0)
				push_event(ex_event(EV_DEV_POOL_EXIT));
		}

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
	long long unsigned int rt = jconf::inst()->GetNetRetry();
	printer::inst()->print_msg(L1, "Pool connection lost. Waiting %lld s before retry.", rt);

	auto work = minethd::miner_work();
	minethd::switch_work(work);

	iReconnectCountdown = sec_to_ticks(rt);
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
		reset_stats();
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
		oPoolJob.iWorkLen, oPoolJob.iResumeCnt, oPoolJob.iTarget, pool_id);

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

	if(bResult)
	{
		uint64_t* targets = (uint64_t*)oResult.bResult;
		log_result_ok(jpsock::t64_to_diff(targets[3]));
		printer::inst()->print_msg(L3, "Result accepted by the pool.");

		iPoolCallTime += t_len;
		iPoolCalls++;
	}
	else
	{
		if(!pool->have_sock_error())
		{
			printer::inst()->print_msg(L3, "Result rejected by the pool.");
			log_result_error(pool->get_call_error());
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
		if(!pool->connect("donate.xmr-stak.net:3333", error))
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
			oPoolJob.iWorkLen, oPoolJob.iResumeCnt, oPoolJob.iTarget, pool_id);

		minethd::switch_work(oWork);

		if(dev_pool->is_running())
			iDevDisconnectCountdown = sec_to_ticks(5);
	}
}

void executor::ex_main()
{
	assert(1000 % iTickTime == 0);

	iReconnectCountdown = 0;
	iDevDisconnectCountdown = 0;

	minethd::miner_work oWork = minethd::miner_work();
	pvThreads = minethd::thread_starter(oWork);
	telem = new telemetry(pvThreads->size());

	current_pool_id = usr_pool_id;
	usr_pool = new jpsock(usr_pool_id);
	dev_pool = new jpsock(dev_pool_id);

	ex_event ev;
	std::thread clock_thd(&executor::ex_clock_thd, this);

	//This will connect us to the pool for the first time
	push_event(ex_event(EV_RECONNECT, usr_pool_id));

	// Place the default success result at postion 0, it needs to
	// be here even if our first result is a failure
	vMineResults.emplace_back();

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
				for (i = 0; i < pvThreads->size(); i++)
					fHps += telem->calc_telemetry_data(2500, i);

				if(fHighestHps < fHps)
					fHighestHps = fHps;
			}
		break;

		case EV_USR_HASHRATE:
			hashrate_report();
			break;

		case EV_USR_RESULTS:
			result_report();
			break;

		case EV_USR_CONNSTAT:
			connection_report();
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
	if(std::isnormal(h))
	{
		snprintf(buf, l, " %03.1f", h);
		return buf;
	}
	else if(h == 0.0) //Zero is not normal but we want it
		return "  0.0";
	else
		return " (na)";
}

void executor::hashrate_report()
{
	std::string output;
	char num[32];
	size_t nthd = pvThreads->size();

	output.reserve(256 + nthd * 64);

	double fTotal[3] = { 0.0, 0.0, 0.0};
	size_t i;

	output.append("HASHRATE REPORT\n");
	output.append("| ID | 2.5s |  60s |  15m |");
	if(nthd != 1)
		output.append(" ID | 2.5s |  60s |  15m |\n");
	else
		output.append(1, '\n');

	for (i = 0; i < nthd; i++)
	{
		double fHps[3];

		fHps[0] = telem->calc_telemetry_data(2500, i);
		fHps[1] = telem->calc_telemetry_data(60000, i);
		fHps[2] = telem->calc_telemetry_data(900000, i);

		snprintf(num, sizeof(num), "| %2u |", (unsigned int)i);
		output.append(num);
		output.append(hps_format(fHps[0], num, sizeof(num))).append(" |");
		output.append(hps_format(fHps[1], num, sizeof(num))).append(" |");
		output.append(hps_format(fHps[2], num, sizeof(num))).append(1, ' ');

		fTotal[0] += fHps[0];
		fTotal[1] += fHps[1];
		fTotal[2] += fHps[2];

		if((i & 0x1) == 1) //Odd i's
			output.append("|\n");
	}

	if((i & 0x1) == 1) //We had odd number of threads
		output.append("|\n");

	if(nthd != 1)
		output.append("-----------------------------------------------------\n");
	else
		output.append("---------------------------\n");

	output.append("Totals:  ");
	output.append(hps_format(fTotal[0], num, sizeof(num)));
	output.append(hps_format(fTotal[1], num, sizeof(num)));
	output.append(hps_format(fTotal[2], num, sizeof(num)));
	output.append(" H/s\nHighest: ");
	output.append(hps_format(fHighestHps, num, sizeof(num)));
	output.append(" H/s\n");

	fputs(output.c_str(), stdout);
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

void executor::result_report()
{
	char num[128];
	char date[32];

	std::string out;
	out.reserve(1024);

	size_t iGoodRes = vMineResults[0].count, iTotalRes = iGoodRes;
	size_t ln = vMineResults.size();

	for(size_t i=1; i < ln; i++)
		iTotalRes += vMineResults[i].count;

	out.append("RESULT REPORT\n");
	if(iTotalRes == 0)
	{
		out.append("You haven't found any results yet.\n");
		fputs(out.c_str(), stdout);
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
	snprintf(num, sizeof(num), "%.1f sec\n", dConnSec / iTotalRes);
	out.append("Avg result time  : ").append(num);
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
		out.append("| Count |                       Error text |           Last seen |\n");
		for(size_t i=1; i < ln; i++)
		{
			snprintf(num, sizeof(num), "| %5llu | %-32.32s | %s |\n", int_port(vMineResults[i].count),
				vMineResults[i].msg.c_str(), time_format(date, sizeof(date), vMineResults[i].time));
			out.append(num);
		}
	}
	else
		out.append("Yay! No errors.\n");

	fputs(out.c_str(), stdout);
}

void executor::connection_report()
{
	char num[128];
	char date[32];

	std::string out;
	out.reserve(512);

	jpsock* pool = pick_pool_by_id(dev_pool_id + 1);

	out.append("CONNECTION REPORT\n");
	if (pool->is_running() && pool->is_logged_in())
		out.append("Connected since : ").append(time_format(date, sizeof(date), tPoolConnTime)).append(1, '\n');
	else
		out.append("Connected since : <not connected>\n");

	if (iPoolCalls > 0)
		out.append("Pool ping time  : ").append(std::to_string(iPoolCallTime / iPoolCalls)).append(" ms\n");
	else
		out.append("Pool ping time  : (n/a)\n");

	out.append("\nNetwork error log:\n");
	size_t ln = vSocketLog.size();
	if(ln > 0)
	{
		out.append("| Date                |                                                       Error text |\n");
		for(size_t i=0; i < ln; i++)
		{
			snprintf(num, sizeof(num), "| %s | %-64.64s |\n",
				time_format(date, sizeof(date), vSocketLog[i].time), vSocketLog[i].msg.c_str());
			out.append(num);
		}
	}
	else
		out.append("Yay! No errors.\n");

	fputs(out.c_str(), stdout);
}
