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

#include "xmrstak/jconf.hpp"
#include "executor.hpp"
#include "xmrstak/net/jpsock.hpp"

#include "telemetry.hpp"
#include "xmrstak/backend/miner_work.hpp"
#include "xmrstak/backend/globalStates.hpp"
#include "xmrstak/backend/backendConnector.hpp"
#include "xmrstak/backend/iBackend.hpp"

#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/console.hpp"
#include "xmrstak/donate-level.hpp"
#include "xmrstak/version.hpp"
#include "xmrstak/http/webdesign.hpp"

#include <thread>
#include <string>
#include <cmath>
#include <algorithm>
#include <functional>
#include <assert.h>
#include <time.h>


#ifdef _WIN32
#define strncasecmp _strnicmp
#endif // _WIN32

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
	size_t tick = 0;
	while (true)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(size_t(iTickTime)));

		push_event(ex_event(EV_PERF_TICK));

		//Eval pool choice every fourth tick
		if((tick++ & 0x03) == 0)
			push_event(ex_event(EV_EVAL_POOL_CHOICE));

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
	}
}

bool executor::get_live_pools(std::vector<jpsock*>& eval_pools, bool is_dev)
{
	size_t limit = jconf::inst()->GetGiveUpLimit();
	size_t wait = jconf::inst()->GetNetRetry();

	if(limit == 0 || is_dev) limit = (-1); //No limit = limit of 2^64-1

	size_t pool_count = 0;
	size_t over_limit = 0;
	for(jpsock& pool : pools)
	{
		if(pool.is_dev_pool() != is_dev)
			continue;

		// Only eval live pools
		size_t num, dtime;
		if(pool.get_disconnects(num, dtime))
			set_timestamp();

		if(dtime == 0 || (dtime >= wait && num <= limit))
			eval_pools.emplace_back(&pool);

		pool_count++;
		if(num > limit)
			over_limit++;
	}

	if(eval_pools.size() == 0)
	{
		if(!is_dev)
		{
			if(xmrstak::globalStates::inst().pool_id != invalid_pool_id)
			{
				printer::inst()->print_msg(L0, "All pools are dead. Idling...");
				auto work = xmrstak::miner_work();
				xmrstak::pool_data dat;
				xmrstak::globalStates::inst().switch_work(work, dat);
			}

			if(over_limit == pool_count)
			{
				printer::inst()->print_msg(L0, "All pools are over give up limit. Exiting.");
				exit(0);
			}

			return false;
		}
		else
			return get_live_pools(eval_pools, false);
	}

	return true;
}

/*
 * This event is called by the timer and whenever something relevant happens.
 * The job here is to decide if we want to connect, disconnect, or switch jobs (or do nothing)
 */
void executor::eval_pool_choice()
{
	std::vector<jpsock*> eval_pools;
	eval_pools.reserve(pools.size());

	bool dev_time = is_dev_time();
	if(!get_live_pools(eval_pools, dev_time))
		return;

	size_t running = 0;
	for(jpsock* pool : eval_pools)
	{
		if(pool->is_running())
			running++;
	}

	// Special case - if we are without a pool, connect to all find a live pool asap
	if(running == 0)
	{
		if(dev_time)
			printer::inst()->print_msg(L1, "Fast-connecting to dev pool ...");

		for(jpsock* pool : eval_pools)
		{
			if(pool->can_connect())
			{
				if(!dev_time)
					printer::inst()->print_msg(L1, "Fast-connecting to %s pool ...", pool->get_pool_addr());
				std::string error;
				if(!pool->connect(error))
					log_socket_error(pool, std::move(error));
			}
		}

		return;
	}

	std::sort(eval_pools.begin(), eval_pools.end(), [](jpsock* a, jpsock* b) { return b->get_pool_weight(true) < a->get_pool_weight(true); });
	jpsock* goal = eval_pools[0];

	if(goal->get_pool_id() != xmrstak::globalStates::inst().pool_id)
	{
		if(!goal->is_running() && goal->can_connect())
		{
			if(dev_time)
				printer::inst()->print_msg(L1, "Connecting to dev pool ...");
			else
				printer::inst()->print_msg(L1, "Connecting to %s pool ...", goal->get_pool_addr());

			std::string error;
			if(!goal->connect(error))
				log_socket_error(goal, std::move(error));
			return;
		}

		if(goal->is_logged_in())
		{
			pool_job oPoolJob;
			if(!goal->get_current_job(oPoolJob))
			{
				goal->disconnect();
				return;
			}

			size_t prev_pool_id = current_pool_id;
			current_pool_id = goal->get_pool_id();
			on_pool_have_job(current_pool_id, oPoolJob);

			jpsock* prev_pool = pick_pool_by_id(prev_pool_id);
			if(prev_pool == nullptr || (!prev_pool->is_dev_pool() && !goal->is_dev_pool()))
				reset_stats();

			if(goal->is_dev_pool() && (prev_pool != nullptr && !prev_pool->is_dev_pool()))
				last_usr_pool_id = prev_pool_id;
			else
				last_usr_pool_id = invalid_pool_id;

			return;
		}
	}
	else
	{
		/* All is good - but check if we can do better */
		std::sort(eval_pools.begin(), eval_pools.end(), [](jpsock* a, jpsock* b) { return b->get_pool_weight(false) < a->get_pool_weight(false); });
		jpsock* goal2 = eval_pools[0];

		if(goal->get_pool_id() != goal2->get_pool_id())
		{
			if(!goal2->is_running() && goal2->can_connect())
			{
				printer::inst()->print_msg(L1, "Background-connect to %s pool ...", goal2->get_pool_addr());
				std::string error;
				if(!goal2->connect(error))
					log_socket_error(goal2, std::move(error));
				return;
			}
		}
	}

	if(!dev_time)
	{
		for(jpsock& pool : pools)
		{
			if(goal->is_logged_in() && pool.is_logged_in() && pool.get_pool_id() != goal->get_pool_id())
				pool.disconnect(true);

			if(pool.is_dev_pool() && pool.is_logged_in())
				pool.disconnect(true);
		}
	}
}

void executor::log_socket_error(jpsock* pool, std::string&& sError)
{
	std::string pool_name;
	pool_name.reserve(128);
	pool_name.append("[").append(pool->get_pool_addr()).append("] ");
	sError.insert(0, pool_name);

	vSocketLog.emplace_back(std::move(sError));
	printer::inst()->print_msg(L1, "SOCKET ERROR - %s", vSocketLog.back().msg.c_str());

	push_event(ex_event(EV_EVAL_POOL_CHOICE));
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
	if(pool_id == invalid_pool_id)
		return nullptr;

	for(jpsock& pool : pools)
		if(pool.get_pool_id() == pool_id)
			return &pool;

	return nullptr;
}

void executor::on_sock_ready(size_t pool_id)
{
	jpsock* pool = pick_pool_by_id(pool_id);

	if(pool->is_dev_pool())
		printer::inst()->print_msg(L1, "Dev pool connected. Logging in...");
	else
		printer::inst()->print_msg(L1, "Pool %s connected. Logging in...", pool->get_pool_addr());

	if(!pool->cmd_login())
	{
		if(pool->have_call_error() && !pool->is_dev_pool())
		{
			std::string str = "Login error: " +  pool->get_call_error();
			log_socket_error(pool, std::move(str));
		}

		if(!pool->have_sock_error())
			pool->disconnect();
	}
}

void executor::on_sock_error(size_t pool_id, std::string&& sError, bool silent)
{
	jpsock* pool = pick_pool_by_id(pool_id);

	pool->disconnect();

	if(pool_id == current_pool_id)
		current_pool_id = invalid_pool_id;

	if(silent)
		return;

	if(!pool->is_dev_pool())
		log_socket_error(pool, std::move(sError));
	else
		printer::inst()->print_msg(L1, "Dev pool socket error - mining on user pool...");
}

void executor::on_pool_have_job(size_t pool_id, pool_job& oPoolJob)
{
	if(pool_id != current_pool_id)
		return;

	jpsock* pool = pick_pool_by_id(pool_id);

	xmrstak::miner_work oWork(oPoolJob.sJobID, oPoolJob.bWorkBlob, oPoolJob.iWorkLen, oPoolJob.iTarget, pool->is_nicehash(), pool_id);

	xmrstak::pool_data dat;
	dat.iSavedNonce = oPoolJob.iSavedNonce;
	dat.pool_id = pool_id;

	xmrstak::globalStates::inst().switch_work(oWork, dat);

	if(dat.pool_id != pool_id)
	{
		jpsock* prev_pool;
		if((prev_pool = pick_pool_by_id(dat.pool_id)) != nullptr)
			prev_pool->save_nonce(dat.iSavedNonce);
	}

	if(pool->is_dev_pool())
		return;

	if(iPoolDiff != pool->get_current_diff())
	{
		iPoolDiff = pool->get_current_diff();
		printer::inst()->print_msg(L2, "Difficulty changed. Now: %llu.", int_port(iPoolDiff));
	}

	if(dat.pool_id != pool_id)
	{
		jpsock* prev_pool;
		if(dat.pool_id != invalid_pool_id && (prev_pool = pick_pool_by_id(dat.pool_id)) != nullptr)
		{
			if(prev_pool->is_dev_pool())
				printer::inst()->print_msg(L2, "Switching back to user pool.");
			else
				printer::inst()->print_msg(L2, "Pool switched.");
		}
		else
			printer::inst()->print_msg(L2, "Pool logged in.");
	}
	else
		printer::inst()->print_msg(L3, "New block detected.");
}

void executor::on_miner_result(size_t pool_id, job_result& oResult)
{
	jpsock* pool = pick_pool_by_id(pool_id);

	const char* backend_name = xmrstak::iBackend::getName(pvThreads->at(oResult.iThreadId)->backendType);
	uint64_t backend_hashcount, total_hashcount = 0;

	backend_hashcount = pvThreads->at(oResult.iThreadId)->iHashCount.load(std::memory_order_relaxed);
	for(size_t i = 0; i < pvThreads->size(); i++)
		total_hashcount += pvThreads->at(i)->iHashCount.load(std::memory_order_relaxed);

	if(pool->is_dev_pool())
	{
		//Ignore errors silently
		if(pool->is_running() && pool->is_logged_in())
			pool->cmd_submit(oResult.sJobID, oResult.iNonce, oResult.bResult, backend_name,
			backend_hashcount, total_hashcount, oResult.algorithm
		);
		return;
	}

	if (!pool->is_running() || !pool->is_logged_in())
	{
		log_result_error("[NETWORK ERROR]");
		return;
	}

	size_t t_start = get_timestamp_ms();
	bool bResult = pool->cmd_submit(oResult.sJobID, oResult.iNonce, oResult.bResult,
		backend_name, backend_hashcount, total_hashcount, oResult.algorithm
	);
	size_t t_len = get_timestamp_ms() - t_start;

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

#ifndef _WIN32

#include <signal.h>
void disable_sigpipe()
{
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = SIG_IGN;
	sa.sa_flags = 0;
	if (sigaction(SIGPIPE, &sa, 0) == -1)
		printer::inst()->print_msg(L1, "ERROR: Call to sigaction failed!");
}

#else
inline void disable_sigpipe() {}
#endif

void executor::ex_main()
{
	disable_sigpipe();

	assert(1000 % iTickTime == 0);

	xmrstak::miner_work oWork = xmrstak::miner_work();

	// \todo collect all backend threads
	pvThreads = xmrstak::BackendConnector::thread_starter(oWork);

	if(pvThreads->size()==0)
	{
		printer::inst()->print_msg(L1, "ERROR: No miner backend enabled.");
		win_exit();
	}

	telem = new xmrstak::telemetry(pvThreads->size());

	set_timestamp();
	size_t pc = jconf::inst()->GetPoolCount();
	bool dev_tls = true;
	bool already_have_cli_pool = false;
	size_t i=0;
	for(; i < pc; i++)
	{
		jconf::pool_cfg cfg;
 		jconf::inst()->GetPoolConfig(i, cfg);
#ifdef CONF_NO_TLS
		if(cfg.tls)
		{
			printer::inst()->print_msg(L1, "ERROR: No miner was compiled without TLS support.");
			win_exit();
		}
#endif
		if(!cfg.tls) dev_tls = false;

		if(!xmrstak::params::inst().poolURL.empty() && xmrstak::params::inst().poolURL == cfg.sPoolAddr)
		{
			auto& params = xmrstak::params::inst();
			already_have_cli_pool = true;

			const char* wallet = params.poolUsername.empty() ? cfg.sWalletAddr : params.poolUsername.c_str();
			const char* rigid = params.userSetRigid ? params.poolRigid.c_str() : cfg.sRigId;
			const char* pwd = params.userSetPwd ? params.poolPasswd.c_str() : cfg.sPasswd;
			bool nicehash = cfg.nicehash || params.nicehashMode;

			pools.emplace_back(i+1, cfg.sPoolAddr, wallet, rigid, pwd, 9.9, false, params.poolUseTls, cfg.tls_fingerprint, nicehash);
		}
		else
			pools.emplace_back(i+1, cfg.sPoolAddr, cfg.sWalletAddr, cfg.sRigId, cfg.sPasswd, cfg.weight, false, cfg.tls, cfg.tls_fingerprint, cfg.nicehash);
	}

	if(!xmrstak::params::inst().poolURL.empty() && !already_have_cli_pool)
	{
		auto& params = xmrstak::params::inst();
		if(params.poolUsername.empty())
		{
			printer::inst()->print_msg(L1, "ERROR: You didn't specify the username / wallet address for %s", xmrstak::params::inst().poolURL.c_str());
			win_exit();
		}

		pools.emplace_back(i+1, params.poolURL.c_str(), params.poolUsername.c_str(), params.poolRigid.c_str(), params.poolPasswd.c_str(), 9.9, false, params.poolUseTls, "", params.nicehashMode);
	}

	switch(jconf::inst()->GetCurrentCoinSelection().GetDescription(0).GetMiningAlgo())
	{
	case cryptonight_heavy:
		if(dev_tls)
			pools.emplace_front(0, "donate.xmr-stak.net:8888", "", "", "", 0.0, true, true, "", true);
		else
			pools.emplace_front(0, "donate.xmr-stak.net:5555", "", "", "", 0.0, true, false, "", true);
		break;
	case cryptonight_monero_v8:
	case cryptonight_monero:
		if(dev_tls)
			pools.emplace_front(0, "donate.xmr-stak.net:8800", "", "", "", 0.0, true, true, "", false);
		else
			pools.emplace_front(0, "donate.xmr-stak.net:5500", "", "", "", 0.0, true, false, "", false);
		break;
	case cryptonight_ipbc:
	case cryptonight_aeon:
	case cryptonight_lite:
		if(dev_tls)
			pools.emplace_front(0, "donate.xmr-stak.net:7777", "", "", "", 0.0, true, true, "", true);
		else
			pools.emplace_front(0, "donate.xmr-stak.net:4444", "", "", "", 0.0, true, false, "", true);
		break;

	case cryptonight:
		if(dev_tls)
			pools.emplace_front(0, "donate.xmr-stak.net:6666", "", "", "", 0.0, true, true, "", false);
		else
			pools.emplace_front(0, "donate.xmr-stak.net:3333", "", "", "", 0.0, true, false, "", false);
		break;

	default:
		break;
	}

	ex_event ev;
	std::thread clock_thd(&executor::ex_clock_thd, this);

	eval_pool_choice();

	// Place the default success result at position 0, it needs to
	// be here even if our first result is a failure
	vMineResults.emplace_back();

	// If the user requested it, start the autohash printer
	if(jconf::inst()->GetVerboseLevel() >= 4)
		push_timed_event(ex_event(EV_HASHRATE_LOOP), jconf::inst()->GetAutohashTime());

	size_t cnt = 0;
	while (true)
	{
		ev = oEventQ.pop();
		switch (ev.iName)
		{
		case EV_SOCK_READY:
			on_sock_ready(ev.iPoolId);
			break;

		case EV_SOCK_ERROR:
			on_sock_error(ev.iPoolId, std::move(ev.oSocketError.sSocketError), ev.oSocketError.silent);
			break;

		case EV_POOL_HAVE_JOB:
			on_pool_have_job(ev.iPoolId, ev.oPoolJob);
			break;

		case EV_MINER_HAVE_RESULT:
			on_miner_result(ev.iPoolId, ev.oJobResult);
			break;

		case EV_EVAL_POOL_CHOICE:
			eval_pool_choice();
			break;

		case EV_GPU_RES_ERROR:
		{
			std::string err_msg = std::string(ev.oGpuError.error_str) + " GPU ID " + std::to_string(ev.oGpuError.idx);
			printer::inst()->print_msg(L0, err_msg.c_str());
			log_result_error(std::move(err_msg));
			break;
		}

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
					fTelem = telem->calc_telemetry_data(10000, i);
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
		case EV_HTML_JSON:
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
		snprintf(buf, l, " %6.1f", h);
		return buf;
	}
	else
		return "   (na)";
}

bool executor::motd_filter_console(std::string& motd)
{
	if(motd.size() > motd_max_length)
		return false;

	motd.erase(std::remove_if(motd.begin(), motd.end(), [](int chr)->bool { return !((chr >= 0x20 && chr <= 0x7e) || chr == '\n');}), motd.end());
	return motd.size() > 0;
}

bool executor::motd_filter_web(std::string& motd)
{
	if(!motd_filter_console(motd))
		return false;

	std::string tmp;
	tmp.reserve(motd.size() + 128);

	for(size_t i=0; i < motd.size(); i++)
	{
		char c = motd[i];
		switch(c)
		{
		case '&':
			tmp.append("&amp;");
			break;
		case '"':
			tmp.append("&quot;");
			break;
		case '\'':
			tmp.append("&#039");
			break;
		case '<':
			tmp.append("&lt;");
			break;
		case '>':
			tmp.append("&gt;");
			break;
		case '\n':
			tmp.append("<br>");
			break;
		default:
			tmp.append(1, c);
			break;
		}
	}

	motd = std::move(tmp);
	return true;
}

void executor::hashrate_report(std::string& out)
{
	out.reserve(2048 + pvThreads->size() * 64);

	if(jconf::inst()->PrintMotd())
	{
		std::string motd;
		for(jpsock& pool : pools)
		{
			motd.empty();
			if(pool.get_pool_motd(motd) && motd_filter_console(motd))
			{
				out.append("Message from ").append(pool.get_pool_addr()).append(":\n");
				out.append(motd).append("\n");
				out.append("-----------------------------------------------------\n");
			}
		}
	}

	char num[32];
	double fTotal[3] = { 0.0, 0.0, 0.0};

	for( uint32_t b = 0; b < 4u; ++b)
	{
		std::vector<xmrstak::iBackend*> backEnds;
		std::copy_if(pvThreads->begin(), pvThreads->end(), std::back_inserter(backEnds),
			[&](xmrstak::iBackend* backend)
			{
				return backend->backendType == b;
			}
		);

		size_t nthd = backEnds.size();
		if(nthd != 0)
		{
			size_t i;
			auto bType = static_cast<xmrstak::iBackend::BackendType>(b);
			std::string name(xmrstak::iBackend::getName(bType));
			std::transform(name.begin(), name.end(), name.begin(), ::toupper);

			out.append("HASHRATE REPORT - ").append(name).append("\n");
			out.append("| ID |    10s |    60s |    15m |");
			if(nthd != 1)
				out.append(" ID |    10s |    60s |    15m |\n");
			else
				out.append(1, '\n');

			double fTotalCur[3] = { 0.0, 0.0, 0.0};
			for (i = 0; i < nthd; i++)
			{
				double fHps[3];

				uint32_t tid = backEnds[i]->iThreadNo;
				fHps[0] = telem->calc_telemetry_data(10000, tid);
				fHps[1] = telem->calc_telemetry_data(60000, tid);
				fHps[2] = telem->calc_telemetry_data(900000, tid);

				snprintf(num, sizeof(num), "| %2u |", (unsigned int)i);
				out.append(num);
				out.append(hps_format(fHps[0], num, sizeof(num))).append(" |");
				out.append(hps_format(fHps[1], num, sizeof(num))).append(" |");
				out.append(hps_format(fHps[2], num, sizeof(num))).append(1, ' ');

				fTotal[0] += (std::isnormal(fHps[0])) ? fHps[0] : 0.0;
				fTotal[1] += (std::isnormal(fHps[1])) ? fHps[1] : 0.0;
				fTotal[2] += (std::isnormal(fHps[2])) ? fHps[2] : 0.0;

				fTotalCur[0] += (std::isnormal(fHps[0])) ? fHps[0] : 0.0;
				fTotalCur[1] += (std::isnormal(fHps[1])) ? fHps[1] : 0.0;
				fTotalCur[2] += (std::isnormal(fHps[2])) ? fHps[2] : 0.0;

				if((i & 0x1) == 1) //Odd i's
					out.append("|\n");
			}

			if((i & 0x1) == 1) //We had odd number of threads
				out.append("|\n");

			out.append("Totals (").append(name).append("): ");
			out.append(hps_format(fTotalCur[0], num, sizeof(num)));
			out.append(hps_format(fTotalCur[1], num, sizeof(num)));
			out.append(hps_format(fTotalCur[2], num, sizeof(num)));
			out.append(" H/s\n");

			out.append("-----------------------------------------------------------------\n");
		}
	}

	out.append("Totals (ALL):  ");
	out.append(hps_format(fTotal[0], num, sizeof(num)));
	out.append(hps_format(fTotal[1], num, sizeof(num)));
	out.append(hps_format(fTotal[2], num, sizeof(num)));
	out.append(" H/s\nHighest: ");
	out.append(hps_format(fHighestHps, num, sizeof(num)));
	out.append(" H/s\n");
	out.append("-----------------------------------------------------------------\n");
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

	jpsock* pool = pick_pool_by_id(current_pool_id);
	if(pool != nullptr && pool->is_dev_pool())
		pool = pick_pool_by_id(last_usr_pool_id);

	out.append("CONNECTION REPORT\n");
	out.append("Pool address    : ").append(pool != nullptr ? pool->get_pool_addr() : "<not connected>").append(1, '\n');
	if(pool != nullptr && pool->is_running() && pool->is_logged_in())
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

	snprintf(buffer, sizeof(buffer), sHtmlCommonHeader, "Hashrate Report", ver_html, "Hashrate Report");
	out.append(buffer);

	bool have_motd = false;
	if(jconf::inst()->PrintMotd())
	{
		std::string motd;
		for(jpsock& pool : pools)
		{
			motd.empty();
			if(pool.get_pool_motd(motd) && motd_filter_web(motd))
			{
				if(!have_motd)
				{
					out.append(sHtmlMotdBoxStart);
					have_motd = true;
				}

				snprintf(buffer, sizeof(buffer), sHtmlMotdEntry, pool.get_pool_addr(), motd.c_str());
				out.append(buffer);
			}
		}
	}

	if(have_motd)
		out.append(sHtmlMotdBoxEnd);

	snprintf(buffer, sizeof(buffer), sHtmlHashrateBodyHigh, (unsigned int)nthd + 3);
	out.append(buffer);

	double fTotal[3] = { 0.0, 0.0, 0.0};
	for(size_t i=0; i < nthd; i++)
	{
		double fHps[3];

		fHps[0] = telem->calc_telemetry_data(10000, i);
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

	snprintf(buffer, sizeof(buffer), sHtmlCommonHeader, "Result Report", ver_html,  "Result Report");
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

	snprintf(buffer, sizeof(buffer), sHtmlCommonHeader, "Connection Report", ver_html,  "Connection Report");
	out.append(buffer);

	jpsock* pool = pick_pool_by_id(current_pool_id);
	if(pool != nullptr && pool->is_dev_pool())
		pool = pick_pool_by_id(last_usr_pool_id);

	const char* cdate = "not connected";
	if (pool != nullptr && pool->is_running() && pool->is_logged_in())
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
		pool != nullptr ? pool->get_pool_addr() : "not connected",
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

inline const char* hps_format_json(double h, char* buf, size_t l)
{
	if(std::isnormal(h) || h == 0.0)
	{
		snprintf(buf, l, "%.1f", h);
		return buf;
	}
	else
		return "null";
}

void executor::http_json_report(std::string& out)
{
	const char *a, *b, *c;
	char num_a[32], num_b[32], num_c[32];
	char hr_buffer[64];
	std::string hr_thds, res_error, cn_error;

	size_t nthd = pvThreads->size();
	double fTotal[3] = { 0.0, 0.0, 0.0};
	hr_thds.reserve(nthd * 32);

	for(size_t i=0; i < nthd; i++)
	{
		if(i != 0) hr_thds.append(1, ',');

		double fHps[3];
		fHps[0] = telem->calc_telemetry_data(10000, i);
		fHps[1] = telem->calc_telemetry_data(60000, i);
		fHps[2] = telem->calc_telemetry_data(900000, i);

		fTotal[0] += fHps[0];
		fTotal[1] += fHps[1];
		fTotal[2] += fHps[2];

		a = hps_format_json(fHps[0], num_a, sizeof(num_a));
		b = hps_format_json(fHps[1], num_b, sizeof(num_b));
		c = hps_format_json(fHps[2], num_c, sizeof(num_c));
		snprintf(hr_buffer, sizeof(hr_buffer), sJsonApiThdHashrate, a, b, c);
		hr_thds.append(hr_buffer);
	}

	a = hps_format_json(fTotal[0], num_a, sizeof(num_a));
	b = hps_format_json(fTotal[1], num_b, sizeof(num_b));
	c = hps_format_json(fTotal[2], num_c, sizeof(num_c));
	snprintf(hr_buffer, sizeof(hr_buffer), sJsonApiThdHashrate, a, b, c);

	a = hps_format_json(fHighestHps, num_a, sizeof(num_a));

	size_t iGoodRes = vMineResults[0].count, iTotalRes = iGoodRes;
	size_t ln = vMineResults.size();

	for(size_t i=1; i < ln; i++)
		iTotalRes += vMineResults[i].count;

	jpsock* pool = pick_pool_by_id(current_pool_id);
	if(pool != nullptr && pool->is_dev_pool())
		pool = pick_pool_by_id(last_usr_pool_id);

	size_t iConnSec = 0;
	if(pool != nullptr && pool->is_running() && pool->is_logged_in())
	{
		using namespace std::chrono;
		iConnSec = duration_cast<seconds>(system_clock::now() - tPoolConnTime).count();
	}

	double fAvgResTime = 0.0;
	if(iPoolCallTimes.size() > 0)
		fAvgResTime = double(iConnSec) / iPoolCallTimes.size();

	char buffer[2048];
	res_error.reserve((vMineResults.size() - 1) * 128);
	for(size_t i=1; i < vMineResults.size(); i++)
	{
		using namespace std::chrono;
		if(i != 1) res_error.append(1, ',');

		snprintf(buffer, sizeof(buffer), sJsonApiResultError, int_port(vMineResults[i].count),
			int_port(duration_cast<seconds>(vMineResults[i].time.time_since_epoch()).count()),
			vMineResults[i].msg.c_str());
		res_error.append(buffer);
	}

	size_t n_calls = iPoolCallTimes.size();
	size_t iPoolPing = 0;
	if (n_calls > 1)
	{
		//Not-really-but-good-enough median
		std::nth_element(iPoolCallTimes.begin(), iPoolCallTimes.begin() + n_calls/2, iPoolCallTimes.end());
		iPoolPing = iPoolCallTimes[n_calls/2];
	}

	cn_error.reserve(vSocketLog.size() * 256);
	for(size_t i=0; i < vSocketLog.size(); i++)
	{
		using namespace std::chrono;
		if(i != 0) cn_error.append(1, ',');

		snprintf(buffer, sizeof(buffer), sJsonApiConnectionError,
			int_port(duration_cast<seconds>(vSocketLog[i].time.time_since_epoch()).count()),
			vSocketLog[i].msg.c_str());
		cn_error.append(buffer);
	}

	size_t bb_size = 2048 + hr_thds.size() + res_error.size() + cn_error.size();
	std::unique_ptr<char[]> bigbuf( new char[ bb_size ] );

	int bb_len = snprintf(bigbuf.get(), bb_size, sJsonApiFormat,
		get_version_str().c_str(), hr_thds.c_str(), hr_buffer, a,
		int_port(iPoolDiff), int_port(iGoodRes), int_port(iTotalRes), fAvgResTime, int_port(iPoolHashes),
		int_port(iTopDiff[0]), int_port(iTopDiff[1]), int_port(iTopDiff[2]), int_port(iTopDiff[3]), int_port(iTopDiff[4]),
		int_port(iTopDiff[5]), int_port(iTopDiff[6]), int_port(iTopDiff[7]), int_port(iTopDiff[8]), int_port(iTopDiff[9]),
		res_error.c_str(), pool != nullptr ? pool->get_pool_addr() : "not connected", int_port(iConnSec), int_port(iPoolPing), cn_error.c_str());

	out = std::string(bigbuf.get(), bigbuf.get() + bb_len);
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

	case EV_HTML_JSON:
		http_json_report(*pHttpString);
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
	assert(ev_id == EV_HTML_HASHRATE || ev_id == EV_HTML_RESULTS
		|| ev_id == EV_HTML_CONNSTAT || ev_id == EV_HTML_JSON);

	pHttpString = &data;
	httpReady = std::promise<void>();
	std::future<void> ready = httpReady.get_future();

	push_event(ex_event(ev_id));

	ready.wait();
	pHttpString = nullptr;
}
