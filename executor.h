#pragma once
#include "thdq.hpp"
#include "msgstruct.h"
#include <atomic>
#include <array>
#include <list>
#include <future>

class jpsock;
class minethd;
class telemetry;

class executor
{
public:
	static executor* inst()
	{
		if (oInst == nullptr) oInst = new executor;
		return oInst;
	};

	void ex_start(bool daemon) { daemon ? ex_main() : std::thread(&executor::ex_main, this).detach(); }

	void get_http_report(ex_event_name ev_id, std::string& data);

	inline void push_event(ex_event&& ev) { oEventQ.push(std::move(ev)); }
	void push_timed_event(ex_event&& ev, size_t sec);

	constexpr static size_t invalid_pool_id = 0;
	constexpr static size_t dev_pool_id = 1;
	constexpr static size_t usr_pool_id = 2;

private:
	struct timed_event
	{
		ex_event event;
		size_t ticks_left;

		timed_event(ex_event&& ev, size_t ticks) : event(std::move(ev)), ticks_left(ticks) {}
	};

	// In miliseconds, has to divide a second (1000ms) into an integer number
	constexpr static size_t iTickTime = 500;

	// Dev donation time period in seconds. 100 minutes by default.
	// We will divide up this period according to the config setting
	constexpr static size_t iDevDonatePeriod = 100 * 60;

	std::list<timed_event> lTimedEvents;
	std::mutex timed_event_mutex;
	thdq<ex_event> oEventQ;

	telemetry* telem;
	std::vector<minethd*>* pvThreads;

	size_t current_pool_id;

	jpsock* usr_pool;
	jpsock* dev_pool;

	jpsock* pick_pool_by_id(size_t pool_id);

	bool is_dev_time;

	executor();
	static executor* oInst;

	void ex_main();

	void ex_clock_thd();
	void pool_connect(jpsock* pool);

	void hashrate_report(std::string& out);
	void result_report(std::string& out);
	void connection_report(std::string& out);

	void http_hashrate_report(std::string& out);
	void http_result_report(std::string& out);
	void http_connection_report(std::string& out);
	void http_json_report(std::string& out);

	void http_report(ex_event_name ev);
	void print_report(ex_event_name ev);

	std::string* pHttpString = nullptr;
	std::promise<void> httpReady;
	std::mutex httpMutex;

	size_t iReconnectAttempts = 0;

	struct sck_error_log
	{
		std::chrono::system_clock::time_point time;
		std::string msg;

		sck_error_log(std::string&& err) : msg(std::move(err))
		{
			time = std::chrono::system_clock::now();
		}
	};
	std::vector<sck_error_log> vSocketLog;

	// Element zero is always the success element.
	// Keep in mind that this is a tally and not a log like above
	struct result_tally
	{
		std::chrono::system_clock::time_point time;
		std::string msg;
		size_t count;

		result_tally() : msg("[OK]"), count(0)
		{
			time = std::chrono::system_clock::now();
		}

		result_tally(std::string&& err) : msg(std::move(err)), count(1)
		{
			time = std::chrono::system_clock::now();
		}

		void increment()
		{
			count++;
			time = std::chrono::system_clock::now();
		}

		bool compare(std::string& err)
		{
			if(msg == err)
				return true;
			else
				return false;
		}
	};
	std::vector<result_tally> vMineResults;

	//More result statistics
	std::array<size_t, 10> iTopDiff { { } }; //Initialize to zero

	std::chrono::system_clock::time_point tPoolConnTime;
	size_t iPoolHashes = 0;
	uint64_t iPoolDiff = 0;

	// Set it to 16 bit so that we can just let it grow
	// Maximum realistic growth rate - 5MB / month
	std::vector<uint16_t> iPoolCallTimes;

	//Those stats are reset if we disconnect
	inline void reset_stats()
	{
		iPoolCallTimes.clear();
		tPoolConnTime = std::chrono::system_clock::now();
		iPoolHashes = 0;
		iPoolDiff = 0;
	}

	double fHighestHps = 0.0;

	void log_socket_error(std::string&& sError);
	void log_result_error(std::string&& sError);
	void log_result_ok(uint64_t iActualDiff);

	void sched_reconnect();

	void on_sock_ready(size_t pool_id);
	void on_sock_error(size_t pool_id, std::string&& sError);
	void on_pool_have_job(size_t pool_id, pool_job& oPoolJob);
	void on_miner_result(size_t pool_id, job_result& oResult);
	void on_reconnect(size_t pool_id);
	void on_switch_pool(size_t pool_id);

	inline size_t sec_to_ticks(size_t sec) { return sec * (1000 / iTickTime); }
};

