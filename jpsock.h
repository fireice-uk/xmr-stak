#pragma once
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <string>

#include "msgstruct.h"

/* Our pool can have two kinds of errors:
	- Parsing or connection error
	Those are fatal errors (we drop the connection if we encounter them).
	After they are constructed from const char* strings from various places.
	(can be from read-only mem), we passs them in an exectutor message
	once the recv thread expires.
	- Call error
	This error happens when the "server says no". Usually because the job was
	outdated, or we somehow got the hash wrong. It isn't fatal.
	We parse it in-situ in the network buffer, after that we copy it to a
	std::string. Executor will move the buffer via an r-value ref.
*/
class base_socket;

class jpsock
{
public:
	jpsock(size_t id, bool tls);
	~jpsock();

	bool connect(const char* sAddr, std::string& sConnectError);
	void disconnect();

	bool cmd_login(const char* sLogin, const char* sPassword);
	bool cmd_submit(const char* sJobId, uint32_t iNonce, const uint8_t* bResult);

	static bool hex2bin(const char* in, unsigned int len, unsigned char* out);
	static void bin2hex(const unsigned char* in, unsigned int len, char* out);

	inline bool is_running() { return bRunning; }
	inline bool is_logged_in() { return bLoggedIn; }

	std::string&& get_call_error();
	bool have_sock_error() { return bHaveSocketError; }

	inline static uint64_t t32_to_t64(uint32_t t) { return 0xFFFFFFFFFFFFFFFFULL / (0xFFFFFFFFULL / ((uint64_t)t)); }
	inline static uint64_t t64_to_diff(uint64_t t) { return 0xFFFFFFFFFFFFFFFFULL / t; }
	inline static uint64_t diff_to_t64(uint64_t d) { return 0xFFFFFFFFFFFFFFFFULL / d; }

	inline uint64_t get_current_diff() { return iJobDiff; }

	bool get_current_job(pool_job& job);

	size_t pool_id;

	bool set_socket_error(const char* a);
	bool set_socket_error(const char* a, const char* b);
	bool set_socket_error(const char* a, size_t len);
	bool set_socket_error_strerr(const char* a);
	bool set_socket_error_strerr(const char* a, int res);

private:
	std::atomic<bool> bRunning;
	std::atomic<bool> bLoggedIn;

	uint8_t* bJsonRecvMem;
	uint8_t* bJsonParseMem;
	uint8_t* bJsonCallMem;

	static constexpr size_t iJsonMemSize = 4096;
	static constexpr size_t iSockBufferSize = 4096;

	struct call_rsp;
	struct opaque_private;
	struct opq_json_val;

	void jpsock_thread();
	bool jpsock_thd_main();
	bool process_line(char* line, size_t len);
	bool process_pool_job(const opq_json_val* params);
	bool cmd_ret_wait(const char* sPacket, opq_json_val& poResult);

	char sMinerId[64];
	std::atomic<uint64_t> iJobDiff;

	std::string sSocketError;
	std::atomic<bool> bHaveSocketError;

	std::mutex call_mutex;
	std::condition_variable call_cond;
	std::thread* oRecvThd;

	std::mutex job_mutex;
	pool_job oCurrentJob;

	opaque_private* prv;
	base_socket* sck;
};

