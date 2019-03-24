#pragma once

#include "msgstruct.hpp"
#include "xmrstak/backend/iBackend.hpp"
#include "xmrstak/jconf.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

/* Our pool can have two kinds of errors:
	- Parsing or connection error
	Those are fatal errors (we drop the connection if we encounter them).
	After they are constructed from const char* strings from various places.
	(can be from read-only mem), we pass them in an executor message
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
	jpsock(size_t id, const char* sAddr, const char* sLogin, const char* sRigId, const char* sPassword, double pool_weight, bool dev_pool, bool tls, const char* tls_fp, bool nicehash);
	~jpsock();

	bool connect(std::string& sConnectError);
	void disconnect(bool quiet = false);

	bool cmd_login();
	bool cmd_submit(const char* sJobId, uint32_t iNonce, const uint8_t* bResult, const char* backend_name, uint64_t backend_hashcount, uint64_t total_hashcount, const xmrstak_algo& algo);

	static bool hex2bin(const char* in, unsigned int len, unsigned char* out);
	static void bin2hex(const unsigned char* in, unsigned int len, char* out);

	inline double get_pool_weight(bool gross_weight)
	{
		double ret = pool_weight;
		if(gross_weight && bRunning)
			ret += 10.0;
		if(gross_weight && bLoggedIn)
			ret += 10.0;
		return ret;
	}

	inline size_t can_connect() { return get_timestamp() != connect_time; }
	inline bool is_running() { return bRunning; }
	inline bool is_logged_in() { return bLoggedIn; }
	inline bool is_dev_pool() { return pool; }
	inline size_t get_pool_id() { return pool_id; }
	inline bool get_disconnects(size_t& att, size_t& time)
	{
		att = connect_attempts;
		time = disconnect_time != 0 ? get_timestamp() - disconnect_time + 1 : 0;
		return pool && usr_login[0];
	}
	inline const char* get_pool_addr() { return net_addr.c_str(); }
	inline const char* get_tls_fp() { return tls_fp.c_str(); }
	inline const char* get_rigid() { return usr_rigid.c_str(); }
	inline bool is_nicehash() { return nicehash; }

	bool get_pool_motd(std::string& strin);

	std::string&& get_call_error();
	bool have_call_error() { return call_error; }
	bool have_sock_error() { return bHaveSocketError; }
	inline uint64_t get_current_diff() { return iJobDiff; }

	void save_nonce(uint32_t nonce);
	bool get_current_job(pool_job& job);

	bool set_socket_error(const char* a);
	bool set_socket_error(const char* a, const char* b);
	bool set_socket_error(const char* a, size_t len);
	bool set_socket_error_strerr(const char* a);
	bool set_socket_error_strerr(const char* a, int res);

  private:
	std::string net_addr;
	std::string usr_login;
	std::string usr_rigid;
	std::string usr_pass;
	std::string tls_fp;

	size_t pool_id;
	double pool_weight;
	bool pool;
	bool nicehash;

	bool ext_algo = false;
	bool ext_backend = false;
	bool ext_hashcount = false;
	bool ext_motd = false;

	std::string pool_motd;
	std::mutex motd_mutex;

	size_t connect_time = 0;
	std::atomic<size_t> connect_attempts;
	std::atomic<size_t> disconnect_time;

	std::atomic<bool> bRunning;
	std::atomic<bool> bLoggedIn;
	std::atomic<bool> quiet_close;
	std::atomic<bool> call_error;

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
	bool process_pool_job(const opq_json_val* params, const uint64_t messageId);
	bool cmd_ret_wait(const char* sPacket, opq_json_val& poResult, uint64_t& messageId);

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

	uint64_t iMessageCnt = 0;
	uint64_t iLastMessageId = 0;
};
