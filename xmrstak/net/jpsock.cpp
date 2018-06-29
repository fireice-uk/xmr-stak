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

#include <stdarg.h>
#include <assert.h>
#include <algorithm>
#include <chrono>

#include "jpsock.hpp"
#include "socks.hpp"
#include "socket.hpp"

#include "xmrstak/misc/executor.hpp"
#include "xmrstak/jconf.hpp"
#include "xmrstak/misc/jext.hpp"
#include "xmrstak/version.hpp"

using namespace rapidjson;

struct jpsock::call_rsp
{
	bool bHaveResponse;
	uint64_t iCallId;
	Value* pCallData;
	std::string sCallErr;
	uint64_t iMessageId;

	call_rsp(Value* val) : pCallData(val), iMessageId(0)
	{
		bHaveResponse = false;
		iCallId = 0;
		sCallErr.clear();
	}
};

typedef GenericDocument<UTF8<>, MemoryPoolAllocator<>, MemoryPoolAllocator<>> MemDocument;

/*
 *
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 * ASSUMPTION - only one calling thread. Multiple calling threads would require better
 * thread safety. The calling thread is assumed to be the executor thread.
 * If there is a reason to call the pool outside of the executor context, consider
 * doing it via an executor event.
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * Call values and allocators are for the calling thread (executor). When processing
 * a call, the recv thread will make a copy of the call response and then erase its copy.
 */

struct jpsock::opaque_private
{
	Value  oCallValue;

	MemoryPoolAllocator<> callAllocator;
	MemoryPoolAllocator<> recvAllocator;
	MemoryPoolAllocator<> parseAllocator;
	MemDocument jsonDoc;
	call_rsp oCallRsp;

	opaque_private(uint8_t* bCallMem, uint8_t* bRecvMem, uint8_t* bParseMem) :
		callAllocator(bCallMem, jpsock::iJsonMemSize),
		recvAllocator(bRecvMem, jpsock::iJsonMemSize),
		parseAllocator(bParseMem, jpsock::iJsonMemSize),
		jsonDoc(&recvAllocator, jpsock::iJsonMemSize, &parseAllocator),
		oCallRsp(nullptr)
	{
	}
};

struct jpsock::opq_json_val
{
	const Value* val;
	opq_json_val(const Value* val) : val(val) {}
};

jpsock::jpsock(size_t id, const char* sAddr, const char* sLogin, const char* sRigId, const char* sPassword, double pool_weight, bool dev_pool, bool tls, const char* tls_fp, bool nicehash) :
	net_addr(sAddr), usr_login(sLogin), usr_rigid(sRigId), usr_pass(sPassword), tls_fp(tls_fp), pool_id(id), pool_weight(pool_weight), pool(dev_pool), nicehash(nicehash),
	connect_time(0), connect_attempts(0), disconnect_time(0), quiet_close(false)
{
	sock_init();

	bJsonCallMem = (uint8_t*)malloc(iJsonMemSize);
	bJsonRecvMem = (uint8_t*)malloc(iJsonMemSize);
	bJsonParseMem = (uint8_t*)malloc(iJsonMemSize);

	prv = new opaque_private(bJsonCallMem, bJsonRecvMem, bJsonParseMem);

#ifndef CONF_NO_TLS
	if(tls)
		sck = new tls_socket(this);
	else
		sck = new plain_socket(this);
#else
	sck = new plain_socket(this);
#endif

	oRecvThd = nullptr;
	bRunning = false;
	bLoggedIn = false;
	iJobDiff = 0;

	memset(&oCurrentJob, 0, sizeof(oCurrentJob));
}

jpsock::~jpsock()
{
	delete prv;
	prv = nullptr;

	free(bJsonCallMem);
	free(bJsonRecvMem);
	free(bJsonParseMem);
}

std::string&& jpsock::get_call_error()
{
	call_error = false;
	return std::move(prv->oCallRsp.sCallErr);
}

bool jpsock::set_socket_error(const char* a)
{
	if(!bHaveSocketError)
	{
		bHaveSocketError = true;
		sSocketError.assign(a);
	}

	return false;
}

bool jpsock::set_socket_error(const char* a, const char* b)
{
	if(!bHaveSocketError)
	{
		bHaveSocketError = true;
		size_t ln_a = strlen(a);
		size_t ln_b = strlen(b);

		sSocketError.reserve(ln_a + ln_b + 2);
		sSocketError.assign(a, ln_a);
		sSocketError.append(b, ln_b);
	}

	return false;
}

bool jpsock::set_socket_error(const char* a, size_t len)
{
	if(!bHaveSocketError)
	{
		bHaveSocketError = true;
		sSocketError.assign(a, len);
	}

	return false;
}

bool jpsock::set_socket_error_strerr(const char* a)
{
	char sSockErrText[512];
	return set_socket_error(a, sock_strerror(sSockErrText, sizeof(sSockErrText)));
}

bool jpsock::set_socket_error_strerr(const char* a, int res)
{
	char sSockErrText[512];
	return set_socket_error(a, sock_gai_strerror(res, sSockErrText, sizeof(sSockErrText)));
}

void jpsock::jpsock_thread()
{
	jpsock_thd_main();

	if(!bHaveSocketError)
		set_socket_error("Socket closed.");

	executor::inst()->push_event(ex_event(std::move(sSocketError), quiet_close, pool_id));

	std::unique_lock<std::mutex> mlock(call_mutex);
	bool bWait = prv->oCallRsp.pCallData != nullptr;

	// If a call is waiting, wait a little bit before blowing it out of the water
	if(bWait)
	{
		mlock.unlock();
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		mlock.lock();
	}

	// If the call is still there send an error to end it
	bool bCallWaiting = false;
	if(prv->oCallRsp.pCallData != nullptr)
	{
		prv->oCallRsp.bHaveResponse = true;
		prv->oCallRsp.iCallId = 0;
		prv->oCallRsp.pCallData = nullptr;
		prv->oCallRsp.iMessageId = 0;
		bCallWaiting = true;
	}
	mlock.unlock();

	if(bCallWaiting)
		call_cond.notify_one();

	bLoggedIn = false;

	if(bHaveSocketError && !quiet_close)
		disconnect_time = get_timestamp();
	else
		disconnect_time = 0;

	std::unique_lock<std::mutex> lck(job_mutex);
	memset(&oCurrentJob, 0, sizeof(oCurrentJob));
	bRunning = false;
}

bool jpsock::jpsock_thd_main()
{
	if(!sck->connect())
		return false;

	executor::inst()->push_event(ex_event(EV_SOCK_READY, pool_id));

	char buf[iSockBufferSize];
	size_t datalen = 0;
	while (true)
	{
		int ret = sck->recv(buf + datalen, sizeof(buf) - datalen);

		if(ret <= 0)
			return false;

		datalen += ret;

		if (datalen >= sizeof(buf))
		{
			sck->close(false);
			return set_socket_error("RECEIVE error: data overflow");
		}

		char* lnend;
		char* lnstart = buf;
		while ((lnend = (char*)memchr(lnstart, '\n', datalen)) != nullptr)
		{
			lnend++;
			int lnlen = lnend - lnstart;

			if (!process_line(lnstart, lnlen))
			{
				sck->close(false);
				return false;
			}

			datalen -= lnlen;
			lnstart = lnend;
		}

		//Got leftover data? Move it to the front
		if (datalen > 0 && buf != lnstart)
			memmove(buf, lnstart, datalen);
	}
}

bool jpsock::process_line(char* line, size_t len)
{
	prv->jsonDoc.SetNull();
	prv->parseAllocator.Clear();
	prv->callAllocator.Clear();
	++iMessageCnt;

	/*NULL terminate the line instead of '\n', parsing will add some more NULLs*/
	line[len-1] = '\0';

	//printf("RECV: %s\n", line);

	if (prv->jsonDoc.ParseInsitu(line).HasParseError())
		return set_socket_error("PARSE error: Invalid JSON");

	if (!prv->jsonDoc.IsObject())
		return set_socket_error("PARSE error: Invalid root");

	const Value* mt;
	if (prv->jsonDoc.HasMember("method"))
	{
		mt = GetObjectMember(prv->jsonDoc, "method");

		if(!mt->IsString())
			return set_socket_error("PARSE error: Protocol error 1");

		if(strcmp(mt->GetString(), "mining.set_extranonce") == 0)
		{
			printer::inst()->print_msg(L0, "Detected buggy NiceHash pool code. Workaround engaged.");
			return true;
		}

		if(strcmp(mt->GetString(), "job") != 0)
			return set_socket_error("PARSE error: Unsupported server method ", mt->GetString());

		mt = GetObjectMember(prv->jsonDoc, "params");
		if(mt == nullptr || !mt->IsObject())
			return set_socket_error("PARSE error: Protocol error 2");

		opq_json_val v(mt);
		return process_pool_job(&v, iMessageCnt);
	}
	else
	{
		uint64_t iCallId;
		mt = GetObjectMember(prv->jsonDoc, "id");
		if (mt == nullptr || !mt->IsUint64())
			return set_socket_error("PARSE error: Protocol error 3");

		iCallId = mt->GetUint64();

		mt = GetObjectMember(prv->jsonDoc, "error");

		const char* sError = nullptr;
		size_t iErrorLen = 0;
		if (mt == nullptr || mt->IsNull())
		{
			/* If there was no error we need a result */
			if ((mt = GetObjectMember(prv->jsonDoc, "result")) == nullptr)
				return set_socket_error("PARSE error: Protocol error 7");
		}
		else
		{
			if(!mt->IsObject())
				return set_socket_error("PARSE error: Protocol error 5");

			const Value* msg = GetObjectMember(*mt, "message");

			if(msg == nullptr || !msg->IsString())
				return set_socket_error("PARSE error: Protocol error 6");

			iErrorLen = msg->GetStringLength();
			sError = msg->GetString();
		}

		std::unique_lock<std::mutex> mlock(call_mutex);
		if (prv->oCallRsp.pCallData == nullptr)
		{
			/*Server sent us a call reply without us making a call*/
			mlock.unlock();
			return set_socket_error("PARSE error: Unexpected call response");
		}

		prv->oCallRsp.bHaveResponse = true;
		prv->oCallRsp.iCallId = iCallId;
		prv->oCallRsp.iMessageId = iMessageCnt;

		if(sError != nullptr)
		{
			prv->oCallRsp.pCallData = nullptr;
			prv->oCallRsp.sCallErr.assign(sError, iErrorLen);
			call_error = true;
		}
		else
			prv->oCallRsp.pCallData->CopyFrom(*mt, prv->callAllocator);

		mlock.unlock();
		call_cond.notify_one();

		return true;
	}
}

bool jpsock::process_pool_job(const opq_json_val* params, const uint64_t messageId)
{
	std::unique_lock<std::mutex> mlock(job_mutex);
	if(messageId < iLastMessageId)
	{
		/* In the case where the processed job message id is lesser than the last
		 * processed job message id we skip the processing to avoid mining old jobs
		 */
		return true;
	}
	iLastMessageId = messageId;

	mlock.unlock();

	if (!params->val->IsObject())
		return set_socket_error("PARSE error: Job error 1");

	const Value *blob, *jobid, *target, *motd;
	jobid = GetObjectMember(*params->val, "job_id");
	blob = GetObjectMember(*params->val, "blob");
	target = GetObjectMember(*params->val, "target");
	motd = GetObjectMember(*params->val, "motd");

	if (jobid == nullptr || blob == nullptr || target == nullptr ||
		!jobid->IsString() || !blob->IsString() || !target->IsString())
	{
		return set_socket_error("PARSE error: Job error 2");
	}

	if(motd != nullptr && motd->IsString() && (motd->GetStringLength() & 0x01) == 0)
	{
		std::unique_lock<std::mutex> lck(motd_mutex);
		if(motd->GetStringLength() > 0)
		{
			pool_motd.resize(motd->GetStringLength()/2 + 1);
			if(!hex2bin(motd->GetString(), motd->GetStringLength(), (unsigned char*)&pool_motd.front()))
				pool_motd.clear();
		}
		else
			pool_motd.clear();
	}

	if (jobid->GetStringLength() >= sizeof(pool_job::sJobID)) // Note >=
		return set_socket_error("PARSE error: Job error 3");

	pool_job oPoolJob;

	const uint32_t iWorkLen = blob->GetStringLength() / 2;
	oPoolJob.iWorkLen = iWorkLen;

	if (iWorkLen > sizeof(pool_job::bWorkBlob))
		return set_socket_error("PARSE error: Invalid job length. Are you sure you are mining the correct coin?");

	if (!hex2bin(blob->GetString(), iWorkLen * 2, oPoolJob.bWorkBlob))
		return set_socket_error("PARSE error: Job error 4");

	// lock reading of oCurrentJob
	std::unique_lock<std::mutex> jobIdLock(job_mutex);
	// compare possible non equal length job id's
	if(iWorkLen == oCurrentJob.iWorkLen &&
		memcmp(oPoolJob.bWorkBlob, oCurrentJob.bWorkBlob, iWorkLen) == 0 &&
		strcmp(jobid->GetString(), oCurrentJob.sJobID) == 0
	)
	{
		return set_socket_error("Duplicate equal job detected! Please contact your pool admin.");
	}
	jobIdLock.unlock();

	memset(oPoolJob.sJobID, 0, sizeof(pool_job::sJobID));
	memcpy(oPoolJob.sJobID, jobid->GetString(), jobid->GetStringLength()); //Bounds checking at proto error 3

	size_t target_slen = target->GetStringLength();
	if(target_slen <= 8)
	{
		uint32_t iTempInt = 0;
		char sTempStr[] = "00000000"; // Little-endian CPU FTW
		memcpy(sTempStr, target->GetString(), target_slen);
		if(!hex2bin(sTempStr, 8, (unsigned char*)&iTempInt) || iTempInt == 0)
			return set_socket_error("PARSE error: Invalid target");


		oPoolJob.iTarget = t32_to_t64(iTempInt);
	}
	else if(target_slen <= 16)
	{
		oPoolJob.iTarget = 0;
		char sTempStr[] = "0000000000000000";
		memcpy(sTempStr, target->GetString(), target_slen);
		if(!hex2bin(sTempStr, 16, (unsigned char*)&oPoolJob.iTarget) || oPoolJob.iTarget == 0)
			return set_socket_error("PARSE error: Invalid target");
	}
	else
		return set_socket_error("PARSE error: Job error 5");

	iJobDiff = t64_to_diff(oPoolJob.iTarget);

	std::unique_lock<std::mutex> lck(job_mutex);
	oCurrentJob = oPoolJob;
	lck.unlock();
	// send event after current job data are updated
	executor::inst()->push_event(ex_event(oPoolJob, pool_id));

	return true;
}

bool jpsock::connect(std::string& sConnectError)
{
	ext_algo = ext_backend = ext_hashcount = ext_motd = false;
	bHaveSocketError = false;
	call_error = false;
	sSocketError.clear();
	iJobDiff = 0;
	connect_attempts++;
	connect_time = get_timestamp();

	if(sck->set_hostname(net_addr.c_str()))
	{
		bRunning = true;
		disconnect_time = 0;
		oRecvThd = new std::thread(&jpsock::jpsock_thread, this);
		return true;
	}

	disconnect_time = get_timestamp();
	sConnectError = std::move(sSocketError);
	return false;
}

void jpsock::disconnect(bool quiet)
{
	quiet_close = quiet;
	sck->close(false);

	if(oRecvThd != nullptr)
	{
		oRecvThd->join();
		delete oRecvThd;
		oRecvThd = nullptr;
	}

	sck->close(true);
	quiet_close = false;
}

bool jpsock::cmd_ret_wait(const char* sPacket, opq_json_val& poResult, uint64_t& messageId)
{
	//printf("SEND: %s\n", sPacket);

	/*Set up the call rsp for the call reply*/
	prv->oCallValue.SetNull();
	prv->callAllocator.Clear();

	std::unique_lock<std::mutex> mlock(call_mutex);
	prv->oCallRsp = call_rsp(&prv->oCallValue);
	mlock.unlock();

	if(!sck->send(sPacket))
	{
		disconnect(); //This will join the other thread;
		return false;
	}

	//Success is true if the server approves, result is true if there was no socket error
	bool bSuccess;
	mlock.lock();
	bool bResult = call_cond.wait_for(mlock, std::chrono::seconds(jconf::inst()->GetCallTimeout()),
		[&]() { return prv->oCallRsp.bHaveResponse; });

	bSuccess = prv->oCallRsp.pCallData != nullptr;
	prv->oCallRsp.pCallData = nullptr;
	mlock.unlock();

	if(bHaveSocketError)
		return false;

	//This means that there was no socket error, but the server is not taking to us
	if(!bResult)
	{
		set_socket_error("CALL error: Timeout while waiting for a reply");
		disconnect();
		return false;
	}

	if(bSuccess)
	{
		poResult.val = &prv->oCallValue;
		messageId = prv->oCallRsp.iMessageId;
	}
	return bSuccess;
}

bool jpsock::cmd_login()
{
	char cmd_buffer[1024];

	snprintf(cmd_buffer, sizeof(cmd_buffer), "{\"method\":\"login\",\"params\":{\"login\":\"%s\",\"pass\":\"%s\",\"rigid\":\"%s\",\"agent\":\"%s\"},\"id\":1}\n",
		usr_login.c_str(), usr_pass.c_str(), usr_rigid.c_str(), get_version_str().c_str());

	opq_json_val oResult(nullptr);
	uint64_t messageId = 0;

	/*Normal error conditions (failed login etc..) will end here*/
	if (!cmd_ret_wait(cmd_buffer, oResult, messageId))
		return false;

	if (!oResult.val->IsObject())
	{
		set_socket_error("PARSE error: Login protocol error 1");
		disconnect();
		return false;
	}

	const Value* id = GetObjectMember(*oResult.val, "id");
	const Value* job = GetObjectMember(*oResult.val, "job");
	const Value* ext = GetObjectMember(*oResult.val, "extensions");

	if (id == nullptr || job == nullptr || !id->IsString())
	{
		set_socket_error("PARSE error: Login protocol error 2");
		disconnect();
		return false;
	}

	if (id->GetStringLength() >= sizeof(sMinerId))
	{
		set_socket_error("PARSE error: Login protocol error 3");
		disconnect();
		return false;
	}

	memset(sMinerId, 0, sizeof(sMinerId));
	memcpy(sMinerId, id->GetString(), id->GetStringLength());

	if(ext != nullptr && ext->IsArray())
	{
		for(size_t i=0; i < ext->Size(); i++)
		{
			const Value& jextname = ext->GetArray()[i];

			if(!jextname.IsString())
				continue;

			std::string tmp(jextname.GetString());
			std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);

			if(tmp == "algo")
				ext_algo = true;
			else if(tmp == "backend")
				ext_backend = true;
			else if(tmp == "hashcount")
				ext_hashcount = true;
			else if(tmp == "motd")
				ext_motd = true;
		}
	}

	opq_json_val v(job);
	if(!process_pool_job(&v, messageId))
	{
		disconnect();
		return false;
	}

	bLoggedIn = true;
	connect_attempts = 0;

	return true;
}

bool jpsock::cmd_submit(const char* sJobId, uint32_t iNonce, const uint8_t* bResult, const char* backend_name, uint64_t backend_hashcount, uint64_t total_hashcount, xmrstak_algo algo)
{
	char cmd_buffer[1024];
	char sNonce[9];
	char sResult[65];
	/*Extensions*/
	char sAlgo[64] = {0};
	char sBackend[64] = {0};
	char sHashcount[128] = {0};

	if(ext_backend)
		snprintf(sBackend, sizeof(sBackend), ",\"backend\":\"%s\"", backend_name);

	if(ext_hashcount)
		snprintf(sHashcount, sizeof(sHashcount), ",\"hashcount\":%llu,\"hashcount_total\":%llu", int_port(backend_hashcount), int_port(total_hashcount));

	if(ext_algo)
	{
		const char* algo_name;
		switch(algo)
		{
		case cryptonight:
			algo_name = "cryptonight";
			break;
		case cryptonight_lite:
			algo_name = "cryptonight_lite";
			break;
		case cryptonight_monero:
			algo_name = "cryptonight_v7";
			break;
		case cryptonight_aeon:
			algo_name = "cryptonight_lite_v7";
			break;
		case cryptonight_stellite:
			algo_name = "cryptonight_v7_stellite";
			break;
		case cryptonight_ipbc:
			algo_name = "cryptonight_lite_v7_xor";
			break;
		case cryptonight_heavy:
			algo_name = "cryptonight_heavy";
			break;
		case cryptonight_haven:
			algo_name = "cryptonight_haven";
			break;
		case cryptonight_masari:
			algo_name = "cryptonight_masari";
			break;
		default:
			algo_name = "unknown";
			break;
		}

		snprintf(sAlgo, sizeof(sAlgo), ",\"algo\":\"%s\"", algo_name);
	}

	bin2hex((unsigned char*)&iNonce, 4, sNonce);
	sNonce[8] = '\0';

	bin2hex(bResult, 32, sResult);
	sResult[64] = '\0';

	snprintf(cmd_buffer, sizeof(cmd_buffer), "{\"method\":\"submit\",\"params\":{\"id\":\"%s\",\"job_id\":\"%s\",\"nonce\":\"%s\",\"result\":\"%s\"%s%s%s},\"id\":1}\n",
		sMinerId, sJobId, sNonce, sResult, sBackend, sHashcount, sAlgo);

	uint64_t messageId = 0;
	opq_json_val oResult(nullptr);
	return cmd_ret_wait(cmd_buffer, oResult, messageId);
}

void jpsock::save_nonce(uint32_t nonce)
{
	std::unique_lock<std::mutex> lck(job_mutex);
	oCurrentJob.iSavedNonce = nonce;
}

bool jpsock::get_current_job(pool_job& job)
{
	std::unique_lock<std::mutex> lck(job_mutex);

	if(oCurrentJob.iWorkLen == 0)
		return false;

	job = oCurrentJob;
	return true;
}

bool jpsock::get_pool_motd(std::string& strin)
{
	if(!ext_motd)
		return false;

	std::unique_lock<std::mutex> lck(motd_mutex);
	if(pool_motd.size() > 0)
	{
		strin.assign(pool_motd);
		return true;
	}

	return false;
}

inline unsigned char hf_hex2bin(char c, bool &err)
{
	if (c >= '0' && c <= '9')
		return c - '0';
	else if (c >= 'a' && c <= 'f')
		return c - 'a' + 0xA;
	else if (c >= 'A' && c <= 'F')
		return c - 'A' + 0xA;

	err = true;
	return 0;
}

bool jpsock::hex2bin(const char* in, unsigned int len, unsigned char* out)
{
	bool error = false;
	for (unsigned int i = 0; i < len; i += 2)
	{
		out[i / 2] = (hf_hex2bin(in[i], error) << 4) | hf_hex2bin(in[i + 1], error);
		if (error) return false;
	}
	return true;
}

inline char hf_bin2hex(unsigned char c)
{
	if (c <= 0x9)
		return '0' + c;
	else
		return 'a' - 0xA + c;
}

void jpsock::bin2hex(const unsigned char* in, unsigned int len, char* out)
{
	for (unsigned int i = 0; i < len; i++)
	{
		out[i * 2] = hf_bin2hex((in[i] & 0xF0) >> 4);
		out[i * 2 + 1] = hf_bin2hex(in[i] & 0x0F);
	}
}
