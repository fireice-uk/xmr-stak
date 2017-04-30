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

#include "jpsock.h"
#include "executor.h"
#include "jconf.h"

#include "rapidjson/document.h"
#include "jext.h"
#include "socks.h"
#include "socket.h"
#include "version.h"

#define AGENTID_STR XMR_STAK_NAME "/" XMR_STAK_VERSION

using namespace rapidjson;

struct jpsock::call_rsp
{
	bool bHaveResponse;
	uint64_t iCallId;
	Value* pCallData;
	std::string sCallErr;

	call_rsp(Value* val) : pCallData(val)
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

jpsock::jpsock(size_t id, bool tls) : pool_id(id)
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
	executor::inst()->push_event(ex_event(std::move(sSocketError), pool_id));

	// If a call is wating, send an error to end it
	bool bCallWaiting = false;
	std::unique_lock<std::mutex> mlock(call_mutex);
	if(prv->oCallRsp.pCallData != nullptr)
	{
		prv->oCallRsp.bHaveResponse = true;
		prv->oCallRsp.iCallId = 0;
		prv->oCallRsp.pCallData = nullptr;
		bCallWaiting = true;
	}
	mlock.unlock();

	if(bCallWaiting)
		call_cond.notify_one();

	bRunning = false;
	bLoggedIn = false;

	std::unique_lock<std::mutex>(job_mutex);
	memset(&oCurrentJob, 0, sizeof(oCurrentJob));
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

		if(strcmp(mt->GetString(), "job") != 0)
			return set_socket_error("PARSE error: Unsupported server method ", mt->GetString());

		mt = GetObjectMember(prv->jsonDoc, "params");
		if(mt == nullptr || !mt->IsObject())
			return set_socket_error("PARSE error: Protocol error 2");

		opq_json_val v(mt);
		return process_pool_job(&v);
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
		size_t iErrorLn = 0;
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

			iErrorLn = msg->GetStringLength();
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

		if(sError != nullptr)
		{
			prv->oCallRsp.pCallData = nullptr;
			prv->oCallRsp.sCallErr.assign(sError, iErrorLn);
		}
		else
			prv->oCallRsp.pCallData->CopyFrom(*mt, prv->callAllocator);

		mlock.unlock();
		call_cond.notify_one();

		return true;
	}
}

bool jpsock::process_pool_job(const opq_json_val* params)
{
	if (!params->val->IsObject())
		return set_socket_error("PARSE error: Job error 1");

	const Value * blob, *jobid, *target;
	jobid = GetObjectMember(*params->val, "job_id");
	blob = GetObjectMember(*params->val, "blob");
	target = GetObjectMember(*params->val, "target");

	if (jobid == nullptr || blob == nullptr || target == nullptr ||
		!jobid->IsString() || !blob->IsString() || !target->IsString())
	{
		return set_socket_error("PARSE error: Job error 2");
	}

	if (jobid->GetStringLength() >= sizeof(pool_job::sJobID)) // Note >=
		return set_socket_error("PARSE error: Job error 3");

	uint32_t iWorkLn = blob->GetStringLength() / 2;
	if (iWorkLn > sizeof(pool_job::bWorkBlob))
		return set_socket_error("PARSE error: Invalid job legth. Are you sure you are mining the correct coin?");

	pool_job oPoolJob;
	if (!hex2bin(blob->GetString(), iWorkLn * 2, oPoolJob.bWorkBlob))
		return set_socket_error("PARSE error: Job error 4");

	oPoolJob.iWorkLen = iWorkLn;
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

	executor::inst()->push_event(ex_event(oPoolJob, pool_id));

	std::unique_lock<std::mutex>(job_mutex);
	oCurrentJob = oPoolJob;
	return true;
}

bool jpsock::connect(const char* sAddr, std::string& sConnectError)
{
	bHaveSocketError = false;
	sSocketError.clear();
	iJobDiff = 0;

	if(sck->set_hostname(sAddr))
	{
		bRunning = true;
		oRecvThd = new std::thread(&jpsock::jpsock_thread, this);
		return true;
	}

	sConnectError = std::move(sSocketError);
	return false;
}

void jpsock::disconnect()
{
	sck->close(false);

	if(oRecvThd != nullptr)
	{
		oRecvThd->join();
		delete oRecvThd;
		oRecvThd = nullptr;
	}

	sck->close(true);
}

bool jpsock::cmd_ret_wait(const char* sPacket, opq_json_val& poResult)
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
		poResult.val = &prv->oCallValue;

	return bSuccess;
}

bool jpsock::cmd_login(const char* sLogin, const char* sPassword)
{
	char cmd_buffer[1024];

	snprintf(cmd_buffer, sizeof(cmd_buffer), "{\"method\":\"login\",\"params\":{\"login\":\"%s\",\"pass\":\"%s\",\"agent\":\"" AGENTID_STR "\"},\"id\":1}\n",
		sLogin, sPassword);

	opq_json_val oResult(nullptr);

	/*Normal error conditions (failed login etc..) will end here*/
	if (!cmd_ret_wait(cmd_buffer, oResult))
		return false;

	if (!oResult.val->IsObject())
	{
		set_socket_error("PARSE error: Login protocol error 1");
		disconnect();
		return false;
	}

	const Value* id = GetObjectMember(*oResult.val, "id");
	const Value* job = GetObjectMember(*oResult.val, "job");

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

	opq_json_val v(job);
	if(!process_pool_job(&v))
	{
		disconnect();
		return false;
	}

	bLoggedIn = true;

	return true;
}

bool jpsock::cmd_submit(const char* sJobId, uint32_t iNonce, const uint8_t* bResult)
{
	char cmd_buffer[1024];
	char sNonce[9];
	char sResult[65];

	bin2hex((unsigned char*)&iNonce, 4, sNonce);
	sNonce[8] = '\0';

	bin2hex(bResult, 32, sResult);
	sResult[64] = '\0';

	snprintf(cmd_buffer, sizeof(cmd_buffer), "{\"method\":\"submit\",\"params\":{\"id\":\"%s\",\"job_id\":\"%s\",\"nonce\":\"%s\",\"result\":\"%s\"},\"id\":1}\n",
		sMinerId, sJobId, sNonce, sResult);

	opq_json_val oResult(nullptr);
	return cmd_ret_wait(cmd_buffer, oResult);
}

bool jpsock::get_current_job(pool_job& job)
{
	std::unique_lock<std::mutex>(job_mutex);

	if(oCurrentJob.iWorkLen == 0)
		return false;

	oCurrentJob.iResumeCnt++;
	job = oCurrentJob;
	return true;
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
