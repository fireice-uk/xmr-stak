#pragma once

#include <stdlib.h>
#include <microhttpd.h>
#if MHD_VERSION >= 0x00097002

#define MHD_RESULT enum MHD_Result

#else

#define MHD_RESULT int

#endif

struct MHD_Daemon;
struct MHD_Connection;

class httpd
{
  public:
	static httpd* inst()
	{
		if(oInst == nullptr)
			oInst = new httpd;
		return oInst;
	};

	bool start_daemon();

  private:
	httpd();
	static httpd* oInst;

    static MHD_RESULT  req_handler(void* cls,
		MHD_Connection* connection,
		const char* url,
		const char* method,
		const char* version,
		const char* upload_data,
		size_t* upload_data_size,
		void** ptr);

	MHD_Daemon* d;
};
