#pragma once

struct MHD_Daemon;
struct MHD_Connection;

class httpd
{
public:
	static httpd* inst()
	{
		if (oInst == nullptr) oInst = new httpd;
		return oInst;
	};

	bool start_daemon();

private:
	httpd();
	static httpd* oInst;

	static int req_handler(void * cls,
	        MHD_Connection* connection,
	        const char* url,
	        const char* method,
	        const char* version,
	        const char* upload_data,
	        size_t* upload_data_size,
	        void ** ptr);

	MHD_Daemon *d;
};
