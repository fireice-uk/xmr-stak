#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>

#include "msgstruct.h"
#include "httpd.h"
#include "console.h"
#include "executor.h"
#include "jconf.h"

#ifdef _WIN32
#include "libmicrohttpd/microhttpd.h"
#define strcasecmp _stricmp
#else
#include <microhttpd.h>
#endif // _WIN32

httpd* httpd::oInst = nullptr;

httpd::httpd()
{

}

int httpd::req_handler(void * cls,
	        MHD_Connection* connection,
	        const char* url,
	        const char* method,
	        const char* version,
	        const char* upload_data,
	        size_t* upload_data_size,
	        void ** ptr)
{
	struct MHD_Response * rsp;

	if (strcmp(method, "GET") != 0)
		return MHD_NO;

	*ptr = nullptr;

	std::string str;
	if(strcasecmp(url, "/h") == 0 || strcasecmp(url, "/hashrate") == 0)
	{
		str.append("<html><head><title>Hashrate Report</title></head><body><pre>");
		executor::inst()->get_http_report(EV_HTML_HASHRATE, str);
		str.append("</pre></body></html>");

		rsp = MHD_create_response_from_buffer(str.size(), (void*)str.c_str(), MHD_RESPMEM_MUST_COPY);
	}
	else if(strcasecmp(url, "/c") == 0 || strcasecmp(url, "/connection") == 0)
	{
		str.append("<html><head><title>Connection Report</title></head><body><pre>");
		executor::inst()->get_http_report(EV_HTML_CONNSTAT, str);
		str.append("</pre></body></html>");

		rsp = MHD_create_response_from_buffer(str.size(), (void*)str.c_str(), MHD_RESPMEM_MUST_COPY);
	}
	else if(strcasecmp(url, "/r") == 0 || strcasecmp(url, "/results") == 0)
	{
		str.append("<html><head><title>Results Report</title></head><body><pre>");
		executor::inst()->get_http_report(EV_HTML_RESULTS, str);
		str.append("</pre></body></html>");

		rsp = MHD_create_response_from_buffer(str.size(), (void*)str.c_str(), MHD_RESPMEM_MUST_COPY);
	}
	else
	{
		char buffer[1024];
		snprintf(buffer, sizeof(buffer), "<html><head><title>Error</title></head><body>"
			"<pre>Unkown url %s - please use /h, /r or /c as url</pre></body></html>", url);

		rsp = MHD_create_response_from_buffer(strlen(buffer),
		(void*)buffer, MHD_RESPMEM_MUST_COPY);
	}

	int ret = MHD_queue_response(connection, MHD_HTTP_OK, rsp);
	MHD_destroy_response(rsp);
	return ret;
}

bool httpd::start_daemon()
{
	d = MHD_start_daemon(MHD_USE_THREAD_PER_CONNECTION,
		jconf::inst()->GetHttpdPort(), NULL, NULL,
		&httpd::req_handler,
		NULL, MHD_OPTION_END);

	if(d == nullptr)
	{
		printer::inst()->print_str("HTTP Daemon failed to start.");
		return false;
	}

	return true;
}

