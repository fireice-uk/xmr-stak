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

