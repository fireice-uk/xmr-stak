#include "version.hpp"

//! git will put "#define GIT_ARCHIVE 1" on the next line inside archives. $Format:%n#define GIT_ARCHIVE 1$
#if defined(GIT_ARCHIVE) && !defined(GIT_COMMIT_HASH)
#define GIT_COMMIT_HASH \
	$Format:            \
	% h$
#endif

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH 0000000
#endif

#ifndef GIT_BRANCH
#define GIT_BRANCH unknown
#endif

#ifndef BACKEND_TYPE
#define BACKEND_TYPE unknown
#endif

#define XMR_STAK_NAME "xmr-stak"
#define XMR_STAK_VERSION "2.10.8"

#if defined(_WIN32)
#define OS_TYPE "win"
#elif defined(__APPLE__)
#define OS_TYPE "mac"
#elif defined(__FreeBSD__)
#define OS_TYPE "bsd"
#elif defined(__linux__)
#define OS_TYPE "lin"
#else
#define OS_TYPE "unk"
#endif

#define XMRSTAK_PP_TOSTRING1(str) #str
#define XMRSTAK_PP_TOSTRING(str) XMRSTAK_PP_TOSTRING1(str)

#define VERSION_LONG XMR_STAK_NAME "/" XMR_STAK_VERSION "/" XMRSTAK_PP_TOSTRING(GIT_COMMIT_HASH) "/" XMRSTAK_PP_TOSTRING(GIT_BRANCH) "/" OS_TYPE "/" XMRSTAK_PP_TOSTRING(BACKEND_TYPE) "/"
#define VERSION_SHORT XMR_STAK_NAME " " XMR_STAK_VERSION " " XMRSTAK_PP_TOSTRING(GIT_COMMIT_HASH)
#define VERSION_HTML "v" XMR_STAK_VERSION "-" XMRSTAK_PP_TOSTRING(GIT_COMMIT_HASH)

const char ver_long[] = VERSION_LONG;
const char ver_short[] = VERSION_SHORT;
const char ver_html[] = VERSION_HTML;
