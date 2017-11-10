#include "version.hpp"

//! git will put "#define GIT_ARCHIVE 1" on the next line inside archives. $Format:%n#define GIT_ARCHIVE 1$
#if defined(GIT_ARCHIVE) && !defined(GIT_COMMIT_HASH)
#define GIT_COMMIT_HASH "$Format:%h$"
#endif

#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "0000000"
#endif

#ifndef GIT_BRANCH
#define GIT_BRANCH "unknown"
#endif

#define XMR_STAK_NAME "xmr-stak"
#define XMR_STAK_VERSION "2.0.0"

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

#if !defined(CONF_NO_CUDA) && !defined(CONF_NO_OPENCL)
#define BACKEND_TYPE "cpu-nvidia-amd"
#elif !defined(CONF_NO_OPENCL)
#define BACKEND_TYPE "cpu-amd"
#elif !defined(CONF_NO_CUDA)
#define BACKEND_TYPE "cpu-nvidia"
#else
#define BACKEND_TYPE "cpu"
#endif

#if defined(CONF_NO_AEON)
#define COIN_TYPE "monero"
#elif defined(CONF_NO_MONERO)
#define COIN_TYPE "aeon"
#else
#define COIN_TYPE "aeon-monero"
#endif

#define VERSION_LONG  XMR_STAK_NAME "/" XMR_STAK_VERSION "/" GIT_COMMIT_HASH "/" GIT_BRANCH "/" OS_TYPE "/" BACKEND_TYPE "/" COIN_TYPE "/"
#define VERSION_SHORT XMR_STAK_NAME " " XMR_STAK_VERSION " " GIT_COMMIT_HASH

const char ver_long[]  = VERSION_LONG;
const char ver_short[] = VERSION_SHORT;
