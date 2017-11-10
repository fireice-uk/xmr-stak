#pragma once

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
#define XMR_STAK_VERSION "2.0.0-predev"
