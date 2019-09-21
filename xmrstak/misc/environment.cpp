#include "environment.hpp"

#include "xmrstak/misc/console.hpp"
#include "xmrstak/backend/cpu/crypto/cryptonight.h"
#include "xmrstak/params.hpp"
#include "xmrstak/misc/executor.hpp"
#include "xmrstak/jconf.hpp"

namespace xmrstak
{
void environment::init_singeltons()
{
	printer::inst();
	globalStates::inst();
	jconf::inst();
	executor::inst();
	params::inst();
	// do not init randomx here else you can get rejected results
	//randomX_global_ctx::inst();
}
}
