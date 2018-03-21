#pragma once

#ifndef  CONF_NO_PROMETHEUS
#include <prometheus/registry.h>
#endif

class printer;
class jconf;
class executor;

namespace xmrstak
{

struct globalStates;
struct params;

struct environment
{
	static inline environment& inst(environment* init = nullptr)
	{
		static environment* env = nullptr;

		if(env == nullptr)
		{
			if(init == nullptr)
				env = new environment;
			else
				env = init;
		}

		return *env;
	}

	environment()
	{
	}

	printer* pPrinter = nullptr;
	globalStates* pglobalStates = nullptr;
	jconf* pJconfConfig = nullptr;
	executor* pExecutor = nullptr;
	params* pParams = nullptr;
#ifndef CONF_NO_PROMETHEUS
	std::shared_ptr<prometheus::Registry>* pRegistry = nullptr;
#endif
};

} // namepsace xmrstak
