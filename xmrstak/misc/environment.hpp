#pragma once

class printer;
class jconf;
class executor;
struct randomX_global_ctx;

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
	randomX_global_ctx* pGlobalCtx = nullptr;
};

} // namespace xmrstak
