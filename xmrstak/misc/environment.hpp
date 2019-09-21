#pragma once

#include <mutex>

class printer;
class jconf;
class executor;
struct randomX_global_ctx;

namespace xmrstak
{

struct globalStates;
struct params;
struct motd;

struct environment
{
	static inline environment& inst(environment* init = nullptr)
	{
		static environment* env = nullptr;

		if(env == nullptr)
		{
			if(init == nullptr)
			{
				env = new environment;
				env->init_singeltons();
			}
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
	motd* pMotd = nullptr;

	std::mutex update;

private:
	void init_singeltons();
};

} // namespace xmrstak
