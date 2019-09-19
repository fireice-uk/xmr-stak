#pragma once

#include <mutex>

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

	std::mutex update;

private:
	void init_singeltons();
};

} // namespace xmrstak
