#pragma once

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
	
	void clean() 
        {
		if (pPrinter)
		{
			delete pPrinter;
			pPrinter = nullptr;	
		}
		if (pglobalStates)
		{
			delete pglobalStates;
			pglobalStates = nullptr;
		}
		if (pJconfConfig)
		{
			delete pJconfConfig;
			pJconfConfig = nullptr;
		}
		if (pExecutor)
		{
			delete pExecutor;
			pExecutor = nullptr;
		}
		if (pParams)
		{
			delete pParams;
			pParams = nullptr;
		}
        }

	printer* pPrinter = nullptr;
	globalStates* pglobalStates = nullptr;
	jconf* pJconfConfig = nullptr;
	executor* pExecutor = nullptr;
	params* pParams = nullptr;
};

} // namespace xmrstak
