#pragma once

class printer;
class jconf;
class executor;

namespace xmrstak
{

class globalStates;
class params;

struct environment
{

	static environment& inst()
	{
		static environment env;
		return env;
	}
	
	environment& operator=(const environment& env)
	{
		this->pPrinter = env.pPrinter;
		this->pglobalStates = env.pglobalStates;
		this->pJconfConfig = env.pJconfConfig;
		this->pExecutor = env.pExecutor;
		this->pParams = env.pParams;
		return *this;
	}


	environment() : pPrinter(nullptr), pglobalStates(nullptr)
	{
	}


	printer* pPrinter;
	globalStates* pglobalStates;
	jconf* pJconfConfig;
	executor* pExecutor;
	params* pParams;

};

} // namepsace xmrstak
