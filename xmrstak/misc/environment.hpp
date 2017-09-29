#pragma once

class printer;
class jconf;
class executor;

namespace xmrstak
{

class GlobalStates;
class Params;

struct Environment
{

	static Environment& inst()
	{
		static Environment env;
		return env;
	}
	
	Environment& operator=(const Environment& env)
	{
		this->pPrinter = env.pPrinter;
		this->pGlobalStates = env.pGlobalStates;
		this->pJconfConfig = env.pJconfConfig;
		this->pExecutor = env.pExecutor;
		this->pParams = env.pParams;
		return *this;
	}


	Environment() : pPrinter(nullptr), pGlobalStates(nullptr)
	{
	}


	printer* pPrinter;
	GlobalStates* pGlobalStates;
	jconf* pJconfConfig;
	executor* pExecutor;
	Params* pParams;

};

} // namepsace xmrstak
