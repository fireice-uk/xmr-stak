#pragma once

class printer;
class jconf;
class executor;

namespace xmrstak
{

class GlobalStates;

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
		return *this;
	}


	Environment() : pPrinter(nullptr), pGlobalStates(nullptr)
	{
	}


	printer* pPrinter;

	GlobalStates* pGlobalStates;

	jconf* pJconfConfig;

	executor* pExecutor;

};

} // namepsace xmrstak
