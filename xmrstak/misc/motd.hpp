#pragma once


#include "xmrstak/misc/environment.hpp"

#include <mutex>
#include <string>

namespace xmrstak
{

struct motd
{
	static inline motd& inst()
	{
		auto& env = environment::inst();
		if(env.pMotd == nullptr)
		{
			std::unique_lock<std::mutex> lck(env.update);
			if(env.pMotd == nullptr)
				env.pMotd = new motd;
		}
		return *env.pMotd;
	}

	std::string get_message()
	{
		std::unique_lock<std::mutex> lck(mtx);
		return message;
	}

	void set_message(const std::string msg)
	{
		std::unique_lock<std::mutex> lck(mtx);
		message = msg;
	}

  private:
	motd()
	{
	}

	std::mutex mtx;
	std::string message;
};

} // namespace xmrstak
