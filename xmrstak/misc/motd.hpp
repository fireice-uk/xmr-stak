#pragma once


#include "xmrstak/misc/environment.hpp"

#include <mutex>
#include <string>
#include <regex>

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

	std::string get_url()
	{
		std::unique_lock<std::mutex> lck(mtx);
		return url;
	}

	void set_message(const std::string msg)
	{
		std::unique_lock<std::mutex> lck(mtx);
		message = msg;
		update_url();
	}

  private:

	inline bool is_url(std::string url)
	{
		std::string pattern = "https?:\\/\\/(www\\.)?([-a-zA-Z0-9@:%_\\+.~#?&//=]*)";
		std::regex url_regex(pattern);

		return std::regex_match(url, url_regex);
	}

	inline void update_url()
	{

		std::string url_string;
		size_t last_start = 0;
		for(int i = 0; i < message.size(); i++)
		{
			if(message[i] == '\n')
			{
				std::string line(&message[last_start], i - last_start);
				if(is_url(line))
				{
					url_string = line;
				}
				last_start = i + 2;
			}
		}
		if(!url_string.empty())
		{
			message.append("\n");
			message.insert(message.size(), 1, K_WHITE);
			message.append("press key 'v' to open the url\n");
			message.insert(message.size(), 1, K_YELLOW);
			message.append("\n");
			message.insert(message.size(), 1, K_WHITE);
		}
		url = url_string;
	}

	motd()
	{
	}

	std::mutex mtx;
	std::string message;
	std::string url;
};

} // namespace xmrstak
