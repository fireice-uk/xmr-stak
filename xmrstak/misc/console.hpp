#pragma once

#include "xmrstak/misc/environment.hpp"

#include <mutex>
#include <vector>

enum out_colours : uint8_t
{
	K_RED = 7,
	K_GREEN = 6,
	K_BLUE = 5,
	K_YELLOW = 4,
	K_CYAN = 3,
	K_MAGENTA = 2,
	K_WHITE = 1,
	K_NONE = 0
};

// Warning - on Linux get_key will detect control keys, but not on Windows.
// We will only use it for alphanum keys anyway.
int get_key();

void set_colour(out_colours cl);
void reset_colour();

// on MSVC sizeof(long int) = 4, gcc sizeof(long int) = 8, this is the workaround
// now we can use %llu on both compilers
inline long long unsigned int int_port(size_t i)
{
	return i;
}

enum verbosity : size_t
{
	L0 = 0,
	L1 = 1,
	L2 = 2,
	L3 = 3,
	L4 = 4,
	LDEBUG = 10,
	LINF = 100
};

struct colored_cstr
{
	colored_cstr(const char* cstr) : c_str(cstr){}
	colored_cstr(const char* cstr, const out_colours & colour) : c_str(cstr), m_colour(colour){}

	const char* c_str;
	out_colours m_colour = out_colours::K_NONE;
};

class printer
{
  public:
	static inline printer* inst()
	{
		auto& env = xmrstak::environment::inst();
		if(env.pPrinter == nullptr)
		{
			std::unique_lock<std::mutex> lck(env.update);
			if(env.pPrinter == nullptr)
				env.pPrinter = new printer;
		}
		return env.pPrinter;
	};

	inline void set_verbose_level(size_t level) { verbose_level = (verbosity)level; }
	void print_msg(verbosity verbose, const char* fmt, ...);
	void print_str(const char* str);
	bool open_logfile(const char* file);

	void print_str(std::vector<colored_cstr> vcs);
	void print_str(out_colours, const char* cstr);
	void print_coloured_str(char * cstr, const size_t length);
	bool open_logfile(char* file);

  private:
	printer();

	std::mutex print_mutex;
	verbosity verbose_level;
	FILE* logfile;
};

void win_exit(int code = 1);
