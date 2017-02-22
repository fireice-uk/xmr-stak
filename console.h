#pragma once
#include <mutex>

enum out_colours { K_RED, K_GREEN, K_BLUE, K_YELLOW, K_CYAN, K_MAGENTA, K_WHITE, K_NONE };

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

enum verbosity : size_t { L0 = 0, L1 = 1, L2 = 2, L3 = 3, L4 = 4, LINF = 100};

class printer
{
public:
	static inline printer* inst()
	{
		if (oInst == nullptr) oInst = new printer;
		return oInst;
	};

	inline void set_verbose_level(size_t level) { verbose_level = (verbosity)level; }
	void print_msg(verbosity verbose, const char* fmt, ...);
	void print_str(const char* str);
	bool open_logfile(const char* file);

private:
	printer();
	static printer* oInst;

	std::mutex print_mutex;
	verbosity verbose_level;
	FILE* logfile;
};
