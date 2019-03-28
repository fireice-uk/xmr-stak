#pragma once

#include <atomic>
#include <fstream>
#include <regex>
#include <streambuf>
#include <string>

#include "../version.hpp"

namespace xmrstak
{

struct configEditor
{
	std::string m_fileContent;

	configEditor()
	{
	}

	static bool file_exist(const std::string filename)
	{
		std::ifstream fstream(filename);
		return fstream.good();
	}

	void set(const std::string&& content)
	{
		m_fileContent = content;
	}

	bool load(const std::string filename)
	{
		std::ifstream fstream(filename);
		m_fileContent = std::string(
			(std::istreambuf_iterator<char>(fstream)),
			std::istreambuf_iterator<char>());
		return fstream.good();
	}

	void write(const std::string filename)
	{
		// endmarks: for filtering full lines inside the template string
		// Platform marks are done globally here
		// "---WINDOWS" endmark keeps lines when compiled for Windows
		// "---LINUX"   endmark keeps lines when compiled for Linux (and anything not-windows)
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__WINDOWS__)
		// windows:
		//   completely drop lines with endmark-linux
		replace(".*---LINUX\n", "");
		//   strip off windows endmarks, keep the lines
		replace("---WINDOWS\n", "\n");
#else
		// not-windows:
		//   completely drop lines with endmark-windows
		replace(".*---WINDOWS\n", "");
		//   strip off linux endmarks, keep the lines
		replace("---LINUX\n", "\n");
#endif
		replace("XMRSTAK_VERSION", get_version_str());
		std::ofstream out(filename);
		out << m_fileContent;
		out.close();
	}

	void replace(const std::string search, const std::string substring)
	{
		m_fileContent = std::regex_replace(m_fileContent, std::regex(search), substring);
	}
};

} // namespace xmrstak
