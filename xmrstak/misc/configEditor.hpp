#pragma once

#include <atomic>
#include <string>
#include <fstream>
#include <streambuf>
#include <regex>


namespace xmrstak
{

struct configEditor
{
	std::string m_fileContent;

	configEditor() 
	{

	}

	static bool file_exist( const std::string filename)
	{
		std::ifstream fstream(filename);
		return fstream.good();
	}

	void set( const std::string && content)
	{
		m_fileContent = content;
	}

	bool load(const std::string filename)
	{
		std::ifstream fstream(filename);
		m_fileContent = std::string(
			(std::istreambuf_iterator<char>(fstream)),
			std::istreambuf_iterator<char>()
		);
		return fstream.good();
	}

	void write(const std::string filename)
	{
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
