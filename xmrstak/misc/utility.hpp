#pragma once

#include <string>

namespace xmrstak
{
	/** case insensitive string compare
	 *
	 * @return true if both strings are equal, else false
	 */
	bool strcmp_i(const std::string& str1, const std::string& str2);

	//! prevent compiler warning for unused variables
	template<typename T>
	void ignore_unused(const T&)
	{}

} // namepsace xmrstak
