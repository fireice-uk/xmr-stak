#pragma once

#include <string>

namespace xmrstak
{
	/** case insensitive string compare
	 *
	 * @return true if both strings are equal, else false
	 */
	bool strcmp_i(const std::string& str1, const std::string& str2);
} // namespace xmrstak
