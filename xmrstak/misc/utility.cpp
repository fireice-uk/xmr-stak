#include <string>
#include <algorithm>


namespace xmrstak
{
	bool strcmp_i(const std::string& str1, const std::string& str2)
	{
		if(str1.size() != str2.size())
			return false;
		else
		return (str1.empty() | str2.empty()) ?
				false :
				std::equal(str1.begin(), str1.end(),str2.begin(),
					[](char c1, char c2)
					{
						return ::tolower(c1) == ::tolower(c2);
					}
				);
	}
} // namespace xmrstak
