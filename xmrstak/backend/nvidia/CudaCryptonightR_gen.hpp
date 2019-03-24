/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include "xmrstak/backend/cryptonight.hpp"

#include <stdint.h>
#include <string>
#include <vector>

namespace xmrstak
{
namespace nvidia
{

void CryptonightR_get_program(std::vector<char>& ptx, std::string& lowered_name,
	const xmrstak_algo algo, uint64_t height, uint32_t precompile_count, int arch_major, int arch_minor, bool background = false);

} // namespace nvidia
} // namespace xmrstak
