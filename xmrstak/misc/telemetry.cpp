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
  * Additional permission under GNU GPL version 3 section 7
  *
  * If you modify this Program, or any covered work, by linking or combining
  * it with OpenSSL (or a modified version of that library), containing parts
  * covered by the terms of OpenSSL License and SSLeay License, the licensors
  * of this Program grant you additional permission to convey the resulting work.
  *
  */

#include "telemetry.hpp"
#include "xmrstak/net/msgstruct.hpp"

#include <cmath>
#include <cstring>
#include <chrono>

namespace xmrstak
{

telemetry::telemetry(size_t iThd)
{
	ppHashCounts = new uint64_t*[iThd];
	ppTimestamps = new uint64_t*[iThd];
	iBucketTop = new uint32_t[iThd];

	for (size_t i = 0; i < iThd; i++)
	{
		ppHashCounts[i] = new uint64_t[iBucketSize];
		ppTimestamps[i] = new uint64_t[iBucketSize];
		iBucketTop[i] = 0;
		memset(ppHashCounts[i], 0, sizeof(uint64_t) * iBucketSize);
		memset(ppTimestamps[i], 0, sizeof(uint64_t) * iBucketSize);
	}
}

double telemetry::calc_telemetry_data(size_t iLastMilisec, size_t iThread)
{
	uint64_t iTimeNow = get_timestamp_ms();

	uint64_t iEarliestHashCnt = 0;
	uint64_t iEarliestStamp = 0;
	uint64_t iLastestStamp = 0;
	uint64_t iLastestHashCnt = 0;
	bool bHaveFullSet = false;

	//Start at 1, buckettop points to next empty
	for (size_t i = 1; i < iBucketSize; i++)
	{
		size_t idx = (iBucketTop[iThread] - i) & iBucketMask; //overflow expected here

		if (ppTimestamps[iThread][idx] == 0)
			break; //That means we don't have the data yet

		if (iLastestStamp == 0)
		{
			iLastestStamp = ppTimestamps[iThread][idx];
			iLastestHashCnt = ppHashCounts[iThread][idx];
		}

		if (iTimeNow - ppTimestamps[iThread][idx] > iLastMilisec)
		{
			bHaveFullSet = true;
			break; //We are out of the requested time period
		}

		iEarliestStamp = ppTimestamps[iThread][idx];
		iEarliestHashCnt = ppHashCounts[iThread][idx];
	}

	if (!bHaveFullSet || iEarliestStamp == 0 || iLastestStamp == 0)
		return nan("");

	//Don't think that can happen, but just in case
	if (iLastestStamp - iEarliestStamp == 0)
		return nan("");

	double fHashes, fTime;
	fHashes = iLastestHashCnt - iEarliestHashCnt;
	fTime = iLastestStamp - iEarliestStamp;
	fTime /= 1000.0;

	return fHashes / fTime;
}

void telemetry::push_perf_value(size_t iThd, uint64_t iHashCount, uint64_t iTimestamp)
{
	size_t iTop = iBucketTop[iThd];
	ppHashCounts[iThd][iTop] = iHashCount;
	ppTimestamps[iThd][iTop] = iTimestamp;

	iBucketTop[iThd] = (iTop + 1) & iBucketMask;
}

} // namepsace xmrstak
