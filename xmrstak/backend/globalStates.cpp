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

#include "miner_work.hpp"
#include "globalStates.hpp"

#include <assert.h>
#include <cmath>
#include <chrono>
#include <cstring>


namespace xmrstak
{

void globalStates::consume_work( miner_work& threadWork, uint64_t& currentJobId)
{
	/* Only the executer thread which updates the job is ever setting iConsumeCnt
	 * to 1000. In this case each consumer must wait until the job is fully updated.
	 */
	uint64_t numConsumer = 0;

	/* Take care that we not consume a job if the job is updated.
	 * If we leave the loop we have increased iConsumeCnt so that
	 * the job will not be updated until we leave the method.
	 */
	do{
		numConsumer = iConsumeCnt.load(std::memory_order_relaxed);
		if(numConsumer < 1000)
		{
			// register that thread try consume job data
			numConsumer = ++iConsumeCnt;
			if(numConsumer >= 1000)
			{
				iConsumeCnt--;
				// 11 is a arbitrary chosen prime number
				std::this_thread::sleep_for(std::chrono::milliseconds(11));
			}
		}
		else
		{
			// an other thread is preparing a new job, 11 is a arbitrary chosen prime number
			std::this_thread::sleep_for(std::chrono::milliseconds(11));
		}
	}
	while(numConsumer >= 1000);

	threadWork = oGlobalWork;
	currentJobId = iGlobalJobNo.load(std::memory_order_relaxed);
	
	// signal that thread consumed work
	iConsumeCnt--;
}

void globalStates::switch_work(miner_work& pWork, pool_data& dat)
{
	/* 1000 is used to notify that the the job will be updated as soon
	 * as all consumer (which currently coping oGlobalWork has copied
	 * all data)
	 */
	iConsumeCnt += 1000;
	// wait until all threads which entered consume_work are finished
	while (iConsumeCnt.load(std::memory_order_relaxed) > 1000)
	{
		// 7 is a arbitrary chosen prime number which is smaller than the consumer waiting time
		std::this_thread::sleep_for(std::chrono::milliseconds(7));
	}
	// BEGIN CRITICAL SECTION
	// this notifies all threads that the job has changed
	iGlobalJobNo++; 

	size_t xid = dat.pool_id;
	dat.pool_id = pool_id;
	pool_id = xid;

	/* Maybe a worker thread is updating the nonce while we read it.
	 * In that case GPUs check the job ID after a nonce update and in the
	 * case that it is a CPU thread we have a small chance (max 6 nonces per CPU thread)
	 * that we recalculate a nonce after we reconnect to the current pool
	 */
	dat.iSavedNonce = iGlobalNonce.exchange(dat.iSavedNonce, std::memory_order_relaxed);
	oGlobalWork = pWork;
	// END CRITICAL SECTION: allow job consume
	iConsumeCnt -= 1000;
}

} // namespace xmrstak
