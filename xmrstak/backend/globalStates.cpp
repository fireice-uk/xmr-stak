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
	jobLock.ReadLock();

	threadWork = oGlobalWork;
	currentJobId = iGlobalJobNo.load(std::memory_order_relaxed);
	
	jobLock.UnLock();
}

void globalStates::switch_work(miner_work& pWork, pool_data& dat)
{
	jobLock.WriteLock();
	
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
	
	jobLock.UnLock();
}

} // namespace xmrstak
