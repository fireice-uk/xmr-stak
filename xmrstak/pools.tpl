R"===(
/*
 * pool_address    - Pool address should be in the form "pool.supportxmr.com:3333". Only stratum pools are supported.
 * wallet_address  - Your wallet, or pool login.
 * rig_id          - Rig identifier for pool-side statistics (needs pool support).
 * pool_password   - Can be empty in most cases or "x".
 * use_nicehash    - Limit the nonce to 3 bytes as required by nicehash.
 * use_tls         - This option will make us connect using Transport Layer Security.
 * tls_fingerprint - Server's SHA256 fingerprint. If this string is non-empty then we will check the server's cert against it.
 * pool_weight     - Pool weight is a number telling the miner how important the pool is. Miner will mine mostly at the pool 
 *                   with the highest weight, unless the pool fails. Weight must be an integer larger than 0.
 *
 * We feature pools up to 1MH/s. For a more complete list see M5M400's pool list at www.moneropools.com
 */
 
"pool_list" :
[
POOLCONF],

/*
 * Currency to mine. Supported values:
 *
 *    aeon7 (use this for Aeon's new PoW)
 *    croat
 *    cryptonight (try this if your coin is not listed)
 *    cryptonight_lite
 *    edollar
 *    electroneum
 *    graft
 *    haven
 *    intense
 *    karbo
 *    monero7 (use this for Monero's new PoW)
 *    sumokoin
 *
 */

"currency" : "CURRENCY",

)==="
		
