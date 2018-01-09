R"===(
/*
 * pool_address    - Pool address should be in the form "pool.supportxmr.com:3333". Only stratum pools are supported.
 * wallet_address  - Your wallet, or pool login.
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
 * currency to mine
 * allowed values: 'monero' or 'aeon'
 */
"currency" : "CURRENCY",

/*
 * Network timeouts.
 * Because of the way this client is written it doesn't need to constantly talk (keep-alive) to the server to make 
 * sure it is there. We detect a buggy / overloaded server by the call timeout. The default values will be ok for 
 * nearly all cases. If they aren't the pool has most likely overload issues. Low call timeout values are preferable -
 * long timeouts mean that we waste hashes on potentially stale jobs. Connection report will tell you how long the
 * server usually takes to process our calls.
 *
 * call_timeout - How long should we wait for a response from the server before we assume it is dead and drop the connection.
 * retry_time	- How long should we wait before another connection attempt.
 *                Both values are in seconds.
 * giveup_limit - Limit how many times we try to reconnect to the pool. Zero means no limit. Note that stak miners
 *                don't mine while the connection is lost, so your computer's power usage goes down to idle.
 */
"call_timeout" : 10,
"retry_time" : 30,
"giveup_limit" : 0,

/*
 * Output control.
 * Since most people are used to miners printing all the time, that's what we do by default too. This is suboptimal
 * really, since you cannot see errors under pages and pages of text and performance stats. Given that we have internal
 * performance monitors, there is very little reason to spew out pages of text instead of concise reports.
 * Press 'h' (hashrate), 'r' (results) or 'c' (connection) to print reports.
 *
 * verbose_level - 0 - Don't print anything.
 *                 1 - Print intro, connection event, disconnect event
 *                 2 - All of level 1, and new job (block) event if the difficulty is different from the last job
 *                 3 - All of level 1, and new job (block) event in all cases, result submission event.
 *                 4 - All of level 3, and automatic hashrate report printing
 *
 * print_motd    - Display messages from your pool operator in the hashrate result.
 */
"verbose_level" : 3,
"print_motd" : true,

/*
 * Automatic hashrate report
 *
 * h_print_time - How often, in seconds, should we print a hashrate report if verbose_level is set to 4.
 *                This option has no effect if verbose_level is not 4.
 */
"h_print_time" : 60,

/*
 * Manual hardware AES override
 *
 * Some VMs don't report AES capability correctly. You can set this value to true to enforce hardware AES or
 * to false to force disable AES or null to let the miner decide if AES is used.
 *
 * WARNING: setting this to true on a CPU that doesn't support hardware AES will crash the miner.
 */
"aes_override" : null,

/*
 * LARGE PAGE SUPPORT
 * Large pages need a properly set up OS. It can be difficult if you are not used to systems administration,
 * but the performance results are worth the trouble - you will get around 20% boost. Slow memory mode is
 * meant as a backup, you won't get stellar results there. If you are running into trouble, especially
 * on Windows, please read the common issues in the README.
 *
 * By default we will try to allocate large pages. This means you need to "Run As Administrator" on Windows.
 * You need to edit your system's group policies to enable locking large pages. Here are the steps from MSDN
 *
 * 1. On the Start menu, click Run. In the Open box, type gpedit.msc.
 * 2. On the Local Group Policy Editor console, expand Computer Configuration, and then expand Windows Settings.
 * 3. Expand Security Settings, and then expand Local Policies.
 * 4. Select the User Rights Assignment folder.
 * 5. The policies will be displayed in the details pane.
 * 6. In the pane, double-click Lock pages in memory.
 * 7. In the Local Security Setting – Lock pages in memory dialog box, click Add User or Group.
 * 8. In the Select Users, Service Accounts, or Groups dialog box, add an account that you will run the miner on
 * 9. Reboot for change to take effect.
 *
 * Windows also tends to fragment memory a lot. If you are running on a system with 4-8GB of RAM you might need
 * to switch off all the auto-start applications and reboot to have a large enough chunk of contiguous memory.
 *
 * On Linux you will need to configure large page support "sudo sysctl -w vm.nr_hugepages=128" and increase your
 * ulimit -l. To do do this you need to add following lines to /etc/security/limits.conf - "* soft memlock 262144"
 * and "* hard memlock 262144". You can also do it Windows-style and simply run-as-root, but this is NOT
 * recommended for security reasons.
 *
 * Memory locking means that the kernel can't swap out the page to disk - something that is unlikely to happen on a
 * command line system that isn't starved of memory. I haven't observed any difference on a CLI Linux system between
 * locked and unlocked memory. If that is your setup see option "no_mlck".
 */

/*
 * use_slow_memory defines our behaviour with regards to large pages. There are three possible options here:
 * always  - Don't even try to use large pages. Always use slow memory.
 * warn    - We will try to use large pages, but fall back to slow memory if that fails.
 * no_mlck - This option is only relevant on Linux, where we can use large pages without locking memory.
 *           It will never use slow memory, but it won't attempt to mlock
 * never   - If we fail to allocate large pages we will print an error and exit.
 */
"use_slow_memory" : "warn",

/*
 * TLS Settings
 * If you need real security, make sure tls_secure_algo is enabled (otherwise MITM attack can downgrade encryption
 * to trivially breakable stuff like DES and MD5), and verify the server's fingerprint through a trusted channel.
 *
 * tls_secure_algo - Use only secure algorithms. This will make us quit with an error if we can't negotiate a secure algo.
 */
"tls_secure_algo" : true,

/*
 * Daemon mode
 *
 * If you are running the process in the background and you don't need the keyboard reports, set this to true.
 * This should solve the hashrate problems on some emulated terminals.
 */
"daemon_mode" : false,

/*
 * Buffered output control.
 * When running the miner through a pipe, standard output is buffered. This means that the pipe won't read
 * each output line immediately. This can cause delays when running in background.
 * Set this option to true to flush stdout after each line, so it can be read immediately.
 */
"flush_stdout" : false,

/*
 * Output file
 *
 * output_file  - This option will log all output to a file.
 *
 */
"output_file" : "",

/*
 * Built-in web server
 * I like checking my hashrate on my phone. Don't you?
 * Keep in mind that you will need to set up port forwarding on your router if you want to access it from
 * outside of your home network. Ports lower than 1024 on Linux systems will require root.
 *
 * httpd_port - Port we should listen on. Default, 0, will switch off the server.
 */
"httpd_port" : HTTP_PORT,

/*
 * HTTP Authentication
 *
 * This allows you to set a password to keep people on the Internet from snooping on your hashrate.
 * Keep in mind that this is based on HTTP Digest, which is based on MD5. To a determined attacker
 * who is able to read your traffic it is as easy to break a bog door latch.
 *
 * http_login - Login. Empty login disables authentication.
 * http_pass  - Password.
 */ 
"http_login" : "",
"http_pass" : "",
 
/*
 * prefer_ipv4 - IPv6 preference. If the host is available on both IPv4 and IPv6 net, which one should be choose?
 *               This setting will only be needed in 2020's. No need to worry about it now.
 */
"prefer_ipv4" : true,

)==="
		
