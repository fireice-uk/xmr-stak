#include "socket.h"
#include "jpsock.h"
#include "jconf.h"

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/opensslconf.h>

#ifndef OPENSSL_THREADS
#error OpenSSL was compiled without thread support
#endif

plain_socket::plain_socket(jpsock* err_callback) : pCallback(err_callback)
{
	hSocket = INVALID_SOCKET;
	pSockAddr = nullptr;
}

bool plain_socket::set_hostname(const char* sAddr)
{
	char sAddrMb[256];
	char *sTmp, *sPort;

	size_t ln = strlen(sAddr);
	if (ln >= sizeof(sAddrMb))
		return pCallback->set_socket_error("CONNECT error: Pool address overflow.");

	memcpy(sAddrMb, sAddr, ln);
	sAddrMb[ln] = '\0';

	if ((sTmp = strstr(sAddrMb, "//")) != nullptr)
		memmove(sAddrMb, sTmp, strlen(sTmp) + 1);

	if ((sPort = strchr(sAddrMb, ':')) == nullptr)
		return pCallback->set_socket_error("CONNECT error: Pool port number not specified, please use format <hostname>:<port>.");

	sPort[0] = '\0';
	sPort++;

	addrinfo hints = { 0 };
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	pAddrRoot = nullptr;
	int err;
	if ((err = getaddrinfo(sAddrMb, sPort, &hints, &pAddrRoot)) != 0)
		return pCallback->set_socket_error_strerr("CONNECT error: GetAddrInfo: ", err);

	addrinfo *ptr = pAddrRoot;
	addrinfo *ipv4 = nullptr, *ipv6 = nullptr;

	while (ptr != nullptr)
	{
		if (ptr->ai_family == AF_INET)
			ipv4 = ptr;
		if (ptr->ai_family == AF_INET6)
			ipv6 = ptr;
		ptr = ptr->ai_next;
	}

	if (ipv4 == nullptr && ipv6 == nullptr)
	{
		freeaddrinfo(pAddrRoot);
		pAddrRoot = nullptr;
		return pCallback->set_socket_error("CONNECT error: I found some DNS records but no IPv4 or IPv6 addresses.");
	}
	else if (ipv4 != nullptr && ipv6 == nullptr)
		pSockAddr = ipv4;
	else if (ipv4 == nullptr && ipv6 != nullptr)
		pSockAddr = ipv6;
	else if (ipv4 != nullptr && ipv6 != nullptr)
	{
		if(jconf::inst()->PreferIpv4())
			pSockAddr = ipv4;
		else
			pSockAddr = ipv6;
	}

	hSocket = socket(pSockAddr->ai_family, pSockAddr->ai_socktype, pSockAddr->ai_protocol);

	if (hSocket == INVALID_SOCKET)
	{
		freeaddrinfo(pAddrRoot);
		pAddrRoot = nullptr;
		return pCallback->set_socket_error_strerr("CONNECT error: Socket creation failed ");
	}

	return true;
}

bool plain_socket::connect()
{
	int ret = ::connect(hSocket, pSockAddr->ai_addr, (int)pSockAddr->ai_addrlen);

	freeaddrinfo(pAddrRoot);
	pAddrRoot = nullptr;

	if (ret != 0)
		return pCallback->set_socket_error_strerr("CONNECT error: ");
	else
		return true;
}

int plain_socket::recv(char* buf, unsigned int len)
{
	int ret = ::recv(hSocket, buf, len, 0);

	if(ret == 0)
		pCallback->set_socket_error("RECEIVE error: socket closed");
	if(ret == SOCKET_ERROR || ret < 0)
		pCallback->set_socket_error_strerr("RECEIVE error: ");

	return ret;
}

bool plain_socket::send(const char* buf)
{
	int pos = 0, slen = strlen(buf);
	while (pos != slen)
	{
		int ret = ::send(hSocket, buf + pos, slen - pos, 0);
		if (ret == SOCKET_ERROR)
		{
			pCallback->set_socket_error_strerr("SEND error: ");
			return false;
		}
		else
			pos += ret;
	}

	return true;
}

void plain_socket::close(bool free)
{
	if(hSocket != INVALID_SOCKET)
	{
		sock_close(hSocket);
		hSocket = INVALID_SOCKET;
	}
}

tls_socket::tls_socket(jpsock* err_callback) : pCallback(err_callback)
{
}

void tls_socket::print_error()
{
	BIO* err_bio = BIO_new(BIO_s_mem());
	ERR_print_errors(err_bio);

	char *buf = nullptr;
	size_t len = BIO_get_mem_data(err_bio, &buf);

	pCallback->set_socket_error(buf, len);

	BIO_free(err_bio);
}

void tls_socket::init_ctx()
{
	const SSL_METHOD* method = SSLv23_method();

	if(method == nullptr)
		return;

	ctx = SSL_CTX_new(method);
	if(ctx == nullptr)
		return;

	SSL_CTX_set_options(ctx, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_COMPRESSION);
}

bool tls_socket::set_hostname(const char* sAddr)
{
	if(ctx == nullptr)
	{
		init_ctx();
		if(ctx == nullptr)
		{
			print_error();
			return false;
		}
	}

	if((bio = BIO_new_ssl_connect(ctx)) == nullptr)
	{
		print_error();
		return false;
	}

	if(BIO_set_conn_hostname(bio, sAddr) != 1)
	{
		print_error();
		return false;
	}

	BIO_get_ssl(bio, &ssl);
	if(ssl == nullptr)
	{
		print_error();
		return false;
	}

	/*if(SSL_set_cipher_list(ssl, "HIGH:!aNULL:!kRSA:!PSK:!SRP:!MD5:!RC4") != 1)
	{
		print_error();
		return false;
	}*/
	return true;
}

bool tls_socket::connect()
{
	if(BIO_do_connect(bio) != 1)
	{
		print_error();
		return false;
	}

	if(BIO_do_handshake(bio) != 1)
	{
		print_error();
		return false;
	}

	/* Step 1: verify a server certificate was presented during the negotiation */
	X509* cert = SSL_get_peer_certificate(ssl);
	if(cert == nullptr)
	{
		print_error();
		return false;
	}

	const EVP_MD* digest;
	unsigned char md[EVP_MAX_MD_SIZE];
	unsigned int dlen;

	digest = EVP_get_digestbyname("sha256");
	if(digest == nullptr)
	{
		print_error();
		false;
	}

	if(X509_digest(cert, digest, md, &dlen) != 1)
	{
		print_error();
		false;
	}

	for(size_t i=0; i < dlen; i++)
		printf("%.2X:", md[i]);
	printf("\n");

	X509_free(cert);
	return true;
}

int tls_socket::recv(char* buf, unsigned int len)
{
	int ret = BIO_read(bio, buf, len);

	if(ret == 0)
		pCallback->set_socket_error("RECEIVE error: socket closed");
	if(ret < 0)
		print_error();

	return ret;
}

bool tls_socket::send(const char* buf)
{
	return BIO_puts(bio, buf) > 0;
}

void tls_socket::close(bool free)
{
	if(bio == nullptr || ssl == nullptr)
		return;

	if(!free)
	{
		sock_close(BIO_get_fd(bio, nullptr));
	}
	else
	{
		BIO_free_all(bio);
		ssl = nullptr;
		bio = nullptr;
	}
}

