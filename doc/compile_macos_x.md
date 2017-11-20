cd <your xmr-stak folder>
brew install openssl libmicrohttpd
mkdir build && cd build
cmake -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DOPENSSL_INCLUDE_DIR=/usr/local/opt/openssl/include/ ..
make
