all:
	- mkdir build
	cd build && cmake -DCUDA_ENABLE=OFF ..
	cd build && make install
