FROM alpine:latest

RUN apk --no-cache upgrade \
        && apk --no-cache add \
                cmake \
                automake \
                autoconf \
                openssl-dev \
                git \
                build-base \
                libmicrohttpd-dev \
        && git clone https://github.com/fireice-uk/xmr-stak.git \
        && cd xmr-stak \
        && cmake . \
                -DCUDA_ENABLE=OFF \
                -DOpenCL_ENABLE=OFF \
                -DHWLOC_ENABLE=OFF \
        && make \
        && cd - \
        && mv /xmr-stak/bin/* /usr/local/bin/ \
        && rm -rf /xmr-stak \
        && apk del \
                automake \
                autoconf \
                build-base \
                git

VOLUME /mnt
WORKDIR /mnt

COPY ["run.sh", "setup.sh", "./"]
CMD ["./run.sh"]
