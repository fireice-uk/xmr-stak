FROM bwstitt/ubuntu:16.04

RUN docker-install \
    build-essential \
    cmake \
    libhwloc-dev \
    libmicrohttpd-dev \
    libssl-dev

# todo: use make install properly
COPY . /usr/local/src/xmr-stak/
RUN set -eux; \
    \
    cd /usr/local/src/xmr-stak/; \
    cmake .; \
    make install; \
    mv docker-entrypoint.sh

USER abc
ENV HOME /home/abc

WORKDIR $HOME

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["xmr-stak"]
