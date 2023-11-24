FROM ubuntu:latest

WORKDIR /STOCK_PRICE_4_FUN

COPY . /STOCK_PRICE_4_FUN

RUN make install

CMD make bot