FROM alpine:3.7
RUN apk add --no-cache curl
COPY src/download_data.sh /home/download_data.sh
CMD /bin/sh /home/download_data.sh
