FROM tensorflow/tensorflow:1.0.1-gpu

ENV PIP_INDEX_URL http://pypi.douban.com/simple/
ENV PIP_TRUSTED_HOST pypi.douban.com

COPY ./sources.list /etc/apt/

RUN apt-get update & apt-get -y install --no-install-recommends \
  git dnsutils \
  python-tk && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  pip install tensorlayer

COPY ./run_train.sh /
COPY ./run_tensorboard /

EXPOSE 2222

WORKDIR /

ENTRYPOINT ["/run_train.sh"]
