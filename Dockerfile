FROM python:3.7.9-slim-stretch
ENV TINI_VERSION v0.19.0
WORKDIR /work/

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    ca-certificates \
    g++ \
    git \
    curl \
    sudo \
    make \
    xz-utils \
    patch \
    file \
    perl \
    cmake \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*  

RUN mkdir /work/tmp && \
    wget https://github.com/linux-test-project/lcov/releases/download/v1.12/lcov-1.12.tar.gz -O /work/tmp/lcov-1.12.tar.gz && \
    tar xfz /work/tmp/lcov-1.12.tar.gz -C /work/tmp/ && \
    cd /work/tmp/lcov-1.12 && \
    make install
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
RUN pip install numpy>=1.11 \
    tqdm>=4.0 \
    optuna>=1.0.0 \
    pandas>=1.0.0 \
    scikit-learn>=0.21.0 \
    scipy>=1.0 \
    lightfm>=1.15 \
    jax \
    jaxlib \
    dm-haiku \
    optax \
    pytest \
    pytest-cov
RUN wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -O $HOME/.ml-100k.zip
COPY irspack /work/irspack
COPY cpp_source /work/cpp_source
COPY setup.py /work/setup.py

RUN IRSPACK_TESTING="true" python setup.py develop
COPY tests /work/tests
