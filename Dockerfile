FROM ubuntu:18.04

WORKDIR /code

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Copy the files to docker image
COPY requirements.txt requirements.txt
COPY .ipynb_checkpoints ./
COPY Report_SER.ipynb ./
COPY app.py ./
COPY start.sh ./

# Install the python requirements
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

RUN  apt-get -y update \
  && apt-get install -y wget \
  && apt-get install -y zip unzip \
  && apt-get install -y libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

# Download data
RUN wget http://emodb.bilderbar.info/download/download.zip


# Unzip and remove the downloaded that
RUN unzip download.zip -d ./data

CMD bash start.sh