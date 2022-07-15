FROM nvidia/cuda:11.7.0-runtime-ubuntu20.04 as base
RUN apt-get update && apt-get install -y curl && apt -y upgrade

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    unzip \
    git \
    wget && rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

# install rclone
# RUN curl https://rclone.org/install.sh | bash -
# COPY config/rclone.conf /root/.config/rclone/

# install jupyter lab with extensions and export port
RUN pip3 install jupyter-http-over-ws==0.0.8 jupyterlab==3.4.2
RUN jupyter serverextension enable --py jupyter_http_over_ws
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter lab build
EXPOSE 8888
# default shell for jupyter lab
ENV SHELL=/bin/bash

# install pip requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# setup /labs folder for experimental work
RUN mkdir -p /labs && chmod -R a+rwx /labs/
WORKDIR /labs
RUN mkdir /.local && chmod a+rwx /.local


RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=/labs --ip 0.0.0.0 --no-browser --allow-root"]
