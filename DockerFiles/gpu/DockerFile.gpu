FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

LABEL maintainer="DLindenbaum <https://github.com/dlindenbaum>"

## Install basic tools
RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y --no-install-recommends \
    bc \
    ca-certificates \
    curl \
    git \
    jq \
    nfs-common \
    parallel \
    python-pip \
    python-wheel \
    python-setuptools \
    unzip \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# install conda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc #&& \
    #echo "conda activate" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

SHELL ["/bin/bash", "-c"]


# Create cw-nets environment
#RUN export PATH=/opt/conda/bin/:$PATH
#RUN /opt/conda/bin/conda activate
RUN /opt/conda/bin/conda update conda
RUN /opt/conda/bin/conda create -n cw-nets python=3.6 pip cython
RUN /opt/conda/bin/conda install --name cw-nets \
                                rtree \
                                gdal

# Install Cosmiq Tools
RUN source /opt/conda/bin/activate /opt/conda/envs/cw-nets && \
        conda update pip && \
        conda install pytorch torchvision cuda91 -c pytorch && \
        conda install opencv scikit-image && \
        pip install git+https://github.com/CosmiQ/cw-tiler.git@dataset_creation && \
        pip install git+https://github.com/CosmiQ/cw-nets.git@pytorch_generator
        

# Install Perl Script for Google Drive Download
RUN mkdir -p /opt/code/gdown.pl
RUN git clone https://github.com/circulosmeos/gdown.pl.git /opt/code/gdown.pl


# Install Teranaus

RUN mkdir -p /opt/code/TernausNetV2
RUN mkdir -p /opt/code/tq
RUN git clone -b bigTiff --single-branch https://github.com/dlindenbaum/TernausNetV2.git /opt/code/TernausNetV2/
RUN cd /opt/code/TernausNetV2/TernausNetV2/modules \
        && chmod +x build.sh \
        && ./build.sh \
        && source /opt/conda/bin/activate /opt/conda/envs/cw-nets \
        && python build.py

RUN source /opt/conda/bin/activate /opt/conda/envs/cw-nets \
        && cd /opt/code/TernausNetV2/ \
        && pip install --upgrade setuptools \
        && pip install -e . 

RUN mkdir -p /opt/code/TernausNetV2/weights/ \
     && /opt/code/gdown.pl/gdown.pl https://drive.google.com/uc?id=1k95VGNZG74Vvu-X-MSpbaHjMDvNEepIi /opt/code/TernausNetV2/weights/deepglobe_buildings.pt
     
RUN source /opt/conda/bin/activate /opt/conda/envs/cw-nets \
        && pip install rio-cogeo







## Install 

#ENTRYPOINT [ "/usr/bin/tini", "--" ]
#CMD [ "/bin/bash" ]

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

#WORKDIR "/notebooks"

#CMD ["/run_jupyter.sh", "--allow-root"]