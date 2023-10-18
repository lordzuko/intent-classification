FROM ubuntu:18.04

# ENVIRONMENTS
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# INSTALL MISSING PACKAGES AND CUDA
RUN apt-get update --fix-missing && \
    apt-get -y upgrade && \
    apt-get -y install --no-install-recommends apt-utils wget ca-certificates && \
    apt-get clean && apt-get autoremove

# # INSTALL MINICONDA
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda clean --all && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#     echo "conda activate base" >> ~/.bashrc && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /

# INSTALL APPLICATION ENVIRONMENT
COPY requirements.txt requirements.txt
pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python3","server.py"]
