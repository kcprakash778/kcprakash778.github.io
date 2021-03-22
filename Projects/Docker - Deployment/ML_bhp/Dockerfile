FROM ubuntu:18.04


ENV PATH = "/root/miniconda3/bin/:${PATH}"
# ARG PATH = "/root/miniconda3/bin/:${PATH}"


RUN apt-get update -y \ 
    && apt-get install -y python3.8 \
    && apt install python3-pip -y
    # wget

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh

# RUN conda create -y -n ml python=3.8


COPY . src/

RUN /bin/bash -c "cd src \
    && pip3 install -r requirements.txt"

EXPOSE 5000