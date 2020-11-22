FROM manifoldai/orbyter-ml-dev:3.0
ADD requirements.txt /build/requirements.txt
WORKDIR /build/
RUN pip install -r requirements.txt
#WORKDIR /volume/ka_azumio
WORKDIR /mnt/
#USER 1008:1008