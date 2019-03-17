FROM ubuntu:16.04

RUN apt-get update

RUN apt-get install -y sudo
RUN apt-get install -y firefox
RUN apt-get install -y python3
RUN apt-get install -y python-imaging-tk
RUN apt-get install -y python3-pil.imagetk
RUN apt-get install -y python3-pip
RUN apt-get install -y git

# Replace 1000 with your user / group id
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \    
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer

USER developer
ENV HOME /home/developer

RUN mkdir /home/developer/work/
RUN cd /home/developer/work/
RUN pip3 install git+https://github.com/prabindh/euclid.git#egg=euclid