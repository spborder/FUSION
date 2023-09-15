# Dockerfile for FUSION: Functional Unit State Identification and Navigation in WSI
#docker build 
FROM python:3.11

LABEL maintainer="Sam Border CMI Lab <samuel.border@medicine.ufl.edu"

RUN apt-get update && \
    apt-get install -y git openssh-client && \
    apt-get clean

ENV DSA_URL='http://ec2-3-230-122-132.compute-1.amazonaws.com:8080/api/v1/'
ENV DSA_USER='fusionguest'
ENV DSA_PWORD='Fus3yWasHere'

RUN mkdir -p /root/.ssh && \
    echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config

WORKDIR /

# Copy the SSH private key into the image
COPY id_rsa /root/.ssh/id_rsa

# Set the permissions and clone the repo using the SSH agent
RUN chmod 600 /root/.ssh/id_rsa && \
    eval $(ssh-agent -s) && \
    ssh-add /root/.ssh/id_rsa && \
    git clone git@github.com:spborder/FUSION.git

RUN echo "Listing contents:" && ls -al /

WORKDIR /FUSION/
RUN python3 -m pip install -r ./requirements.txt --no-cache-dir
RUN python3 -m pip freeze > pip_installed_packages.txt

EXPOSE 8000

ENTRYPOINT [ "python3" ]
CMD ["FUSION_Main.py"]