# Dockerfile for FUSION: Functional Unit State Identification and Navigation in WSI
#docker build 
FROM python:3.11

LABEL maintainer="Sam Border CMI Lab <samuel.border@medicine.ufl.edu"

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

WORKDIR /
COPY / /
RUN python3 -m pip install -r requirements.txt --no-cache-dir
RUN python3 -m pip freeze > pip_installed_packages.txt

EXPOSE 8000

ENTRYPOINT [ "python3" ]
CMD ["FUSION.py"]