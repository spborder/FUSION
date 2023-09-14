# Dockerfile for FUSION: Functional Unit State Identification and Navigation in WSI
FROM python:3.11

LABEL maintainer="Sam Border CMI Lab <samuel.border@medicine.ufl.edu"

RUN apt-get update && \
    apt-get install -y --no-install-recomends \
    git

ENV DSA_URL='http://ec2-3-230-122-132.compute-1.amazonaws.com:8080/api/v1/'
ENV DSA_USER='fusionguest'
ENV DSA_PWORD='Fus3yWasHere'

WORKDIR /
RUN git clone https://github.com/SarderLab/FUSION.git
RUN python3 -m pip install -r ./requirements.txt --no-cache-Dockerfile
RUN python3 -m pip freeze > pip_installed_packages.txt

EXPOSE 8000

ENTRYPOINT [ "python3" ]
CMD ["FUSION_Main.py"]