FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m lmaack

RUN chown -R lmaack:lmaack /home/lmaack/

COPY --chown=lmaack . /home/lmaack/app/

USER lmaack

RUN cd /home/lmaack/app/ && pip3 install -r requirements.txt

WORKDIR /home/lmaack/app

EXPOSE 8080

ENTRYPOINT python3 api.py

