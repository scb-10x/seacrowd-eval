FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install dumb-init
COPY requirements.txt /
RUN pip install -r /requirements.txt
ENV TRUST_REMOTE_CODE=true

COPY . .
ENTRYPOINT ["dumb-init", "--"]