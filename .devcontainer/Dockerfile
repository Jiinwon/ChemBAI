# -------------------------------
# Dockerfile (Customized for Your Environment)
# * 사전 준비: 호스트에서 다음 명령어로 requirements.txt 파일을 생성하세요:
#     pip freeze > requirements.txt
#   생성된 requirements.txt를 프로젝트 루트에 커밋합니다.
# -------------------------------

# 1) Base image: Python 3.12.4과 Debian Bullseye 기반
FROM python:3.12.4-bullseye

# 2) Disable interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# 3) Install system dependencies
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      cmake \
      git \
      python3-rdkit \
      librdkit1 \
      rdkit-data && \
    rm -rf /var/lib/apt/lists/*

# 4) Map host user UID/GID to container user
ARG HOST_UID=1417       # 호스트에서 확인된 UID
ARG HOST_GID=100        # 호스트에서 확인된 GID
RUN groupadd --gid ${HOST_GID} hostgroup && \
    useradd --uid ${HOST_UID} --gid hostgroup --shell /bin/bash --create-home hostuser

# 5) Switch to non-root user
USER hostuser
WORKDIR /home/hostuser/workspace

# 6) Install Python dependencies
COPY --chown=hostuser:hostgroup requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# 7) Copy project source into workspace
COPY --chown=hostuser:hostgroup . .

# 8) GPU 지원: 실행 시 --gpus all 플래그 사용
# docker run --gpus all -v $PWD:/home/hostuser/workspace your-image-name

# 9) Default entrypoint: Bash shell
CMD ["bash"]