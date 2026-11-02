# Prologue
# DO NOT CHANGE
from 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:cb01-main

workdir /tmp/docker-build/work/

shell [ \
    "/usr/bin/env", "bash", \
    "-o", "errexit", \
    "-o", "pipefail", \
    "-o", "nounset", \
    "-o", "verbose", \
    "-o", "errtrace", \
    "-O", "inherit_errexit", \
    "-O", "shift_verbose", \
    "-c" \
]
env TZ='Etc/UTC'
env LANG='en_US.UTF-8'

arg DEBIAN_FRONTEND=noninteractive

# Latch SDK
# DO NOT REMOVE
run pip install latch==2.78.0
run mkdir /opt/latch

# Install only what wf/ imports, at the same pins as requirements.txt.
# The base image runs Python 3.11; scipy==1.18.1 and anndata==0.13.4 need >=3.12.
run pip install numpy==2.4.4 pandas==3.0.6 torch==2.14.1

# Copy workflow data (use .dockerignore to skip files)
copy . /root/

# Epilogue

# Latch workflow registration metadata
# DO NOT CHANGE
arg tag
# DO NOT CHANGE
env FLYTE_INTERNAL_IMAGE $tag

workdir /root
