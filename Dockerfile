# latch base image + dependencies for latch SDK --- removing these will break the workflow
from 812206152185.dkr.ecr.us-west-2.amazonaws.com/latch-base:ace9-main
run pip install latch
run mkdir /opt/latch

# install your requirements here
copy requirements.txt /root/requirements.txt
run pip install -r /root/requirements.txt

# copy all code from package (use .dockerignore to skip files)
copy . /root/

# latch internal tagging system + expected root directory --- changing these lines will break the workflow
arg tag
env FLYTE_INTERNAL_IMAGE $tag
workdir /root
