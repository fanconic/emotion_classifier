#!/bin/bash
#https://docs.docker.com/config/containers/multi-service_container/

# runs 2 commands simultaneously:

jupyter notebook --port=9000 --no-browser --ip=0.0.0.0 --allow-root & # your first application
P1=$!
python3 app.py & # your second application
P2=$!
wait $P1 $P2