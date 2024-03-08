#!/bin/bash

cmd="source /env/bin/activate && $@"
docker exec -it georg-mil bash -c "$cmd"