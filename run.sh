#!/bin/bash

cmd="source /mil_env/bin/activate && $@"
docker exec -it georg-mil bash -c "$cmd"