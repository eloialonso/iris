#!/usr/bin/bash

fps=15
header=1
mode="episode_replay"

while [ "$1" != "" ]; do
    case $1 in
        -f | --fps )
            shift
            fps=$1
            ;;
        -h | --no-header )
            header=0
            ;;
        -w | --world-model )
            mode="world_model"
            ;;
        -a | --agent )
            mode="agent"
            ;;
        * )
            echo Invalid usage : $1
            exit 1
    esac
    shift
done

python src/play.py hydra.run.dir=. hydra.output_subdir=null +mode="${mode}" +fps="${fps}" +header="${header}"
