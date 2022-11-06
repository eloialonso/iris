#!/usr/bin/bash

fps=15
header=0
reconstruction=0
save_mode=0
mode="agent_in_env"

while [ "$1" != "" ]; do
    case $1 in
        -f | --fps )
            shift
            fps=$1
            ;;
        -h | --header )
            header=1
            ;;
        -r | --reconstruction )
            reconstruction=1
            ;;
        -s | --save-mode )
            save_mode=1
            ;;
        -a | --agent-world-model )
            mode="agent_in_world_model"
            ;;
        -e | --episode )
            mode="episode_replay"
            ;;
        -w | --world-model )
            mode="play_in_world_model"
            ;;
        * )
            echo Invalid usage : $1
            exit 1
    esac
    shift
done

python src/play.py hydra.run.dir=. hydra.output_subdir=null +mode="${mode}" +fps="${fps}" +header="${header}" +reconstruction="${reconstruction}" +save_mode="${save_mode}"
