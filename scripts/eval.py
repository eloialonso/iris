#! python

import argparse
from datetime import datetime
from pathlib import Path
import subprocess

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-episodes', type=int, default=100, help='Number of evaluation episodes to collect.')
    parser.add_argument('-p', '--num-envs', type=int, default=25, help='Number of environments used to collect the evaluation episodes.')
    args = parser.parse_args()

    path_to_config = Path('config') / 'trainer.yaml'
    cfg = OmegaConf.load(path_to_config)

    path_to_checkpoint = Path('checkpoints') / 'last.pt'
    assert path_to_checkpoint.is_file()

    cmd = f'python src/main.py hydra.run.dir=eval_outputs/{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")} '

    cmd += 'wandb.mode=online '
    cmd += f'wandb.name=eval-{cfg.wandb.name} '
    cmd += f'wandb.group=eval-{cfg.wandb.group} '

    cmd += f'initialization.path_to_checkpoint={str(path_to_checkpoint.absolute())} '
    cmd += 'initialization.load_tokenizer=True '
    cmd += 'initialization.load_world_model=False '
    cmd += 'initialization.load_actor_critic=True '

    cmd += 'common.epochs=1 '
    cmd += 'common.device=cuda:0 '
    cmd += 'common.do_checkpoint=False '
    cmd += 'common.seed=0 '

    cmd += 'collection.test.num_episodes_to_save=0 '
    cmd += f'collection.test.num_envs={args.num_envs} '
    cmd += f'collection.test.config.num_episodes={args.num_episodes} '

    cmd += 'training.should=False '

    # Turn on data collection only
    cmd += 'evaluation.should=True '
    cmd += 'evaluation.every=1 '
    cmd += 'evaluation.tokenizer.start_after_epochs=1 '
    cmd += 'evaluation.tokenizer.save_reconstructions=False '
    cmd += 'evaluation.world_model.start_after_epochs=1 '
    cmd += 'evaluation.actor_critic.start_after_epochs=1 '

    subprocess.run(cmd, shell=True, check=True)


if __name__ == '__main__':
    main()
