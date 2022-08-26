from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from agent import Agent
from envs import SingleProcessEnv, WorldModelEnv
from game import AgentEnv, EpisodeReplayEnv, Game
from models.actor_critic import ActorCritic
from models.world_model import WorldModel


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    device = torch.device(cfg.common.device)
    assert cfg.mode in ('world_model', 'episode_replay', 'agent')

    if cfg.mode in ['world_model', 'agent']:
        env_fn = lambda: instantiate(cfg.env.test)
        test_env = SingleProcessEnv(env_fn)
        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions, config=instantiate(cfg.world_model))
        if cfg.mode == 'world_model':
            env = WorldModelEnv(tokenizer=tokenizer, world_model=world_model, pretrained_agent_path=Path('checkpoints/last.pt'), device=device, env=env_fn())
            keymap = cfg.env.keymap
        else:
            actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=test_env.num_actions)
            agent = Agent(tokenizer, world_model, actor_critic).to(device)
            agent.load(Path('checkpoints/last.pt'), device)
            env = AgentEnv(agent, test_env, cfg.env.keymap)
            keymap = 'empty'

    else:
        env = EpisodeReplayEnv(replay_keymap_name=cfg.env.keymap, episode_dir=Path('media/episodes'))
        keymap = 'episode_replay'

    game = Game(env, keymap_name=keymap, size=(600, 1200 if cfg.mode == 'agent' else 600), fps=cfg.fps, verbose=bool(cfg.header))
    game.run()


if __name__ == "__main__":
    main()
