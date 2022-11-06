from src.models.world_model import WorldModel
from src.models.tokenizer.tokenizer import Tokenizer
from src.envs.world_model_env import WorldModelEnv
import hydra
from hydra.utils import instantiate
import torch
import numpy as np


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg):
    world_model = WorldModel(obs_vocab_size=512, act_vocab_size=7,
                                     #act_continuous_size=env.num_continuous,
                                     config=instantiate(cfg.world_model))  # TODO add continuous
    tokenizer = instantiate(cfg.tokenizer)
    wm_env = WorldModelEnv(tokenizer, world_model, 'cpu')
    wm_env.reset_from_initial_observations(torch.distributions.uniform.Uniform(0,1).sample([1,3,64,64]).float())
    wm_env.step(np.array([[1]]))


if __name__ == "__main__":
    main()