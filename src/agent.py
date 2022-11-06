from pathlib import Path
from typing import Tuple

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from src.models.actor_critic import ActorCritic
from src.utils import extract_state_dict


class Agent(nn.Module):
    actor_critic: ActorCritic

    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True,
             load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(
            self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        out = self.actor_critic(input_ac)
        logits_actions = out.logits_actions[:, -1] / temperature  # TODO split logits (from updated OutputClass) #1done
        act_token = Categorical(logits=logits_actions).sample(
            sample_shape=(1,)) if should_sample else logits_actions.argmax(dim=-1)
        mean_continuous, std_continuous = out.mean_continuous, out.std_continuous
        act_continuous = torch.sigmoid(Normal(mean_continuous, std_continuous).rsample())
        return act_token, act_continuous  # TODO add sigmoid on the logits split #1done
