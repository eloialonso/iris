from dataclasses import dataclass
from typing import Any, Optional, Union
import sys

from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import Batch
from src.envs.world_model_env import WorldModelEnv
from src.models.tokenizer import Tokenizer
from src.models.world_model import WorldModel
from src.utils import compute_lambda_returns, LossWithIntermediateLosses


@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    mean_continuous: torch.FloatTensor
    std_continuous: torch.FloatTensor
    means_values: torch.FloatTensor


@dataclass
class ImagineOutput:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    actions_continuous: torch.LongTensor
    logits_actions: torch.FloatTensor
    continuous_means: torch.FloatTensor
    continuous_stds: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor


class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size, act_continuous_size, use_original_obs: bool = False) -> None:
        # TODO add act continuous size #1done
        super().__init__()
        self.use_original_obs = use_original_obs
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        self.hx, self.cx = None, None

        self.act_vocab_size = act_vocab_size
        self.act_continuous_size = act_continuous_size
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512,
                                      self.act_vocab_size + 2 * self.act_continuous_size)  # TODO add more entries for continuous (2*x) for mean and std #1done

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None,
              mask_padding: Optional[torch.Tensor] = None) -> None:
        device = self.conv1.weight.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)
        if burnin_observations is not None:
            assert burnin_observations.ndim == 5 and burnin_observations.size(
                0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None) -> ActorCriticOutput:
        assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
        assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
        assert mask_padding is None or (
                    mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
        x = inputs[mask_padding] if mask_padding is not None else inputs

        x = x.mul(2).sub(1)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)

        if mask_padding is None:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))

        full_logits = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
        logits_actions = full_logits[..., :self.act_vocab_size]  # TODO split logits to create another OutputClass #1done
        mean_continuous = full_logits[..., self.act_vocab_size:self.act_vocab_size + self.act_continuous_size]
        std_continuous = full_logits[..., self.act_vocab_size + self.act_continuous_size:]
        means_values = rearrange(self.critic_linear(self.hx), 'b 1 -> b 1 1')

        return ActorCriticOutput(logits_actions, mean_continuous, std_continuous, means_values)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, imagine_horizon: int,
                     gamma: float, lambda_: float, entropy_weight: float, **kwargs: Any) -> LossWithIntermediateLosses:
        assert not self.use_original_obs
        outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]

        d = Categorical(logits=outputs.logits_actions[:, :-1])  # TODO accept new OutputClass #1done
        log_probs = d.log_prob(outputs.actions[:, :-1])  # TODO #1done
        loss_actions = -1 * (
                    log_probs * (lambda_returns - values.detach())).mean()  # TODO define loss for continuous actions #1done

        cont = Normal(outputs.continuous_means[:,:-1], outputs.continuous_stds[:,:-1])
        log_probs_continuous = cont.log_prob(outputs.actions_continuous[:,:-1])
        loss_continuous_actions = -1 * (
                    log_probs_continuous * (lambda_returns - values.detach())).mean()

        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_entropy_continuous = - entropy_weight * cont.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)

        return LossWithIntermediateLosses(loss_actions=loss_actions,
                                          loss_continuous_actions=loss_continuous_actions,
                                          loss_values=loss_values,
                                          loss_entropy=loss_entropy,
                                          loss_entropy_continuous=loss_entropy_continuous)

    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, horizon: int,
                show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        initial_observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = initial_observations.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_continuous = []
        all_logits_actions = []
        all_continuous_means = []
        all_continuous_stds = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        burnin_observations = torch.clamp(
            tokenizer.encode_decode(initial_observations[:, :-1], should_preprocess=True, should_postprocess=True), 0,
            1) if initial_observations.size(1) > 1 else None
        self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations,
                   mask_padding=mask_padding[:, :-1])

        obs = wm_env.reset_from_initial_observations(initial_observations[:, -1])
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):
            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = Categorical(
                logits=outputs_ac.logits_actions).sample()  # TODO add continuous (from new OutputClass) #1done
            action_continuous = Normal(outputs_ac.mean_continuous, outputs_ac.std_continuous).rsample()
            obs, reward, done, _ = wm_env.step(action_token,
                                               continuous=action_continuous,
                                               should_predict_next_obs=(k < horizon - 1))  # TODO add continuous #1done

            all_actions.append(action_token)  # TODO concat #1done
            all_continuous.append(action_continuous)
            all_logits_actions.append(outputs_ac.logits_actions)  # TODO concat #1done
            all_continuous_means.append(outputs_ac.mean_continuous)
            all_continuous_stds.append(outputs_ac.std_continuous)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),  # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),  # (B, T)
            actions_continuous=torch.stack(all_continuous, dim=1),  # (B, T, #actions)
            logits_actions=torch.cat(all_logits_actions, dim=1),  # (B, T, #actions)
            continuous_means=torch.stack(all_continuous_means, dim=1),  # (B, T, #actions)
            continuous_stds=torch.stack(all_continuous_stds, dim=1),  # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),  # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),  # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),  # (B, T)
        )
