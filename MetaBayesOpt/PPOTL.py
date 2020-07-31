
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.buffers import RolloutBuffer

from typing import Type, Union, Callable, Optional, Dict, Any

import torch as th

class PPOTL(PPO):
    def __init__(self, policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Callable] = 3e-4,
                 n_steps: int = 64,
                 batch_size: Optional[int] = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 clip_range_vf: Optional[float] = None,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True):

        super(PPOTL, self).__init__(policy=policy, env=env, learning_rate=learning_rate,
                                    n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                                    gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range,
                                    clip_range_vf=clip_range_vf, ent_coef=ent_coef, vf_coef=vf_coef,
                                    max_grad_norm=max_grad_norm, use_sde=use_sde, sde_sample_freq=sde_sample_freq,
                                    target_kl=target_kl, tensorboard_log=tensorboard_log, create_eval_env=create_eval_env,
                                    policy_kwargs=policy_kwargs, verbose=verbose, seed=seed, device=device,
                                    _init_setup_model=_init_setup_model)


    def set_policy(self, policy):
        self.policy = policy
        self.setup_PPO_model()

    def setup_PPO_model(self) -> None:

        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = RolloutBuffer(self.n_steps, self.observation_space,
                                            self.action_space, self.device,
                                            gamma=self.gamma, gae_lambda=self.gae_lambda,
                                            n_envs=self.n_envs)
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, ("`clip_range_vf` must be positive, "
                                                "pass `None` to deactivate vf clipping")

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)


