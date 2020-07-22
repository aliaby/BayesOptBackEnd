from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Union, Type, Dict, List, Tuple, Optional, Any, Callable

import gym

import torch as th
import torch.nn as nn


from stable_baselines3.common.torch_layers import (FlattenExtractor, BaseFeaturesExtractor, create_mlp,
                                                   NatureCNN, MlpExtractor)

class MLPAF(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable,
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 device: Union[th.device, str] = 'auto',
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 ortho_init: bool = True,
                 use_sde: bool = False,
                 log_std_init: float = 0.0,
                 full_std: bool = True,
                 sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False,
                 squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
    
        super(MLPAF, self).__init__(observation_space,
                                    action_space,
                                    lr_schedule,
                                    net_arch,
                                    device,
                                    activation_fn,
                                    ortho_init,
                                    use_sde,
                                    log_std_init,
                                    full_std,
                                    sde_net_arch,
                                    use_expln,
                                    squash_output,
                                    features_extractor_class,
                                    features_extractor_kwargs,
                                    normalize_images,
                                    optimizer_class,
                                    optimizer_kwargs)

    def get_value(self, obs: th.Tensor):
        actions, values, _ = self.forward(th.FloatTensor(obs).to(self.device))
        return actions, values
