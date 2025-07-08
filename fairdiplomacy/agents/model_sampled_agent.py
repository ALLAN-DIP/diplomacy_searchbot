# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from conf import agents_cfgs
from fairdiplomacy.agents.base_agent import BaseAgent

from fairdiplomacy.agents.model_wrapper import ModelWrapper
from fairdiplomacy.agents.plausible_order_sampling import PlausibleOrderSampler, renormalize_policy
from fairdiplomacy.models.consts import POWERS
from fairdiplomacy.utils.thread_pool_encoding import FeatureEncoder


_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelSampledAgent(BaseAgent):
    def __init__(self, cfg: agents_cfgs.ModelSampledAgent, device=_DEFAULT_DEVICE):
        self.model = ModelWrapper(
            cfg.model_path, device=device, half_precision=cfg.half_precision
        )
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.device = device
        self.input_encoder = FeatureEncoder()

        self.order_sampler = PlausibleOrderSampler(
            cfg.plausible_orders_cfg, model=self.model
        )

    def get_orders(self, game, power, **kwargs):
        if len(game.get_orderable_locations().get(power, [])) == 0:
            return []
        return self.get_orders_many_powers(game, [power], **kwargs)[power]

    def get_orders_many_powers(self, game, powers, *, temperature=None, top_p=None):
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        encode_fn = (
            self.input_encoder.encode_inputs_all_powers
            if self.model.is_all_powers()
            else self.input_encoder.encode_inputs
        )
        inputs = encode_fn([game])

        inputs = inputs.to(self.device)
        actions, _, _ = self.model.do_model_request(inputs, temperature=temperature, top_p=top_p)
        actions = actions[0]  # batch dim
        return {p: a for p, a in zip(POWERS, actions) if p in powers}
    
    def get_plausible_orders_policy(self, game):
        # Determine the set of plausible actions to consider for each power
        policy = self.order_sampler.sample_orders(game)

        return policy