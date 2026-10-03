"""AMP model whose per-component critic head can be a HL-Gauss classifier.

Subclasses AMPModel so that `learning/{ppo_model,amp_model}.py` stay
byte-identical: the PPO ladder that the whole cross-family comparison is
paired against must keep running the exact code that produced it.

With `w5_norm.method != "hlgauss"` this is AMPModel with no behavioural
change at all; the reward-normalization and return-scaling arms need no model
change, only agent-side wiring.
"""

import torch

import learning.amp_model as amp_model
import learning.w5_norm as w5_norm
import util.torch_util as torch_util

class W5AMPModel(amp_model.AMPModel):
    def __init__(self, config, env):
        w5_config = dict(config.get("w5_norm", {}))
        self._w5_method = str(w5_config.get("method", "none"))
        self._w5_config = w5_config
        super().__init__(config, env)
        return

    def uses_hlgauss(self):
        return self._w5_method == "hlgauss"

    def get_hlgauss(self):
        return self._hlgauss

    def _build_critic(self, config, env):
        super()._build_critic(config, env)
        if (not self.uses_hlgauss()):
            self._hlgauss = None
            return

        assert(self._toppo_enabled), "HL-Gauss requires the component critic."
        assert(not self._toppo_use_popart), \
            "HL-Gauss REPLACES PopArt; enable exactly one."

        self._hlgauss = w5_norm.HLGaussHead.from_config(self._w5_config)
        # one set of logits per component
        layers_out_size = torch_util.calc_layers_out_size(self._critic_layers)
        self._toppo_critic_out = torch.nn.Linear(
            layers_out_size, self._toppo_num_tasks * self._hlgauss.num_bins)
        with torch.no_grad():
            self._toppo_critic_out.bias.copy_(
                self._hlgauss.logits_for_value(0.0).repeat(self._toppo_num_tasks))
        return

    def eval_toppo_critic_logits(self, obs):
        """[batch, K, num_bins] -- HL-Gauss only."""
        assert(self.uses_hlgauss())
        h = self._critic_layers(obs)
        logits = self._toppo_critic_out(h)
        # obs arrives flat during the update and as [T, num_envs, ...] while the
        # train data is built, so keep every leading dim
        return logits.view(*logits.shape[:-1], self._toppo_num_tasks,
                           self._hlgauss.num_bins)

    def eval_toppo_critic_components(self, obs, normalized=False):
        if (not self.uses_hlgauss()):
            return super().eval_toppo_critic_components(obs, normalized=normalized)
        # `normalized` is meaningless without PopArt; the categorical readout is
        # always in raw value units.
        return self._hlgauss.from_logits(self.eval_toppo_critic_logits(obs))
