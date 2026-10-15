"""ALRA: Action-Latent Residual Adaptation."""

import logging

import torch
import torch.nn as nn

from .adajepa import AdaJEPATrainer

log = logging.getLogger(__name__)


class Residual(nn.Module):
    def __init__(self, action_dim: int, proprio_dim: int):
        super().__init__()
        self.head = nn.Linear(action_dim + proprio_dim, action_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, act_emb, proprio_emb):
        return self.head(torch.cat([act_emb, proprio_emb], dim=-1))


class _ActionSlot:
    def __init__(self, wm):
        if wm.concat_dim != 1:
            raise ValueError(f"residual adaptation requires concat_dim=1, got {wm.concat_dim}")
        self.A, self.P = wm.action_dim, wm.proprio_dim
        self.a_rep = wm.num_action_repeat
        self.action_dim = self.A // self.a_rep
        self.proprio_dim = self.P // wm.num_proprio_repeat

    def read(self, z):
        return (
            z[:, :, 0, -self.A :][..., : self.action_dim],
            z[:, :, 0, -(self.P + self.A) : -self.A][..., : self.proprio_dim],
        )

    def write(self, z, act):
        z = z.clone()
        z[..., -self.A :] = act.repeat(1, 1, self.a_rep).unsqueeze(2).expand(-1, -1, z.shape[2], -1)
        return z


def attach_residual(wm):
    slot = _ActionSlot(wm)
    # keep the global RNG unchanged by the Linear init
    rng_state = torch.get_rng_state()
    residual = Residual(slot.action_dim, slot.proprio_dim)
    torch.set_rng_state(rng_state)
    residual.to(next(wm.parameters()).device)
    original = wm.predict

    def predict(z):
        act, proprio = slot.read(z)
        return original(slot.write(z, act + residual(act, proprio)))

    # not registered as a submodule, so freezing wm.parameters() leaves it trainable
    object.__setattr__(wm, "_action_residual", residual)
    wm.predict = predict
    return residual


class ResidualTrainer(AdaJEPATrainer):
    def __init__(self, wm, lr: float, steps: int = 1, optimizer_name: str = "adam"):
        # before super().__init__(), which calls _select_predictor_params()
        self.residual = attach_residual(wm)
        super().__init__(
            wm=wm, lr=lr, steps=steps, optimizer_name=optimizer_name, finetune_encoder=False
        )
        log.info(
            "Action residual: linear %d -> %d | %d params, lr=%.2e, steps=%d",
            self.residual.head.in_features,
            self.residual.head.out_features,
            sum(p.numel() for p in self._ada_predictor_params),
            self.lr,
            self.steps,
        )

    def _select_predictor_params(self):
        return list(self.residual.parameters())

    def _set_train_mode(self, mode: bool):
        pass  # predictor stays in eval mode
