"""Exponential Moving Average (EMA) of model parameters.

Maintains a shadow copy of the model weights that is updated after each
training step. The EMA weights typically give better validation and test
performance than the raw trained weights.
"""

from __future__ import annotations

import typing
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.nn as nn


class EMAWrapper:
    """Exponential moving average of model parameters.

    Parameters
    ----------
    parameters : iterable of Tensor
        The model parameters to track.
    decay : float
        EMA decay factor. Typical values: 0.999 or 0.9999.
        Higher values give smoother averages.
    """

    def __init__(
        self,
        parameters: typing.Iterator[nn.Parameter],
        decay: float = 0.999,
    ) -> None:
        self.decay: float = decay
        self._shadow: list[torch.Tensor] = [p.clone().detach() for p in parameters]
        self._parameters: list[nn.Parameter] = list(parameters)

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        for shadow, param in zip(self._shadow, self._parameters):
            shadow.lerp_(param.data, 1.0 - self.decay)

    @contextmanager
    def average_parameters(self) -> typing.Iterator[None]:
        """Context manager that swaps in EMA parameters for evaluation.

        The original parameters are restored when the context exits,
        even if an exception occurs.

        Usage::

            with ema.average_parameters():
                predictions = model(batch)
        """
        originals: list[torch.Tensor] = [p.data.clone() for p in self._parameters]
        try:
            for param, shadow in zip(self._parameters, self._shadow):
                param.data.copy_(shadow)
            yield
        finally:
            for param, original in zip(self._parameters, originals):
                param.data.copy_(original)

    def state_dict(self) -> dict[str, list[torch.Tensor]]:
        """Serialise EMA state for checkpointing."""
        return {"shadow": [s.clone() for s in self._shadow]}

    def load_state_dict(self, state_dict: dict[str, list[torch.Tensor]]) -> None:
        """Restore EMA state from a checkpoint."""
        self._shadow = [s.clone() for s in state_dict["shadow"]]
