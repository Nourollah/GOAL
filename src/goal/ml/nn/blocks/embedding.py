"""Atomic number embedding — map integer Z to learnable feature vectors."""

from __future__ import annotations

import torch
import torch.nn as nn


class AtomicNumberEmbedding(nn.Module):
    """Learnable embedding for atomic numbers.

    Maps integer atomic numbers to dense feature vectors that initialise
    the node features at the start of the model.

    Parameters
    ----------
    num_elements : int
        Maximum atomic number supported (default 120 covers all elements).
    embedding_dim : int
        Dimension of the embedding vectors.
    """

    def __init__(
        self,
        num_elements: int = 120,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_elements, embedding_dim)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for the given atomic numbers.

        Parameters
        ----------
        atomic_numbers : Tensor
            Integer atomic numbers, shape ``(N,)``.

        Returns
        -------
        Tensor
            Embedding vectors, shape ``(N, embedding_dim)``.
        """
        return self.embedding(atomic_numbers)                                # (N, embedding_dim)
