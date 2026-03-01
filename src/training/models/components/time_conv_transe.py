"""Time-ConvTransE decoder for TiRGN.

Extends the standard ConvTransE scoring function with two additional temporal
channels: a learned non-periodic (linear) time embedding and a learned periodic
(sinusoidal) time embedding. The 4-channel input [subject, relation,
time_nonperiodic, time_periodic] is processed through 1D convolution, batch
normalization, ReLU, dropout, and a final FC projection. Scores are produced
via dot product with all entity embeddings.

Reference:
    Li et al. (2022). TiRGN: Time-Guided Recurrent Graph Network with
    Local-Global Historical Patterns for Temporal Knowledge Graph Reasoning.
    IJCAI 2022.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


class TimeConvTransEDecoder(nnx.Module):
    """ConvTransE decoder with learned periodic and non-periodic time channels.

    Scores (subject, relation, time) queries against all entities by:
    1. Gathering subject, relation embeddings and computing time embeddings
    2. Stacking into a 4-channel tensor: (batch, 4, embedding_dim)
    3. Applying 1D convolution across the channel-stacked embedding dimension
    4. BatchNorm + ReLU + Dropout + FC projection
    5. Dot product with all entity embeddings -> (batch, num_entities)

    The decoder does NOT own entity embeddings. It receives them from the
    parent TiRGN module (shared between raw and history decoders).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        num_filters: int = 32,
        kernel_size: int = 3,
        dropout_rate: float = 0.2,
        max_time_steps: int = 366,
        *,
        rngs: nnx.Rngs,
    ):
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Relation embeddings owned by this decoder (independent per decoder
        # instance -- the raw and history decoders have separate rel embeddings)
        self.rel_emb = nnx.Param(
            jax.random.normal(rngs.params(), (num_relations, embedding_dim)) * 0.01
        )

        # Learned time embeddings: non-periodic (linear) and periodic (sin)
        # weight and bias are (embedding_dim,) -- broadcast across batch via
        # elementwise ops with time index scalar expanded to (batch, 1)
        self.weight_t_nonperiodic = nnx.Param(
            jax.random.normal(rngs.params(), (embedding_dim,)) * 0.01
        )
        self.bias_t_nonperiodic = nnx.Param(jnp.zeros((embedding_dim,)))
        self.weight_t_periodic = nnx.Param(
            jax.random.normal(rngs.params(), (embedding_dim,)) * 0.01
        )
        self.bias_t_periodic = nnx.Param(jnp.zeros((embedding_dim,)))

        # 1D convolution: input shape (batch, 4, embedding_dim)
        # Flax nnx.Conv expects (batch, spatial..., features). We treat
        # embedding_dim as the spatial axis and 4 channels as features.
        # So input is (batch, embedding_dim, 4) after transpose.
        self.conv = nnx.Conv(
            in_features=4,
            out_features=num_filters,
            kernel_size=(kernel_size,),
            padding="VALID",
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

        # BatchNorm after conv: feature axis is the last dim (num_filters)
        # Input to BN: (batch, out_len, num_filters)
        self.bn = nnx.BatchNorm(
            num_features=num_filters,
            axis=-1,
            rngs=rngs,
        )

        # FC projection: flatten conv output -> embedding_dim
        conv_out_len = embedding_dim - kernel_size + 1
        fc_in_dim = num_filters * conv_out_len
        self.fc = nnx.Linear(
            in_features=fc_in_dim,
            out_features=embedding_dim,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

        # Dropout
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def get_time_encoding(
        self, time_indices: Array
    ) -> tuple[Array, Array]:
        """Compute periodic and non-periodic time embeddings.

        Args:
            time_indices: Integer time step indices (batch,).

        Returns:
            Tuple of (nonperiodic, periodic), each (batch, embedding_dim).
        """
        # (batch,) -> (batch, 1) for broadcasting with (embedding_dim,)
        t = time_indices[:, None].astype(jnp.float32)

        # Non-periodic: linear function of time
        t_nonperiodic = (
            self.weight_t_nonperiodic.value * t + self.bias_t_nonperiodic.value
        )

        # Periodic: sinusoidal function of time
        t_periodic = jnp.sin(
            self.weight_t_periodic.value * t + self.bias_t_periodic.value
        )

        return t_nonperiodic, t_periodic

    def __call__(
        self,
        entity_emb: Array,
        triples: Array,
        time_indices: Array,
        training: bool = True,
    ) -> Array:
        """Score all entities for given (subject, relation, time) queries.

        Args:
            entity_emb: Entity embeddings (num_entities, embedding_dim).
            triples: (batch, 3) [subject, relation, object] indices.
            time_indices: (batch,) integer day index for time encoding.
            training: Whether to use training-mode dropout and batch norm.

        Returns:
            Raw logits of shape (batch, num_entities). Softmax is NOT applied
            here -- the caller is responsible for applying softmax or
            log-softmax as needed.
        """
        subjects = triples[:, 0]
        relations = triples[:, 1]

        # Gather subject and relation embeddings
        subj_emb = entity_emb[subjects]  # (batch, dim)
        rel_emb = self.rel_emb.value[relations]  # (batch, dim)

        # Compute time embeddings
        t_nonperiodic, t_periodic = self.get_time_encoding(time_indices)

        # Cast to bfloat16 for compute efficiency
        subj_emb = subj_emb.astype(jnp.bfloat16)
        rel_emb = rel_emb.astype(jnp.bfloat16)
        t_nonperiodic = t_nonperiodic.astype(jnp.bfloat16)
        t_periodic = t_periodic.astype(jnp.bfloat16)

        # Stack 4 channels: (batch, 4, embedding_dim)
        x = jnp.stack([subj_emb, rel_emb, t_nonperiodic, t_periodic], axis=1)

        # Transpose to (batch, embedding_dim, 4) for Flax Conv
        # Flax Conv expects (batch, spatial..., features)
        x = jnp.transpose(x, (0, 2, 1))

        # 1D convolution: (batch, embedding_dim, 4) -> (batch, out_len, num_filters)
        x = self.conv(x)

        # BatchNorm (axis=-1 operates on num_filters dimension)
        x = self.bn(x, use_running_average=not training)

        # ReLU activation
        x = jax.nn.relu(x)

        # Dropout
        x = self.dropout(x, deterministic=not training)

        # Flatten: (batch, out_len * num_filters)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

        # FC projection: (batch, out_len * num_filters) -> (batch, embedding_dim)
        x = self.fc(x)

        # Cast entity embeddings and score
        entity_emb_bf16 = entity_emb.astype(jnp.bfloat16)

        # Score via dot product with ALL entity embeddings
        # (batch, dim) @ (dim, num_entities) -> (batch, num_entities)
        scores = x @ entity_emb_bf16.T

        # Cast back to float32 for numerical stability in downstream softmax
        return scores.astype(jnp.float32)
