"""
RE-GCN implementation using local JAX equivalents (jraph-free).

jraph was archived by Google DeepMind in May 2025. This module replaces
jraph.GraphsTuple with a local NamedTuple and jraph.segment_sum with
jax.ops.segment_sum. Behavior is identical.

Key design:
- Local GraphsTuple NamedTuple replaces jraph.GraphsTuple
- jax.ops.segment_sum replaces jraph.segment_sum (same XLA kernel)
- Fully vectorized message passing via jax.lax.fori_loop

Reference:
    Li et al. (2021). Temporal Knowledge Graph Reasoning Based on
    Evolutional Representation Learning. SIGIR 2021.
"""

from typing import Callable, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

Array = jax.Array


class GraphsTuple(NamedTuple):
    """Local replacement for jraph.GraphsTuple (archived library).

    Minimal subset used by RE-GCN: nodes, edges, senders, receivers,
    n_node, n_edge, globals. Same field names as jraph for drop-in
    compatibility.
    """
    nodes: Optional[Array]
    edges: Optional[Array]
    senders: Array
    receivers: Array
    n_node: Array
    n_edge: Array
    globals: Optional[Array] = None


class TemporalGraph(NamedTuple):
    """Temporal graph with relation types."""
    graph: GraphsTuple
    relation_types: Array  # (n_edges,) relation type per edge


def create_graph(
    senders: Array,
    receivers: Array,
    relation_types: Array,
    num_nodes: int,
    node_features: Optional[Array] = None,
) -> TemporalGraph:
    """
    Create a temporal graph from edge data.

    Args:
        senders: Source node indices (n_edges,)
        receivers: Target node indices (n_edges,)
        relation_types: Relation type per edge (n_edges,)
        num_nodes: Total number of nodes
        node_features: Optional node features (num_nodes, dim)

    Returns:
        TemporalGraph with local GraphsTuple
    """
    n_edge = jnp.array([len(senders)])
    n_node = jnp.array([num_nodes])

    # Use relation types as edge features for jraph
    edges = relation_types.astype(jnp.float32)[:, None]  # (n_edges, 1)

    if node_features is None:
        nodes = None
    else:
        nodes = node_features

    graph = GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=None,
    )

    return TemporalGraph(graph=graph, relation_types=relation_types)


class RGCNLayer(nnx.Module):
    """
    Relational Graph Convolution layer using JAX segment operations.

    Uses basis decomposition: W_r = sum_b(coeff[r,b] * basis[b])
    Message passing is vectorized using jax.ops.segment_sum.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_relations: int,
        num_bases: int = 30,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        # Basis decomposition
        self.basis = nnx.Param(
            jax.random.normal(rngs.params(), (self.num_bases, in_features, out_features)) * 0.01
        )
        self.coeff = nnx.Param(
            jax.random.normal(rngs.params(), (num_relations, self.num_bases)) * 0.01
        )

        # Self-loop weight
        self.self_weight = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, out_features)) * 0.01
        )

        # Bias
        self.bias = nnx.Param(jnp.zeros((out_features,)))

    def __call__(
        self,
        node_features: Array,
        graph: TemporalGraph,
    ) -> Array:
        """
        Apply R-GCN layer using memory-efficient per-relation processing.

        Instead of materializing (n_edges, in, out) tensor, we iterate over
        relations and accumulate messages using jax.lax.fori_loop.
        This trades compute for memory, but keeps GPU utilization high.

        Args:
            node_features: Node embeddings (num_nodes, in_features)
            graph: TemporalGraph with edges and relation types

        Returns:
            Updated node features (num_nodes, out_features)
        """
        senders = graph.graph.senders
        receivers = graph.graph.receivers
        relation_types = graph.relation_types
        num_nodes = node_features.shape[0]
        n_edges = senders.shape[0]

        # Compute all relation weight matrices from basis
        # (num_relations, in_features, out_features)
        rel_weights = jnp.einsum('rb,bio->rio', self.coeff.value, self.basis.value)

        # Get source node features once
        src_features = node_features[senders]  # (n_edges, in_features)

        # Memory-efficient: accumulate messages across relations using fori_loop
        # Peak memory: O(n_edges * out_features) instead of O(n_edges * in * out)
        def body_fn(r: int, aggregated: Array) -> Array:
            """Process edges of relation r and accumulate."""
            # Mask for this relation's edges
            mask = (relation_types == r).astype(jnp.float32)[:, None]  # (n_edges, 1)

            # Transform source features through this relation's weight
            # (n_edges, in) @ (in, out) -> (n_edges, out)
            messages = src_features @ rel_weights[r]

            # Zero out messages from other relations
            masked_messages = messages * mask

            # Aggregate to receivers and add to running total
            rel_aggregated = jax.ops.segment_sum(masked_messages, receivers, num_segments=num_nodes)
            return aggregated + rel_aggregated

        # Initialize accumulator
        init_aggregated = jnp.zeros((num_nodes, self.out_features))

        # Iterate over all relations
        aggregated = jax.lax.fori_loop(0, self.num_relations, body_fn, init_aggregated)

        # Compute in-degree for normalization
        ones = jnp.ones(n_edges)
        in_degree = jax.ops.segment_sum(ones, receivers, num_segments=num_nodes)
        in_degree = jnp.maximum(in_degree, 1.0)

        # Normalize by in-degree
        out = aggregated / in_degree[:, None]

        # Self-loop contribution
        out = out + node_features @ self.self_weight.value

        # Bias
        out = out + self.bias.value

        return out


class GRUCell(nnx.Module):
    """GRU cell for temporal evolution."""

    def __init__(self, hidden_dim: int, *, rngs: nnx.Rngs):
        self.hidden_dim = hidden_dim

        # Combined gates for efficiency
        self.W_gates = nnx.Param(
            jax.random.normal(rngs.params(), (hidden_dim, hidden_dim * 3)) * 0.01
        )
        self.U_gates = nnx.Param(
            jax.random.normal(rngs.params(), (hidden_dim, hidden_dim * 3)) * 0.01
        )
        self.b_gates = nnx.Param(jnp.zeros((hidden_dim * 3,)))

    def __call__(self, h: Array, x: Array) -> Array:
        """GRU step: h=previous hidden, x=input."""
        gates = x @ self.W_gates.value + h @ self.U_gates.value + self.b_gates.value

        r, z, h_tilde_pre = jnp.split(gates, 3, axis=-1)
        r = jax.nn.sigmoid(r)
        z = jax.nn.sigmoid(z)

        # Recompute candidate with reset gate
        h_tilde = jnp.tanh(h_tilde_pre + (r * h) @ self.U_gates.value[:, 2*self.hidden_dim:])

        return (1 - z) * h + z * h_tilde


class SimpleDecoder(nnx.Module):
    """
    Simple MLP decoder for link prediction.

    Scores (s, r, o) triples using:
    score = MLP([h_s; h_r; h_o])
    """

    def __init__(
        self,
        embedding_dim: int,
        num_relations: int,
        hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        self.embedding_dim = embedding_dim

        # Relation embeddings
        self.rel_emb = nnx.Param(
            jax.random.normal(rngs.params(), (num_relations, embedding_dim)) * 0.01
        )

        # MLP layers
        input_dim = embedding_dim * 3  # concat of subject, relation, object
        self.fc1 = nnx.Param(
            jax.random.normal(rngs.params(), (input_dim, hidden_dim)) * 0.01
        )
        self.fc1_bias = nnx.Param(jnp.zeros((hidden_dim,)))

        self.fc2 = nnx.Param(
            jax.random.normal(rngs.params(), (hidden_dim, 1)) * 0.01
        )
        self.fc2_bias = nnx.Param(jnp.zeros((1,)))

    def __call__(self, entity_emb: Array, triples: Array) -> Array:
        """
        Score triples.

        Args:
            entity_emb: Entity embeddings (num_entities, dim)
            triples: (batch, 3) of [subject, relation, object]

        Returns:
            Scores (batch,)
        """
        subjects = triples[:, 0]
        relations = triples[:, 1]
        objects = triples[:, 2]

        h_s = entity_emb[subjects]
        h_r = self.rel_emb.value[relations]
        h_o = entity_emb[objects]

        # Concatenate and score
        combined = jnp.concatenate([h_s, h_r, h_o], axis=-1)
        hidden = jax.nn.relu(combined @ self.fc1.value + self.fc1_bias.value)
        scores = (hidden @ self.fc2.value + self.fc2_bias.value).squeeze(-1)

        return scores


class REGCNJraph(nnx.Module):
    """
    RE-GCN using local JAX graph primitives (jraph-free).

    Architecture:
    1. R-GCN layers for graph structure (using jax.ops.segment_sum)
    2. GRU for temporal evolution
    3. Simple MLP decoder for scoring
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout_rate: float = 0.2,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Entity embeddings
        self.entity_emb = nnx.Param(
            jax.random.normal(rngs.params(), (num_entities, embedding_dim)) * 0.01
        )

        # R-GCN layers
        self.rgcn_layers = nnx.List([
            RGCNLayer(
                embedding_dim,
                embedding_dim,
                num_relations * 2,  # Include inverse relations
                num_bases,
                rngs=rngs,
            )
            for _ in range(num_layers)
        ])

        # GRU for temporal evolution
        self.gru = GRUCell(embedding_dim, rngs=rngs)

        # Decoder
        self.decoder = SimpleDecoder(
            embedding_dim,
            num_relations * 2,
            rngs=rngs,
        )

    def encode_snapshot(
        self,
        node_features: Array,
        graph: TemporalGraph,
        training: bool = True,
        dropout_key: Optional[Array] = None,
    ) -> Array:
        """Encode a single graph snapshot through R-GCN layers."""
        x = node_features

        for i, layer in enumerate(self.rgcn_layers):
            x = layer(x, graph)
            x = jax.nn.relu(x)

            if training and self.dropout_rate > 0 and dropout_key is not None:
                key = jax.random.fold_in(dropout_key, i)
                mask = jax.random.bernoulli(key, 1.0 - self.dropout_rate, x.shape)
                x = x * mask / (1.0 - self.dropout_rate)

        return x

    def evolve_embeddings(
        self,
        graphs: List[TemporalGraph],
        training: bool = True,
        rng_key: Optional[Array] = None,
    ) -> Array:
        """
        Evolve entity embeddings through temporal snapshots.

        Args:
            graphs: List of temporal graph snapshots
            training: Whether in training mode
            rng_key: Random key for dropout

        Returns:
            Final entity embeddings
        """
        h = self.entity_emb.value

        for t, graph in enumerate(graphs):
            # Get dropout key for this timestep
            if rng_key is not None:
                dropout_key = jax.random.fold_in(rng_key, t)
            else:
                dropout_key = None

            # Encode current snapshot
            x = self.encode_snapshot(h, graph, training, dropout_key)

            # Temporal evolution via GRU
            h = self.gru(h, x)

        return h

    def compute_scores(self, entity_emb: Array, triples: Array) -> Array:
        """Score triples given entity embeddings."""
        return self.decoder(entity_emb, triples)

    def compute_loss(
        self,
        graphs: List[TemporalGraph],
        pos_triples: Array,
        neg_triples: Array,
        margin: float = 1.0,
        rng_key: Optional[Array] = None,
    ) -> Array:
        """
        Compute margin ranking loss.

        Args:
            graphs: Temporal graph snapshots
            pos_triples: Positive triples (batch, 3)
            neg_triples: Negative triples (batch * num_neg, 3)
            margin: Margin for ranking loss
            rng_key: Random key for dropout

        Returns:
            Scalar loss
        """
        entity_emb = self.evolve_embeddings(graphs, training=True, rng_key=rng_key)

        pos_scores = self.compute_scores(entity_emb, pos_triples)
        neg_scores = self.compute_scores(entity_emb, neg_triples)

        # Reshape for margin loss
        num_pos = pos_triples.shape[0]
        num_neg_per_pos = neg_triples.shape[0] // num_pos
        neg_scores = neg_scores.reshape(num_pos, num_neg_per_pos)

        # Margin ranking loss
        pos_scores_exp = pos_scores[:, None]
        losses = jax.nn.relu(margin - pos_scores_exp + neg_scores)

        return jnp.mean(losses)

    def predict(
        self,
        graphs: List[TemporalGraph],
        query_triples: Array,
    ) -> Array:
        """Predict scores for query triples."""
        entity_emb = self.evolve_embeddings(graphs, training=False)
        return self.compute_scores(entity_emb, query_triples)


def create_model(
    num_entities: int,
    num_relations: int,
    embedding_dim: int = 200,
    num_layers: int = 2,
    seed: int = 0,
) -> REGCNJraph:
    """Factory function to create RE-GCN model."""
    rngs = nnx.Rngs(params=seed, dropout=seed + 1)
    return REGCNJraph(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        rngs=rngs,
    )
