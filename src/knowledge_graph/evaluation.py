"""
Embedding Quality Evaluation and Validation

Implements comprehensive evaluation metrics for knowledge graph embeddings:
    - Link prediction (MRR, Hits@K)
    - Ranking metrics
    - Embedding visualization with t-SNE
    - Inference latency benchmarking
    - Comparison against baselines

Target Performance (as per plan):
    - MRR > 0.30 on test set
    - RotatE outperforms TransE by >20%
    - Inference < 1ms per triple
    - Meaningful clusters in visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import time
from pathlib import Path
import json

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.knowledge_graph.embeddings import RotatEModel
from src.knowledge_graph.temporal_embeddings import TemporalRotatEModel


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for knowledge graph embeddings."""

    # Link prediction
    mrr: float  # Mean Reciprocal Rank
    hits_at_1: float  # Hits@1
    hits_at_3: float  # Hits@3
    hits_at_10: float  # Hits@10

    # Ranking
    mean_rank: float  # Average rank of true entity
    median_rank: float  # Median rank

    # Performance
    inference_time_ms: float  # Average inference time per triple (milliseconds)
    throughput: float  # Triples per second

    # Quality
    num_test_samples: int  # Number of test samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Evaluation Metrics:\n"
            f"  MRR: {self.mrr:.4f}\n"
            f"  Hits@1: {self.hits_at_1:.4f}\n"
            f"  Hits@3: {self.hits_at_3:.4f}\n"
            f"  Hits@10: {self.hits_at_10:.4f}\n"
            f"  Mean Rank: {self.mean_rank:.1f}\n"
            f"  Median Rank: {self.median_rank:.1f}\n"
            f"  Inference Time: {self.inference_time_ms:.3f} ms\n"
            f"  Throughput: {self.throughput:.0f} triples/sec\n"
            f"  Test Samples: {self.num_test_samples}"
        )


class EmbeddingEvaluator:
    """
    Evaluator for knowledge graph embeddings.

    Implements link prediction evaluation with filtered ranking to avoid
    penalizing models for predicting other true facts.
    """

    def __init__(
        self,
        model: Union[RotatEModel, TemporalRotatEModel],
        device: str = 'cpu'
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained embedding model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

    def evaluate_link_prediction(
        self,
        test_triples: List[Tuple[int, int, int]],
        all_triples: Optional[List[Tuple[int, int, int]]] = None,
        batch_size: int = 32,
        use_filtered_setting: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate link prediction on test set.

        Uses filtered setting by default: remove all other true triples
        from candidates when ranking, to avoid penalizing correct predictions.

        Args:
            test_triples: List of (head, relation, tail) test triples
            all_triples: All triples (train + test) for filtered evaluation
            batch_size: Batch size for evaluation
            use_filtered_setting: Whether to use filtered ranking

        Returns:
            EvaluationMetrics with all metrics
        """
        print(f"Evaluating link prediction on {len(test_triples)} triples...")

        # Build filtering dictionary if needed
        filter_dict = None
        if use_filtered_setting and all_triples:
            filter_dict = self._build_filter_dict(all_triples)

        # Metrics accumulators
        reciprocal_ranks = []
        ranks = []
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_10 = 0

        total_inference_time = 0.0
        num_predictions = 0

        # Evaluate each triple
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i + batch_size]

            for head, relation, tail in batch:
                # Predict tail (head corruption)
                start_time = time.time()
                rank_tail = self._rank_tail(
                    head, relation, tail, filter_dict
                )
                inference_time = (time.time() - start_time) * 1000  # ms

                # Predict head (tail corruption)
                start_time = time.time()
                rank_head = self._rank_head(
                    head, relation, tail, filter_dict
                )
                inference_time += (time.time() - start_time) * 1000

                total_inference_time += inference_time
                num_predictions += 2  # head and tail prediction

                # Accumulate metrics (average of both directions)
                avg_rank = (rank_tail + rank_head) / 2.0
                ranks.append(avg_rank)

                reciprocal_ranks.append(1.0 / rank_tail)
                reciprocal_ranks.append(1.0 / rank_head)

                if rank_tail <= 1:
                    hits_at_1 += 1
                if rank_tail <= 3:
                    hits_at_3 += 1
                if rank_tail <= 10:
                    hits_at_10 += 1

                if rank_head <= 1:
                    hits_at_1 += 1
                if rank_head <= 3:
                    hits_at_3 += 1
                if rank_head <= 10:
                    hits_at_10 += 1

            # Progress
            if (i + batch_size) % 100 == 0:
                print(f"  Evaluated {i + batch_size}/{len(test_triples)} triples...")

        # Compute final metrics
        num_samples = len(test_triples) * 2  # Both directions

        metrics = EvaluationMetrics(
            mrr=np.mean(reciprocal_ranks),
            hits_at_1=hits_at_1 / num_samples,
            hits_at_3=hits_at_3 / num_samples,
            hits_at_10=hits_at_10 / num_samples,
            mean_rank=np.mean(ranks),
            median_rank=np.median(ranks),
            inference_time_ms=total_inference_time / num_predictions,
            throughput=num_predictions / (total_inference_time / 1000),
            num_test_samples=num_samples
        )

        print("\n" + str(metrics))

        return metrics

    def _build_filter_dict(
        self,
        all_triples: List[Tuple[int, int, int]]
    ) -> Dict[Tuple[int, int], set]:
        """
        Build dictionary mapping (head, relation) -> set of true tails.

        Used for filtered ranking to exclude other true facts.

        Args:
            all_triples: All known triples

        Returns:
            Dictionary for filtering
        """
        filter_dict = {}

        for head, relation, tail in all_triples:
            key = (head, relation)
            if key not in filter_dict:
                filter_dict[key] = set()
            filter_dict[key].add(tail)

        return filter_dict

    def _rank_tail(
        self,
        head: int,
        relation: int,
        true_tail: int,
        filter_dict: Optional[Dict[Tuple[int, int], set]] = None
    ) -> int:
        """
        Rank true tail among all candidate tails.

        Args:
            head: Head entity ID
            relation: Relation ID
            true_tail: True tail entity ID
            filter_dict: Optional filtering dictionary

        Returns:
            Rank of true tail (1 = best)
        """
        with torch.no_grad():
            # Get base model (handle both RotatEModel and TemporalRotatEModel)
            base_model = getattr(self.model, 'base_model', self.model)

            # Get embeddings
            head_emb = base_model.entity_embeddings.weight[head].unsqueeze(0)
            relation_emb = base_model.relation_embeddings.weight[relation].unsqueeze(0)

            # Get all tail embeddings
            all_tail_emb = base_model.entity_embeddings.weight

            # Rotate head by relation
            head_rotated = base_model.complex_multiply(head_emb, relation_emb)

            # Compute distances to all tails
            distances = torch.sum(
                torch.abs(head_rotated - all_tail_emb),
                dim=-1
            )

            # Convert to scores (negative distance)
            scores = -distances

            # Apply filtering if provided
            if filter_dict and (head, relation) in filter_dict:
                true_tails = filter_dict[(head, relation)]
                # Set scores of other true tails to -inf (exclude from ranking)
                for other_tail in true_tails:
                    if other_tail != true_tail:
                        scores[other_tail] = float('-inf')

            # Get rank of true tail
            # Rank = number of entities with higher score + 1
            true_score = scores[true_tail].item()
            rank = (scores > true_score).sum().item() + 1

            return rank

    def _rank_head(
        self,
        true_head: int,
        relation: int,
        tail: int,
        filter_dict: Optional[Dict[Tuple[int, int], set]] = None
    ) -> int:
        """
        Rank true head among all candidate heads.

        Args:
            true_head: True head entity ID
            relation: Relation ID
            tail: Tail entity ID
            filter_dict: Optional filtering dictionary

        Returns:
            Rank of true head (1 = best)
        """
        with torch.no_grad():
            # Get base model (handle both RotatEModel and TemporalRotatEModel)
            base_model = getattr(self.model, 'base_model', self.model)

            # Get embeddings
            relation_emb = base_model.relation_embeddings.weight[relation].unsqueeze(0)
            tail_emb = base_model.entity_embeddings.weight[tail].unsqueeze(0)

            # Get all head embeddings
            all_head_emb = base_model.entity_embeddings.weight

            # Rotate all heads by relation
            relation_expanded = relation_emb.expand(all_head_emb.size(0), -1)
            heads_rotated = base_model.complex_multiply(
                all_head_emb, relation_expanded
            )

            # Compute distances to tail
            distances = torch.sum(
                torch.abs(heads_rotated - tail_emb),
                dim=-1
            )

            # Convert to scores
            scores = -distances

            # Filtering for heads is more complex, skip for now
            # In practice, head prediction is less common

            # Get rank
            true_score = scores[true_head].item()
            rank = (scores > true_score).sum().item() + 1

            return rank


def visualize_embeddings_tsne(
    model: Union[RotatEModel, TemporalRotatEModel],
    entity_to_id: Dict[str, int],
    output_path: str,
    num_samples: Optional[int] = None,
    highlight_entities: Optional[List[str]] = None,
    perplexity: int = 30,
    n_iter: int = 1000
):
    """
    Visualize entity embeddings using t-SNE projection.

    Args:
        model: Trained embedding model
        entity_to_id: Entity name to ID mapping
        output_path: Path to save visualization
        num_samples: Number of entities to visualize (None = all)
        highlight_entities: Optional list of entity names to highlight
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
    """
    print("Generating t-SNE visualization...")

    # Get embeddings
    if isinstance(model, TemporalRotatEModel):
        embeddings = model.base_model.entity_embeddings.weight.detach().cpu().numpy()
    else:
        embeddings = model.entity_embeddings.weight.detach().cpu().numpy()

    # Sample if requested
    entity_names = list(entity_to_id.keys())
    entity_ids = list(entity_to_id.values())

    if num_samples and num_samples < len(entity_names):
        indices = np.random.choice(len(entity_names), num_samples, replace=False)
        entity_names = [entity_names[i] for i in indices]
        entity_ids = [entity_ids[i] for i in indices]
        embeddings = embeddings[entity_ids]
    else:
        num_samples = len(entity_names)

    # Run t-SNE
    print(f"Running t-SNE on {num_samples} entities...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, num_samples - 1), max_iter=n_iter)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create visualization
    plt.figure(figsize=(12, 10))

    # Plot all entities
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        alpha=0.5,
        c='blue',
        s=50
    )

    # Highlight specific entities if requested
    if highlight_entities:
        highlight_indices = []
        for name in highlight_entities:
            if name in entity_names:
                idx = entity_names.index(name)
                highlight_indices.append(idx)

        if highlight_indices:
            highlight_coords = embeddings_2d[highlight_indices]
            plt.scatter(
                highlight_coords[:, 0],
                highlight_coords[:, 1],
                c='red',
                s=200,
                alpha=0.8,
                edgecolors='black',
                linewidths=2
            )

            # Add labels
            for idx, name in zip(highlight_indices, highlight_entities):
                if name in entity_names:
                    plt.annotate(
                        name,
                        (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                        fontsize=10,
                        fontweight='bold'
                    )

    plt.title('Entity Embeddings Visualization (t-SNE)', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def compare_models(
    model1: Union[RotatEModel, TemporalRotatEModel],
    model2: Union[RotatEModel, TemporalRotatEModel],
    test_triples: List[Tuple[int, int, int]],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    all_triples: Optional[List[Tuple[int, int, int]]] = None
) -> Dict[str, Any]:
    """
    Compare two models on link prediction task.

    Args:
        model1: First model
        model2: Second model
        test_triples: Test triples
        model1_name: Name for first model
        model2_name: Name for second model
        all_triples: All triples for filtered evaluation

    Returns:
        Dictionary with comparison results
    """
    print(f"\nComparing {model1_name} vs {model2_name}...")

    # Evaluate both models
    evaluator1 = EmbeddingEvaluator(model1)
    evaluator2 = EmbeddingEvaluator(model2)

    metrics1 = evaluator1.evaluate_link_prediction(test_triples, all_triples)
    metrics2 = evaluator2.evaluate_link_prediction(test_triples, all_triples)

    # Compute improvements
    mrr_improvement = ((metrics1.mrr - metrics2.mrr) / metrics2.mrr) * 100
    hits10_improvement = ((metrics1.hits_at_10 - metrics2.hits_at_10) / metrics2.hits_at_10) * 100

    comparison = {
        model1_name: metrics1.to_dict(),
        model2_name: metrics2.to_dict(),
        "improvements": {
            "mrr_improvement_pct": mrr_improvement,
            "hits10_improvement_pct": hits10_improvement
        }
    }

    print(f"\nComparison Results:")
    print(f"  {model1_name} MRR: {metrics1.mrr:.4f}")
    print(f"  {model2_name} MRR: {metrics2.mrr:.4f}")
    print(f"  MRR Improvement: {mrr_improvement:+.1f}%")
    print(f"\n  {model1_name} Hits@10: {metrics1.hits_at_10:.4f}")
    print(f"  {model2_name} Hits@10: {metrics2.hits_at_10:.4f}")
    print(f"  Hits@10 Improvement: {hits10_improvement:+.1f}%")

    return comparison


def save_evaluation_results(
    metrics: EvaluationMetrics,
    output_path: str,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save evaluation results to JSON file.

    Args:
        metrics: Evaluation metrics
        output_path: Path to save results
        additional_info: Optional additional information
    """
    results = {
        "metrics": metrics.to_dict(),
        "timestamp": time.time()
    }

    if additional_info:
        results.update(additional_info)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
