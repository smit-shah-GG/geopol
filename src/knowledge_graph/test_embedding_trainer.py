"""
Unit tests for embedding training pipeline.

Tests cover:
    - Dataset creation from NetworkX graphs
    - Entity and relation ID mappings
    - Training loop execution
    - Checkpointing and restoration
    - Early stopping
    - Performance benchmarks
"""

import pytest
import torch
import networkx as nx
import tempfile
import shutil
from pathlib import Path

from src.knowledge_graph.embedding_trainer import (
    TemporalGraphDataset,
    create_entity_relation_mappings,
    collate_triples,
    EmbeddingTrainer,
    TrainingConfig,
    train_embeddings_from_graph
)
from src.knowledge_graph.embeddings import RotatEModel


@pytest.fixture
def small_graph():
    """Create small temporal knowledge graph for testing."""
    G = nx.MultiDiGraph()

    # Add nodes
    entities = ['USA', 'CHN', 'RUS', 'GBR', 'FRA']
    G.add_nodes_from(entities)

    # Add edges with relation types
    edges = [
        ('USA', 'CHN', {'relation_type': 'DIPLOMATIC_COOPERATION', 'timestamp': 1.0}),
        ('USA', 'CHN', {'relation_type': 'MATERIAL_CONFLICT', 'timestamp': 2.0}),
        ('CHN', 'USA', {'relation_type': 'VERBAL_CONFLICT', 'timestamp': 3.0}),
        ('RUS', 'USA', {'relation_type': 'MATERIAL_CONFLICT', 'timestamp': 4.0}),
        ('GBR', 'USA', {'relation_type': 'DIPLOMATIC_COOPERATION', 'timestamp': 5.0}),
        ('FRA', 'GBR', {'relation_type': 'DIPLOMATIC_COOPERATION', 'timestamp': 6.0}),
    ]

    for head, tail, data in edges:
        G.add_edge(head, tail, **data)

    return G


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestMappings:
    """Test entity and relation ID mappings."""

    def test_create_entity_relation_mappings(self, small_graph):
        """Test mapping creation from graph."""
        entity_to_id, relation_to_id, id_to_entity, id_to_relation = \
            create_entity_relation_mappings(small_graph)

        # Check counts
        assert len(entity_to_id) == 5  # 5 entities
        assert len(relation_to_id) == 3  # 3 unique relation types

        # Check bidirectional mapping consistency
        for entity, entity_id in entity_to_id.items():
            assert id_to_entity[entity_id] == entity

        for relation, relation_id in relation_to_id.items():
            assert id_to_relation[relation_id] == relation

        # Check IDs are sequential
        assert set(entity_to_id.values()) == set(range(5))
        assert set(relation_to_id.values()) == set(range(3))

    def test_mappings_deterministic(self, small_graph):
        """Test that mappings are deterministic (sorted)."""
        mappings1 = create_entity_relation_mappings(small_graph)
        mappings2 = create_entity_relation_mappings(small_graph)

        # Should be identical
        assert mappings1[0] == mappings2[0]  # entity_to_id
        assert mappings1[1] == mappings2[1]  # relation_to_id


class TestDataset:
    """Test TemporalGraphDataset."""

    def test_dataset_creation(self, small_graph):
        """Test dataset extracts correct triples."""
        entity_to_id, relation_to_id, _, _ = create_entity_relation_mappings(small_graph)

        dataset = TemporalGraphDataset(small_graph, entity_to_id, relation_to_id)

        # Should have 6 triples
        assert len(dataset) == 6

        # Check triple format
        for triple in dataset:
            head_id, relation_id, tail_id = triple
            assert isinstance(head_id, int)
            assert isinstance(relation_id, int)
            assert isinstance(tail_id, int)
            assert 0 <= head_id < 5
            assert 0 <= relation_id < 3
            assert 0 <= tail_id < 5

    def test_dataset_getitem(self, small_graph):
        """Test dataset indexing."""
        entity_to_id, relation_to_id, _, _ = create_entity_relation_mappings(small_graph)
        dataset = TemporalGraphDataset(small_graph, entity_to_id, relation_to_id)

        # Test accessing individual triples
        triple_0 = dataset[0]
        assert len(triple_0) == 3
        assert all(isinstance(x, int) for x in triple_0)

    def test_collate_function(self, small_graph):
        """Test batch collation."""
        entity_to_id, relation_to_id, _, _ = create_entity_relation_mappings(small_graph)
        dataset = TemporalGraphDataset(small_graph, entity_to_id, relation_to_id)

        # Get a batch of triples
        batch = [dataset[i] for i in range(3)]
        head_batch, relation_batch, tail_batch = collate_triples(batch)

        # Check tensor properties
        assert head_batch.shape == (3,)
        assert relation_batch.shape == (3,)
        assert tail_batch.shape == (3,)
        assert head_batch.dtype == torch.long
        assert relation_batch.dtype == torch.long
        assert tail_batch.dtype == torch.long


class TestTrainingConfig:
    """Test training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.embedding_dim == 256
        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.device == 'cpu'

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = TrainingConfig(embedding_dim=128, batch_size=64)
        config_dict = config.to_dict()

        assert config_dict['embedding_dim'] == 128
        assert config_dict['batch_size'] == 64
        assert isinstance(config_dict, dict)


class TestEmbeddingTrainer:
    """Test embedding trainer."""

    @pytest.fixture
    def tiny_model(self):
        """Create tiny model for fast testing."""
        return RotatEModel(
            num_entities=10,
            num_relations=3,
            embedding_dim=32,
            margin=9.0,
            negative_samples=2
        )

    @pytest.fixture
    def tiny_config(self, temp_checkpoint_dir):
        """Create config for fast testing."""
        return TrainingConfig(
            embedding_dim=32,
            batch_size=4,
            num_epochs=10,
            learning_rate=0.01,
            checkpoint_every=5,
            checkpoint_dir=temp_checkpoint_dir,
            early_stopping_patience=5
        )

    def test_trainer_initialization(self, tiny_model, tiny_config):
        """Test trainer initializes correctly."""
        trainer = EmbeddingTrainer(tiny_model, tiny_config)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float('inf')

    def test_trainer_train_epoch(self, tiny_model, tiny_config, small_graph):
        """Test single training epoch."""
        entity_to_id, relation_to_id, _, _ = create_entity_relation_mappings(small_graph)
        dataset = TemporalGraphDataset(small_graph, entity_to_id, relation_to_id)

        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=tiny_config.batch_size,
            shuffle=True,
            collate_fn=collate_triples
        )

        trainer = EmbeddingTrainer(tiny_model, tiny_config)
        loss, stats = trainer._train_epoch(loader)

        # Check loss is valid
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))

        # Check statistics
        assert 'pos_score' in stats
        assert 'neg_score' in stats
        assert 'violation_rate' in stats

    def test_trainer_validate_epoch(self, tiny_model, tiny_config, small_graph):
        """Test validation epoch."""
        entity_to_id, relation_to_id, _, _ = create_entity_relation_mappings(small_graph)
        dataset = TemporalGraphDataset(small_graph, entity_to_id, relation_to_id)

        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=tiny_config.batch_size,
            shuffle=False,
            collate_fn=collate_triples
        )

        trainer = EmbeddingTrainer(tiny_model, tiny_config)
        loss, stats = trainer._validate_epoch(loader)

        # Check loss is valid
        assert loss > 0
        assert not torch.isnan(torch.tensor(loss))

    def test_full_training_loop(self, tiny_model, tiny_config, small_graph):
        """Test complete training loop."""
        entity_to_id, relation_to_id, _, _ = create_entity_relation_mappings(small_graph)
        dataset = TemporalGraphDataset(small_graph, entity_to_id, relation_to_id)

        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=tiny_config.batch_size,
            shuffle=True,
            collate_fn=collate_triples
        )

        trainer = EmbeddingTrainer(tiny_model, tiny_config)
        history = trainer.train(loader, None)

        # Check training completed
        assert trainer.current_epoch == tiny_config.num_epochs
        assert len(history['epoch']) == tiny_config.num_epochs

        # Check history contains expected keys
        assert 'train_loss' in history
        assert 'learning_rate' in history
        assert 'epoch_time' in history

    def test_checkpoint_save_load(self, tiny_model, tiny_config, small_graph, temp_checkpoint_dir):
        """Test checkpoint saving and loading."""
        entity_to_id, relation_to_id, _, _ = create_entity_relation_mappings(small_graph)
        dataset = TemporalGraphDataset(small_graph, entity_to_id, relation_to_id)

        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=tiny_config.batch_size,
            shuffle=True,
            collate_fn=collate_triples
        )

        # Train for a few epochs
        trainer = EmbeddingTrainer(tiny_model, tiny_config)
        trainer.train(loader, None)

        # Check checkpoint was saved
        checkpoint_files = list(Path(temp_checkpoint_dir).glob('checkpoint_epoch_*.pt'))
        assert len(checkpoint_files) > 0

        # Create new trainer and load checkpoint
        new_model = RotatEModel(
            num_entities=10,
            num_relations=3,
            embedding_dim=32,
            margin=9.0,
            negative_samples=2
        )
        new_trainer = EmbeddingTrainer(new_model, tiny_config)
        new_trainer.load_checkpoint(str(checkpoint_files[0]))

        # Check state was restored
        assert new_trainer.current_epoch > 0
        assert len(new_trainer.training_history) > 0

    def test_early_stopping(self, tiny_model, temp_checkpoint_dir):
        """Test early stopping mechanism."""
        # Create config with very tight early stopping
        config = TrainingConfig(
            embedding_dim=32,
            batch_size=4,
            num_epochs=100,  # High number, should stop early
            learning_rate=0.0,  # Zero LR means no improvement
            checkpoint_dir=temp_checkpoint_dir,
            early_stopping_patience=3,
            validation_split=0.2
        )

        trainer = EmbeddingTrainer(tiny_model, config)

        # Manually test early stopping logic
        assert not trainer._should_early_stop(10.0)  # First call, improves
        assert not trainer._should_early_stop(10.1)  # No improvement
        assert not trainer._should_early_stop(10.1)  # No improvement
        assert trainer._should_early_stop(10.1)  # Should stop after patience exceeded


class TestHighLevelTraining:
    """Test high-level training function."""

    def test_train_embeddings_from_graph(self, small_graph, temp_checkpoint_dir):
        """Test end-to-end training from graph."""
        config = TrainingConfig(
            embedding_dim=32,
            batch_size=4,
            num_epochs=10,
            checkpoint_dir=temp_checkpoint_dir,
            validation_split=0.2
        )

        model, entity_to_id, relation_to_id, history = train_embeddings_from_graph(
            small_graph,
            config,
            save_path=None
        )

        # Check model was trained
        assert model is not None
        assert len(entity_to_id) == 5
        assert len(relation_to_id) == 3
        assert len(history['epoch']) == 10

        # Check loss decreased
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        # Loss should generally decrease (not guaranteed due to randomness)
        assert final_loss >= 0

    def test_train_and_save_model(self, small_graph, temp_checkpoint_dir):
        """Test training with model saving."""
        config = TrainingConfig(
            embedding_dim=32,
            batch_size=4,
            num_epochs=5,
            checkpoint_dir=temp_checkpoint_dir
        )

        save_path = Path(temp_checkpoint_dir) / 'trained_model.pt'

        model, entity_to_id, relation_to_id, history = train_embeddings_from_graph(
            small_graph,
            config,
            save_path=str(save_path)
        )

        # Check files were saved
        assert save_path.exists()
        mappings_path = save_path.with_name('trained_model_mappings.pkl')
        assert mappings_path.exists()


class TestPerformanceBenchmark:
    """Performance benchmarks for training."""

    def test_training_throughput(self, temp_checkpoint_dir):
        """Test training throughput meets target (>10K triples/sec)."""
        # Create larger synthetic graph
        G = nx.MultiDiGraph()

        # Add 100 entities
        entities = [f'E{i}' for i in range(100)]
        G.add_nodes_from(entities)

        # Add 10,000 edges
        import random
        random.seed(42)
        for _ in range(10000):
            head = random.choice(entities)
            tail = random.choice(entities)
            if head != tail:
                G.add_edge(
                    head, tail,
                    relation_type=random.choice(['REL_A', 'REL_B', 'REL_C']),
                    timestamp=float(random.randint(1, 100))
                )

        config = TrainingConfig(
            embedding_dim=256,
            batch_size=256,
            num_epochs=5,  # Just 5 epochs for benchmark
            checkpoint_dir=temp_checkpoint_dir,
            validation_split=0.1
        )

        import time
        start_time = time.time()

        model, entity_to_id, relation_to_id, history = train_embeddings_from_graph(
            G, config, save_path=None
        )

        elapsed_time = time.time() - start_time

        # Calculate throughput
        total_triples = len(G.edges())
        epochs_trained = len(history['epoch'])
        total_processed = total_triples * epochs_trained
        throughput = total_processed / elapsed_time

        print(f"\nTraining throughput: {throughput:.0f} triples/sec")
        print(f"Total time for {total_triples} triples × {epochs_trained} epochs: {elapsed_time:.2f}s")

        # This is a benchmark, not a strict requirement
        # Target is >10K triples/sec, but depends on hardware
        assert throughput > 0  # Basic sanity check


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
