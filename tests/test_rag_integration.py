"""
Tests for RAG pipeline integration with reasoning orchestrator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from src.forecasting.rag_pipeline import RAGPipeline
from src.forecasting.reasoning_orchestrator import ReasoningOrchestrator
from src.forecasting.models import Scenario, Entity, ScenarioTree
from src.knowledge_graph.graph_builder import TemporalKnowledgeGraph


class TestRAGIntegration:
    """Test RAG integration with reasoning orchestrator."""

    def test_rag_pipeline_initialization(self):
        """Test RAG pipeline initialization."""
        rag = RAGPipeline(persist_dir="./test_chroma_db")

        assert rag.persist_dir.name == "test_chroma_db"
        assert rag.collection_name == "graph_patterns"
        assert rag.embed_model is not None
        assert rag.vector_store is not None

    def test_orchestrator_with_rag(self):
        """Test orchestrator initialization with RAG."""
        rag = Mock(spec=RAGPipeline)
        rag.index = Mock()  # Simulate indexed data

        # Mock GeminiClient to avoid API key requirement
        mock_client = Mock()

        orchestrator = ReasoningOrchestrator(client=mock_client, rag_pipeline=rag, enable_rag=True)

        assert orchestrator.rag_pipeline is not None
        assert orchestrator.rag_pipeline == rag

    def test_orchestrator_without_rag(self):
        """Test orchestrator initialization without RAG."""
        # Mock GeminiClient to avoid API key requirement
        mock_client = Mock()

        orchestrator = ReasoningOrchestrator(client=mock_client, enable_rag=False)

        assert orchestrator.rag_pipeline is None

    @patch('src.forecasting.reasoning_orchestrator.RAGPipeline')
    def test_validation_with_rag(self, mock_rag_class):
        """Test scenario validation using RAG pipeline."""
        # Mock GeminiClient
        mock_client = Mock()

        # Create mock RAG pipeline
        mock_rag = Mock(spec=RAGPipeline)
        mock_rag.index = Mock()  # Simulate indexed data
        mock_rag.query_historical_context.return_value = {
            'query': 'Test scenario',
            'entities': ['USA', 'China'],
            'lookback_days': 730,
            'retrieved_patterns': [
                {
                    'type': 'escalation',
                    'relevance_score': 0.8,
                    'summary': 'Historical escalation pattern detected',
                    'data': {'event_count': 10}
                },
                {
                    'type': 'bilateral_history',
                    'relevance_score': 0.7,
                    'summary': 'Bilateral relationship history',
                    'data': {'cooperation_ratio': 0.3}
                }
            ],
            'insights': [
                'Historical escalations typically span 30 days',
                'Similar actors show 40% conflict engagement rate'
            ]
        }

        # Create orchestrator with mock RAG and client
        orchestrator = ReasoningOrchestrator(client=mock_client, rag_pipeline=mock_rag, enable_rag=True)

        # Create test scenario
        scenario = Scenario(
            scenario_id='test_1',
            description='Conflict scenario between USA and China',
            probability=0.6,
            entities=[
                Entity(name='USA', type='COUNTRY', role='Primary'),
                Entity(name='China', type='COUNTRY', role='Secondary')
            ]
        )

        # Validate scenario
        feedback = orchestrator._validate_with_rag('test_1', scenario)

        # Verify RAG was called
        mock_rag.query_historical_context.assert_called_once_with(
            scenario_description='Conflict scenario between USA and China',
            entities=['USA', 'China'],
            lookback_days=730
        )

        # Check feedback
        assert feedback.scenario_id == 'test_1'
        assert feedback.is_valid == True  # No contradictions
        assert feedback.confidence_score == min(0.95, 0.75)  # Average of 0.8 and 0.7
        assert len(feedback.historical_patterns) == 2
        assert len(feedback.suggestions) == 2

    def test_validation_fallback_to_mock(self):
        """Test validation falls back to mock when RAG not available."""
        # Mock GeminiClient
        mock_client = Mock()

        # Create orchestrator without RAG or Graph validation
        orchestrator = ReasoningOrchestrator(
            client=mock_client,
            enable_rag=False,
            enable_graph_validation=False
        )

        # Create test scenario
        scenario = Scenario(
            scenario_id='test_2',
            description='Test scenario',
            probability=0.5,
            entities=[]
        )

        # Create mock state
        from src.forecasting.reasoning_orchestrator import ReasoningState
        state = ReasoningState(
            question='Test question',
            context=[],
            initial_scenarios=ScenarioTree(
                question='Test question',
                root_scenario=scenario,
                scenarios={'test_2': scenario}
            )
        )

        # Validate scenarios
        feedback_list = orchestrator._validate_scenarios(state)

        # Check that mock validation was used
        assert len(feedback_list) == 1
        assert feedback_list[0].scenario_id == 'test_2'
        assert feedback_list[0].is_valid == True  # probability > 0.3
        assert 'validate_scenarios' in state.step_outputs
        assert 'Mock' in state.step_outputs['validate_scenarios']['output']['validation_methods']

    def test_index_graph_patterns(self):
        """Test indexing graph patterns."""
        # Create RAG pipeline
        rag = RAGPipeline(persist_dir="./test_chroma_db")

        # Create mock documents
        mock_documents = [
            Mock(text='Pattern 1', metadata={'pattern_type': 'escalation'}),
            Mock(text='Pattern 2', metadata={'pattern_type': 'actor_profile'})
        ]

        # Patch GraphPatternExtractor
        with patch('src.forecasting.rag_pipeline.GraphPatternExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor.extract_all_patterns.return_value = mock_documents
            mock_extractor_class.return_value = mock_extractor

            # Create mock graph
            mock_graph = Mock(spec=TemporalKnowledgeGraph)

            # Mock the index creation
            with patch.object(rag, '_create_query_engine'):
                with patch('src.forecasting.rag_pipeline.VectorStoreIndex') as mock_index:
                    mock_index.from_documents.return_value = Mock()

                    # Index patterns
                    stats = rag.index_graph_patterns(mock_graph)

            # Verify extraction was called
            mock_extractor.extract_all_patterns.assert_called_once()

        # Check stats
        assert stats['patterns_extracted'] == 2
        assert stats['documents_indexed'] == 2

    def test_retrieve_similar_patterns(self):
        """Test retrieving similar patterns."""
        # Create RAG pipeline
        rag = RAGPipeline(persist_dir="./test_chroma_db")

        # Mock index and retriever
        mock_node = Mock()
        mock_node.score = 0.85
        mock_node.node.text = 'Similar pattern text'
        mock_node.metadata = {'pattern_type': 'escalation'}
        mock_node.node.extra_info = {'pattern_data': json.dumps({'test': 'data'})}

        mock_retriever = Mock()
        mock_retriever.retrieve.return_value = [mock_node]

        mock_index = Mock()
        rag.index = mock_index

        with patch('src.forecasting.rag_pipeline.VectorIndexRetriever', return_value=mock_retriever):
            # Retrieve patterns
            results = rag.retrieve_similar_patterns('test query', top_k=1)

        # Check results
        assert len(results) == 1
        assert results[0]['score'] == 0.85
        assert results[0]['text'] == 'Similar pattern text'
        assert results[0]['pattern_type'] == 'escalation'
        assert results[0]['pattern_data']['test'] == 'data'

    def test_query_historical_context(self):
        """Test querying historical context."""
        # Create RAG pipeline
        rag = RAGPipeline(persist_dir="./test_chroma_db")

        # Mock retrieve_similar_patterns
        with patch.object(rag, 'retrieve_similar_patterns') as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    'type': 'escalation',
                    'score': 0.9,
                    'text': 'Escalation pattern',
                    'pattern_data': {'duration_days': 30}
                }
            ]

            # Query context
            context = rag.query_historical_context(
                'Test scenario',
                entities=['Entity1', 'Entity2']
            )

        # Check context
        assert context['query'] == 'Test scenario'
        assert context['entities'] == ['Entity1', 'Entity2']
        assert len(context['retrieved_patterns']) > 0
        assert len(context['insights']) >= 0

    def test_full_forecast_with_rag(self):
        """Test full forecasting pipeline with RAG integration."""
        # Create mock RAG
        mock_rag = Mock(spec=RAGPipeline)
        mock_rag.index = Mock()
        mock_rag.query_historical_context.return_value = {
            'retrieved_patterns': [
                {
                    'type': 'escalation',
                    'relevance_score': 0.85,
                    'summary': 'Pattern summary',
                    'data': {}
                }
            ],
            'insights': ['Insight 1']
        }

        # Create mock generator
        mock_generator = Mock()
        mock_scenario = Scenario(
            scenario_id='s1',
            description='Test scenario',
            probability=0.7,
            entities=[]
        )
        mock_tree = ScenarioTree(
            question='Test?',
            root_scenario=mock_scenario,
            scenarios={'s1': mock_scenario}
        )
        mock_generator.generate_scenarios.return_value = mock_tree
        mock_generator.refine_scenarios.return_value = mock_tree

        # Mock GeminiClient
        mock_client = Mock()

        # Create orchestrator
        orchestrator = ReasoningOrchestrator(
            client=mock_client,
            generator=mock_generator,
            rag_pipeline=mock_rag,
            enable_rag=True
        )

        # Run forecast
        forecast = orchestrator.forecast(
            question='Will there be conflict?',
            enable_validation=True,
            enable_refinement=True
        )

        # Verify RAG was used in validation
        assert mock_rag.query_historical_context.called

        # Check forecast
        assert forecast.question == 'Will there be conflict?'
        assert forecast.probability > 0
        assert forecast.confidence > 0
        assert len(forecast.selected_scenario_ids) > 0