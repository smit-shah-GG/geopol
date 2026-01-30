"""
RAG Pipeline for Historical Pattern Grounding.

Uses LlamaIndex to index graph patterns and retrieve relevant historical
context for LLM forecasting. Integrates with ChromaDB for vector storage.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

from .graph_pattern_extractor import GraphPatternExtractor
from ..knowledge_graph.graph_builder import TemporalKnowledgeGraph

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG pipeline for historical pattern retrieval.

    Indexes temporal graph patterns and provides similarity search
    for grounding LLM predictions in historical data.
    """

    def __init__(self,
                 persist_dir: str = "./chroma_db",
                 collection_name: str = "graph_patterns",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize RAG pipeline.

        Args:
            persist_dir: Directory for ChromaDB persistence
            collection_name: ChromaDB collection name
            embedding_model: HuggingFace model for embeddings
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Create persist directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_embeddings()
        self._init_vector_store()
        self.index = None
        self.query_engine = None

    def _init_embeddings(self):
        """Initialize embedding model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            cache_folder=str(self.persist_dir / "embeddings_cache")
        )

        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

    def _init_vector_store(self):
        """Initialize ChromaDB vector store."""
        logger.info(f"Initializing ChromaDB at {self.persist_dir}")

        # Create ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )

        # Get or create collection
        try:
            self.chroma_collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.chroma_collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Temporal graph patterns for forecasting"}
            )
            logger.info(f"Created new collection: {self.collection_name}")

        # Create vector store
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )

    def index_graph_patterns(self,
                            graph: TemporalKnowledgeGraph,
                            time_window_days: int = 30,
                            min_pattern_size: int = 3,
                            rebuild: bool = False) -> Dict[str, Any]:
        """
        Index graph patterns for retrieval.

        Args:
            graph: TemporalKnowledgeGraph to extract patterns from
            time_window_days: Days to group events
            min_pattern_size: Minimum edges in pattern
            rebuild: Whether to rebuild index from scratch

        Returns:
            Indexing statistics
        """
        stats = {
            'start_time': datetime.utcnow().isoformat(),
            'patterns_extracted': 0,
            'documents_indexed': 0
        }

        # Clear existing if rebuilding
        if rebuild:
            logger.info("Clearing existing index")
            self.chroma_client.delete_collection(self.collection_name)
            self.chroma_collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Temporal graph patterns for forecasting"}
            )
            self.vector_store = ChromaVectorStore(
                chroma_collection=self.chroma_collection
            )

        # Extract patterns
        logger.info("Extracting graph patterns")
        extractor = GraphPatternExtractor(graph)
        documents = extractor.extract_all_patterns(
            time_window_days=time_window_days,
            min_pattern_size=min_pattern_size
        )

        stats['patterns_extracted'] = len(documents)

        if not documents:
            logger.warning("No patterns extracted from graph")
            return stats

        # Create index
        logger.info(f"Indexing {len(documents)} patterns")
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        stats['documents_indexed'] = len(documents)

        # Note: query_engine creation removed - we use VectorIndexRetriever directly
        # in retrieve_similar_patterns() without needing an LLM

        stats['end_time'] = datetime.utcnow().isoformat()
        logger.info(f"Indexing complete: {stats['documents_indexed']} documents")

        return stats

    def _create_query_engine(self):
        """Create query engine for retrieval."""
        if self.index is None:
            logger.warning("No index available for query engine")
            return

        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10
        )

        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever
        )

    def retrieve_similar_patterns(self,
                                 query: str,
                                 top_k: int = 5,
                                 pattern_type: Optional[str] = None) -> List[Dict]:
        """
        Retrieve similar historical patterns.

        Args:
            query: Query string describing scenario
            top_k: Number of patterns to retrieve
            pattern_type: Optional filter by pattern type

        Returns:
            List of similar patterns with scores
        """
        if self.index is None:
            logger.error("No index available. Run index_graph_patterns first.")
            return []

        # Create retriever with specific top_k
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )

        # Retrieve nodes
        nodes = retriever.retrieve(query)

        # Format results
        results = []
        for node in nodes:
            # Parse metadata and pattern data
            metadata = node.metadata
            pattern_data = {}

            if hasattr(node.node, 'extra_info') and 'pattern_data' in node.node.extra_info:
                try:
                    pattern_data = json.loads(node.node.extra_info['pattern_data'])
                except:
                    pass

            # Filter by pattern type if specified
            if pattern_type and metadata.get('pattern_type') != pattern_type:
                continue

            results.append({
                'score': node.score,
                'text': node.node.text,
                'pattern_type': metadata.get('pattern_type'),
                'metadata': metadata,
                'pattern_data': pattern_data
            })

        return results[:top_k]

    def query_historical_context(self,
                                scenario_description: str,
                                entities: List[str] = None,
                                lookback_days: int = 365) -> Dict[str, Any]:
        """
        Query historical context for scenario.

        Args:
            scenario_description: Natural language scenario
            entities: Optional list of entities to focus on
            lookback_days: Days of history to consider

        Returns:
            Historical context with patterns and insights
        """
        context = {
            'query': scenario_description,
            'entities': entities or [],
            'lookback_days': lookback_days,
            'retrieved_patterns': [],
            'insights': []
        }

        # Build enhanced query
        enhanced_query = scenario_description
        if entities:
            enhanced_query += f" Entities involved: {', '.join(entities)}"

        # Retrieve different pattern types
        pattern_types = ['escalation', 'actor_profile', 'bilateral_history', 'conflict_chain']

        for pattern_type in pattern_types:
            patterns = self.retrieve_similar_patterns(
                enhanced_query,
                top_k=3,
                pattern_type=pattern_type
            )

            for pattern in patterns:
                context['retrieved_patterns'].append({
                    'type': pattern_type,
                    'relevance_score': pattern['score'],
                    'summary': pattern['text'],
                    'data': pattern['pattern_data']
                })

        # Generate insights from patterns
        context['insights'] = self._generate_insights(context['retrieved_patterns'])

        return context

    def _generate_insights(self, patterns: List[Dict]) -> List[str]:
        """Generate insights from retrieved patterns."""
        insights = []

        # Group by pattern type
        by_type = {}
        for pattern in patterns:
            ptype = pattern['type']
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(pattern)

        # Generate type-specific insights
        if 'escalation' in by_type:
            escalations = by_type['escalation']
            avg_duration = sum(
                p['data'].get('duration_days', 0) for p in escalations
            ) / max(1, len(escalations))
            insights.append(
                f"Historical escalations typically span {avg_duration:.0f} days"
            )

        if 'actor_profile' in by_type:
            profiles = by_type['actor_profile']
            conflict_ratios = [
                p['data'].get('conflict_ratio', 0) for p in profiles
                if p['data'].get('conflict_ratio') is not None
            ]
            if conflict_ratios:
                avg_conflict = sum(conflict_ratios) / len(conflict_ratios)
                insights.append(
                    f"Similar actors show {avg_conflict:.1%} conflict engagement rate"
                )

        if 'bilateral_history' in by_type:
            bilateral = by_type['bilateral_history']
            cooperation_rates = [
                p['data'].get('cooperation_ratio', 0) for p in bilateral
                if p['data'].get('cooperation_ratio') is not None
            ]
            if cooperation_rates:
                avg_coop = sum(cooperation_rates) / len(cooperation_rates)
                insights.append(
                    f"Similar bilateral relationships show {avg_coop:.1%} cooperation rate"
                )

        if 'conflict_chain' in by_type:
            chains = by_type['conflict_chain']
            chain_lengths = [
                p['data'].get('total_events', 0) for p in chains
                if p['data'].get('total_events')
            ]
            if chain_lengths:
                avg_length = sum(chain_lengths) / len(chain_lengths)
                insights.append(
                    f"Conflict chains typically involve {avg_length:.0f} sequential events"
                )

        return insights

    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about indexed patterns.

        Returns:
            Index statistics
        """
        stats = {
            'collection_name': self.collection_name,
            'persist_dir': str(self.persist_dir),
            'embedding_model': self.embedding_model_name
        }

        try:
            # Get collection stats
            collection_data = self.chroma_collection.get()
            stats['total_documents'] = len(collection_data['ids']) if collection_data else 0

            # Count by pattern type
            if collection_data and 'metadatas' in collection_data:
                pattern_counts = {}
                for metadata in collection_data['metadatas']:
                    ptype = metadata.get('pattern_type', 'unknown')
                    pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
                stats['patterns_by_type'] = pattern_counts

        except Exception as e:
            logger.error(f"Error getting index statistics: {e}")
            stats['error'] = str(e)

        return stats

    def load_existing_index(self) -> bool:
        """
        Load existing index from storage.

        Returns:
            True if index loaded successfully
        """
        try:
            # Check if collection exists
            collection_data = self.chroma_collection.get()
            if not collection_data or len(collection_data['ids']) == 0:
                logger.warning("No existing index data found")
                return False

            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

            # Load index
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=storage_context
            )

            # Note: query_engine creation removed - we use VectorIndexRetriever directly

            logger.info(f"Loaded existing index with {len(collection_data['ids'])} documents")
            return True

        except Exception as e:
            logger.error(f"Error loading existing index: {e}")
            return False

    def clear_index(self):
        """Clear all indexed data."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.chroma_collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Temporal graph patterns for forecasting"}
            )
            self.vector_store = ChromaVectorStore(
                chroma_collection=self.chroma_collection
            )
            self.index = None
            self.query_engine = None
            logger.info("Index cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing index: {e}")


def create_rag_pipeline(persist_dir: str = "./chroma_db",
                       collection_name: str = "graph_patterns") -> RAGPipeline:
    """
    Create RAG pipeline instance.

    Args:
        persist_dir: Directory for persistence
        collection_name: Collection name

    Returns:
        Initialized RAGPipeline
    """
    return RAGPipeline(persist_dir=persist_dir, collection_name=collection_name)