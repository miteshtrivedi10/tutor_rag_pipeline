"""
Simple RAG Processor with clear documentation and reduced complexity.

This module provides a simplified implementation of the RAG (Retrieval-Augmented Generation)
processor with clear documentation and reduced complexity. It's designed to be:
- Easy to understand and maintain
- Well-documented with clear comments
- Focused on core functionality
- Modular and extensible
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from pathlib import Path
from functools import lru_cache

from src.rag.document_parser import DocumentParser
from src.config.settings import settings
from src.utils.file_handler import FileHandler
from src.utils.exceptions import FileProcessingError
from src.processors.image_processor import ImageModalProcessor
from src.processors.table_processor import TableModalProcessor
from src.processors.equation_processor import EquationModalProcessor
from src.processors.generic_processor import GenericModalProcessor
from src.rag.openrouter import OpenRouterClient
from src.rag.nomic_embedding import NomicEmbeddingGenerator
from src.rag.storage import MilvusStorage
from src.rag.performance_monitor import get_global_monitor
from src.rag.questionnaire_generator import QuestionnaireGenerator

logger = logging.getLogger(__name__)


class SimpleRAGProcessor:
    """
    Simplified RAG processor with clear documentation and reduced complexity.
    
    This processor handles document processing, content analysis, and educational
    questionnaire generation in a streamlined way.
    """

    def __init__(self, storage: Optional[MilvusStorage] = None):
        """
        Initialize the simplified RAG processor.
        
        Args:
            storage: Optional Milvus storage instance. If not provided, creates a default one.
        """
        # Core components
        self.parser = DocumentParser()
        self.file_handler = FileHandler()
        
        # Storage (use provided or create default)
        self.storage = storage or MilvusStorage(
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN
        )
        
        # Initialize content processors
        self.processors = {
            "image": ImageModalProcessor(),
            "table": TableModalProcessor(),
            "equation": EquationModalProcessor(),
            "generic": GenericModalProcessor(),
        }
        
        # Initialize embedding generator
        self.embedding_generator = NomicEmbeddingGenerator()
        
        # Initialize questionnaire generator for educational content
        self.questionnaire_generator = QuestionnaireGenerator(
            openrouter_client=OpenRouterClient()
        )
        
        logger.info("SimpleRAGProcessor initialized")

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single document and return processed content items.
        
        Args:
            file_path: Path to the document file to process
            
        Returns:
            List of processed content items with embeddings and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileProcessingError(f"File not found: {file_path}")
        
        try:
            # Step 1: Parse the document
            raw_content = self.parser.parse_document(str(file_path))
            
            if not raw_content:
                logger.warning(f"No content extracted from {file_path}")
                return []
            
            # Step 2: Process content items
            processed_content = []
            for item in raw_content:
                processed_item = self._process_content_item(item)
                if processed_item:
                    processed_content.append(processed_item)
            
            # Step 3: Generate embeddings
            content_with_embeddings = []
            for item in processed_content:
                item_with_embedding = self._add_embedding(item)
                if item_with_embedding:
                    content_with_embeddings.append(item_with_embedding)
            
            # Step 4: Store in database
            if content_with_embeddings:
                self._store_content(content_with_embeddings)
                
                # Step 5: Generate educational questionnaires
                self.questionnaire_generator.generate_and_print_questionnaires(
                    content_with_embeddings
                )
            
            logger.info(f"Processed {len(content_with_embeddings)} items from {file_path}")
            return content_with_embeddings
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise FileProcessingError(f"Failed to process document: {e}")

    def _process_content_item(self, content_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single content item through the appropriate processor.
        
        Args:
            content_item: Content item to process
            
        Returns:
            Processed content item or None if processing fails
        """
        content_type = content_item.get("type", "generic")
        
        # Get appropriate processor (fallback to generic if type not found)
        processor = self.processors.get(content_type, self.processors["generic"])
        
        try:
            # Generate enhanced description for the content
            if content_type == "image":
                # Images get special multimodal processing
                enhanced_item = processor.process_multimodal_content(content_item)
            else:
                # Other content types get text-based enhancement
                enhanced_item = processor.generate_description_only(content_item)
            
            if not enhanced_item:
                return None
            
            # Add source information
            enhanced_item["source_file"] = content_item.get("source_file", "")
            enhanced_item["page_id"] = content_item.get("page", 1)
            
            # Ensure we have text content for storage
            if "text_content" not in enhanced_item:
                enhanced_item["text_content"] = (
                    enhanced_item.get("enhanced_text", "") or
                    enhanced_item.get("text", "") or
                    f"Content from {enhanced_item.get('source_file', 'unknown')}"
                )
            
            return enhanced_item
            
        except Exception as e:
            logger.error(f"Error processing {content_type} item: {e}")
            return None

    def _add_embedding(self, content_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add embedding to a content item.
        
        Args:
            content_item: Content item to add embedding to
            
        Returns:
            Content item with embedding or None if embedding fails
        """
        try:
            # Extract text content for embedding
            text_content = content_item.get("text_content", "")
            if not text_content:
                logger.warning("No text content for embedding generation")
                return None
            
            # Generate embedding
            embeddings = self.embedding_generator.generate_embeddings([text_content])
            if not embeddings or len(embeddings) == 0:
                logger.warning("Failed to generate embedding")
                return None
            
            # Add embedding to content item
            content_item["embedding"] = embeddings[0].tolist()
            return content_item
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def _store_content(self, content_items: List[Dict[str, Any]]):
        """
        Store content items with embeddings in the database.
        
        Args:
            content_items: List of content items to store
        """
        try:
            # Prepare data for batch insertion
            embeddings_data = []
            for item in content_items:
                if "embedding" in item:
                    embeddings_data.append({
                        "embedding": item["embedding"],
                        "text_content": item["text_content"],
                        "content_type": item.get("type", "generic"),
                        "source_file": item.get("source_file", ""),
                        "page_id": str(item.get("page_id", 1)),
                        "metadata": item.get("metadata", {}),
                    })
            
            # Insert in batch
            if embeddings_data:
                doc_ids = self.storage.insert_batch(embeddings_data)
                logger.info(f"Stored {len(doc_ids)} content items")
                
        except Exception as e:
            logger.error(f"Error storing content: {e}")

    def search_similar_content(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for content similar to the query.
        
        Args:
            query: Text query to search for similar content
            top_k: Number of similar items to return
            
        Returns:
            List of similar content items
        """
        try:
            # Generate query embedding
            query_embeddings = self.embedding_generator.generate_embeddings([query])
            if not query_embeddings or len(query_embeddings) == 0:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Search in storage
            results = self.storage.search_similar_content(
                query_embeddings[0].tolist(),
                top_k=top_k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar content: {e}")
            return []