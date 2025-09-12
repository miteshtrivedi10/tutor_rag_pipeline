import unittest
from unittest.mock import Mock, patch
import numpy as np
from src.rag.description_generator import ContentDescriptionGenerator
from src.rag.storage import MilvusStorage
from src.rag.openrouter import OpenRouterEmbeddingGenerator


class TestContentDescriptionGenerator(unittest.TestCase):
    """Test cases for ContentDescriptionGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_generator = Mock(spec=OpenRouterEmbeddingGenerator)
        self.storage = Mock(spec=MilvusStorage)
        self.llm_model_func = Mock()
        
        # Configure mock LLM response
        self.llm_model_func.return_value = {
            "text": "This is a meaningful description of the content."
        }
        
        self.description_generator = ContentDescriptionGenerator(
            self.embedding_generator, 
            self.storage, 
            self.llm_model_func
        )
    
    def test_generate_description_success(self):
        """Test successful description generation."""
        content = {
            "text_content": "This is sample educational content about science concepts.",
            "content_type": "text",
            "source_file": "test.pdf",
            "chapter": "Chapter 1",
            "page_number": 1
        }
        
        description = self.description_generator.generate_description(content)
        
        # Verify description was generated
        self.assertEqual(description, "This is a meaningful description of the content.")
    
    def test_generate_description_no_content(self):
        """Test description generation with no content."""
        content = {}
        description = self.description_generator.generate_description(content)
        self.assertEqual(description, "No content available for description.")
    
    def test_generate_description_llm_failure(self):
        """Test description generation when LLM fails."""
        self.llm_model_func.side_effect = Exception("LLM error")
        
        content = {
            "text_content": "This is sample educational content.",
            "content_type": "text"
        }
        
        description = self.description_generator.generate_description(content)
        
        # Should fall back to simple description
        self.assertIn("text content discusses", description)
        self.assertIn("text", description)
    
    def test_generate_simple_description(self):
        """Test simple description generation."""
        text_content = "This is the first sentence. This is the second sentence."
        content_type = "text"
        
        # Use reflection to test private method
        description = self.description_generator._generate_simple_description(
            text_content, content_type
        )
        
        self.assertIn("text content discusses", description)
        self.assertIn("This is the first sentence", description)
    
    def test_generate_simple_description_empty_content(self):
        """Test simple description generation with empty content."""
        text_content = ""
        content_type = "text"
        
        description = self.description_generator._generate_simple_description(
            text_content, content_type
        )
        
        self.assertIn("text content contains educational material", description)


if __name__ == '__main__':
    unittest.main()