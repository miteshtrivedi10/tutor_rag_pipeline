import unittest
from unittest.mock import Mock, patch
import numpy as np
from src.rag.qa_manager import QAManager
from src.rag.storage import MilvusStorage
from src.rag.openrouter import OpenRouterEmbeddingGenerator


class TestQAManager(unittest.TestCase):
    """Test cases for QAManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_generator = Mock(spec=OpenRouterEmbeddingGenerator)
        self.storage = Mock(spec=MilvusStorage)
        self.llm_model_func = Mock()
        
        # Configure mock responses
        self.llm_model_func.return_value = {
            "text": "Generated response from LLM"
        }
        
        self.qa_manager = QAManager(
            self.embedding_generator,
            self.storage,
            self.llm_model_func
        )
    
    def test_generate_qa_pairs_success(self):
        """Test successful Q&A pair generation."""
        # Mock the QA generator's methods
        self.qa_manager.qa_generator.generate_questions = Mock(return_value=[
            {"question": "What is this?", "type": "qa"},
            {"question": "How does it work?", "type": "qa"}
        ])
        self.qa_manager.qa_generator.generate_answers = Mock(return_value=[
            {"question": "What is this?", "type": "qa", "answer": "This is an answer."},
            {"question": "How does it work?", "type": "qa", "answer": "It works like this."}
        ])
        
        content = {
            "text_content": "This is sample educational content.",
            "content_id": "test_1"
        }
        
        qa_pairs = self.qa_manager.generate_qa_pairs(content, 2)
        
        # Verify Q&A pairs were generated
        self.assertEqual(len(qa_pairs), 2)
        self.assertEqual(qa_pairs[0]["question"], "What is this?")
        self.assertEqual(qa_pairs[0]["answer"], "This is an answer.")
        self.assertEqual(qa_pairs[1]["question"], "How does it work?")
        self.assertEqual(qa_pairs[1]["answer"], "It works like this.")
    
    def test_generate_qa_pairs_no_questions(self):
        """Test Q&A pair generation when no questions are generated."""
        # Mock the QA generator to return no questions
        self.qa_manager.qa_generator.generate_questions = Mock(return_value=[])
        
        content = {
            "text_content": "This is sample educational content.",
            "content_id": "test_1"
        }
        
        qa_pairs = self.qa_manager.generate_qa_pairs(content, 2)
        self.assertEqual(qa_pairs, [])
    
    def test_generate_quiz_questions_success(self):
        """Test successful quiz question generation."""
        # Mock the quiz generator's method
        self.qa_manager.quiz_generator.generate_questions = Mock(return_value=[
            {
                "question": "What is this?",
                "options": ["A", "B", "C", "D"],
                "type": "quiz"
            }
        ])
        
        content = {
            "text_content": "This is sample educational content.",
            "content_id": "test_1"
        }
        
        quiz_questions = self.qa_manager.generate_quiz_questions(content, 1)
        
        # Verify quiz questions were generated
        self.assertEqual(len(quiz_questions), 1)
        self.assertEqual(quiz_questions[0]["question"], "What is this?")
        self.assertIn("options", quiz_questions[0])
    
    def test_generate_content_description_success(self):
        """Test successful content description generation."""
        # Mock the description generator's method
        self.qa_manager.description_generator.generate_description = Mock(
            return_value="This is a meaningful description."
        )
        
        content = {
            "text_content": "This is sample educational content.",
            "content_id": "test_1"
        }
        
        description = self.qa_manager.generate_content_description(content)
        self.assertEqual(description, "This is a meaningful description.")
    
    def test_generate_learning_material_success(self):
        """Test successful learning material generation."""
        # Mock all generator methods
        self.qa_manager.qa_generator.generate_questions = Mock(return_value=[
            {"question": "What is this?", "type": "qa"}
        ])
        self.qa_manager.qa_generator.generate_answers = Mock(return_value=[
            {"question": "What is this?", "type": "qa", "answer": "This is an answer."}
        ])
        self.qa_manager.quiz_generator.generate_questions = Mock(return_value=[
            {
                "question": "Quiz question?",
                "options": ["A", "B", "C", "D"],
                "type": "quiz"
            }
        ])
        self.qa_manager.description_generator.generate_description = Mock(
            return_value="This is a meaningful description."
        )
        
        content = {
            "text_content": "This is sample educational content.",
            "content_id": "test_1",
            "content_type": "text",
            "source_file": "test.pdf",
            "chapter": "Chapter 1",
            "page_number": 1
        }
        
        learning_material = self.qa_manager.generate_learning_material(content, 1, 1)
        
        # Verify learning material was generated
        self.assertEqual(learning_material["content_id"], "test_1")
        self.assertEqual(learning_material["content_type"], "text")
        self.assertEqual(learning_material["description"], "This is a meaningful description.")
        self.assertEqual(len(learning_material["qa_pairs"]), 1)
        self.assertEqual(len(learning_material["quiz_questions"]), 1)
    
    def test_generate_learning_material_for_similar_content_success(self):
        """Test successful learning material generation for similar content."""
        # Mock storage search
        self.storage.search_similar = Mock(return_value=[
            {
                "content_id": "similar_1",
                "text_content": "Similar content 1",
                "content_type": "text"
            },
            {
                "content_id": "similar_2",
                "text_content": "Similar content 2",
                "content_type": "text"
            }
        ])
        
        # Mock embedding generation
        self.embedding_generator.generate_embedding = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        
        # Mock learning material generation
        self.qa_manager.generate_learning_material = Mock(return_value={
            "content_id": "similar_1",
            "description": "Description",
            "qa_pairs": [{"question": "Q1", "answer": "A1"}],
            "quiz_questions": [{"question": "Quiz1", "options": ["A", "B", "C", "D"]}]
        })
        
        learning_materials = self.qa_manager.generate_learning_material_for_similar_content(
            "test query", 2, 1, 1
        )
        
        # Verify learning materials were generated
        self.assertEqual(len(learning_materials), 2)
        self.storage.search_similar.assert_called_once()
    
    def test_generate_learning_material_for_similar_content_no_results(self):
        """Test learning material generation when no similar content is found."""
        # Mock storage search to return no results
        self.storage.search_similar = Mock(return_value=[])
        
        # Mock embedding generation
        self.embedding_generator.generate_embedding = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        
        learning_materials = self.qa_manager.generate_learning_material_for_similar_content(
            "test query", 2, 1, 1
        )
        
        # Verify no learning materials were generated
        self.assertEqual(learning_materials, [])
        self.storage.search_similar.assert_called_once()


if __name__ == '__main__':
    unittest.main()