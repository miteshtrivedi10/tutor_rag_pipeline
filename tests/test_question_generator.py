import unittest
from unittest.mock import Mock, patch
import numpy as np
from src.rag.question_generator import QAQuestionGenerator, QuizQuestionGenerator
from src.rag.storage import MilvusStorage
from src.rag.openrouter import OpenRouterEmbeddingGenerator


class TestQAQuestionGenerator(unittest.TestCase):
    """Test cases for QAQuestionGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_generator = Mock(spec=OpenRouterEmbeddingGenerator)
        self.storage = Mock(spec=MilvusStorage)
        self.llm_model_func = Mock()
        
        # Configure mock LLM response
        self.llm_model_func.return_value = {
            "text": "1. What is the main concept?\n2. How does it work?\n3. Why is it important?"
        }
        
        self.qa_generator = QAQuestionGenerator(
            self.embedding_generator, 
            self.storage, 
            self.llm_model_func
        )
    
    def test_generate_questions_success(self):
        """Test successful question generation."""
        content = {
            "text_content": "This is sample educational content about science concepts.",
            "content_id": "test_1",
            "content_type": "text",
            "source_file": "test.pdf",
            "chapter": "Chapter 1",
            "page_number": 1
        }
        
        questions = self.qa_generator.generate_questions(content, 3)
        
        # Verify questions were generated
        self.assertEqual(len(questions), 3)
        self.assertEqual(questions[0]["question"], "What is the main concept?")
        self.assertEqual(questions[1]["question"], "How does it work?")
        self.assertEqual(questions[2]["question"], "Why is it important?")
        
        # Verify metadata was added
        for question in questions:
            self.assertEqual(question["content_id"], "test_1")
            self.assertEqual(question["content_type"], "text")
            self.assertEqual(question["source_file"], "test.pdf")
            self.assertEqual(question["chapter"], "Chapter 1")
            self.assertEqual(question["page_number"], 1)
    
    def test_generate_questions_no_content(self):
        """Test question generation with no content."""
        content = {}
        questions = self.qa_generator.generate_questions(content, 3)
        self.assertEqual(questions, [])
    
    def test_generate_questions_llm_failure(self):
        """Test question generation when LLM fails."""
        self.llm_model_func.side_effect = Exception("LLM error")
        
        content = {
            "text_content": "This is sample educational content.",
            "content_id": "test_1"
        }
        
        questions = self.qa_generator.generate_questions(content, 3)
        
        # Should fall back to default questions
        self.assertEqual(len(questions), 3)
        self.assertTrue("meaning" in questions[0]["question"])
    
    def test_generate_answers_success(self):
        """Test successful answer generation."""
        questions = [
            {"question": "What is the main concept?"},
            {"question": "How does it work?"}
        ]
        
        content = {
            "text_content": "This is sample educational content about science concepts."
        }
        
        # Configure mock LLM response for answers
        self.llm_model_func.return_value = "This is the answer."
        
        questions_with_answers = self.qa_generator.generate_answers(questions, content)
        
        self.assertEqual(len(questions_with_answers), 2)
        self.assertEqual(questions_with_answers[0]["answer"], "This is the answer.")
        self.assertEqual(questions_with_answers[1]["answer"], "This is the answer.")
    
    def test_generate_answers_no_content(self):
        """Test answer generation with no content."""
        questions = [{"question": "What is this?"}]
        content = {}
        questions_with_answers = self.qa_generator.generate_answers(questions, content)
        self.assertEqual(questions_with_answers[0]["answer"], "Answer not available")


class TestQuizQuestionGenerator(unittest.TestCase):
    """Test cases for QuizQuestionGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_generator = Mock(spec=OpenRouterEmbeddingGenerator)
        self.storage = Mock(spec=MilvusStorage)
        self.llm_model_func = Mock()
        
        # Configure mock LLM response
        self.llm_model_func.return_value = {
            "text": "1. What is the main concept?\nA) Option A\nB) Option B\nC) Option C\nD) Option D\nCorrect: B"
        }
        
        self.quiz_generator = QuizQuestionGenerator(
            self.embedding_generator, 
            self.storage, 
            self.llm_model_func
        )
    
    def test_generate_quiz_questions_success(self):
        """Test successful quiz question generation."""
        content = {
            "text_content": "This is sample educational content about science concepts.",
            "content_id": "test_1",
            "content_type": "text",
            "source_file": "test.pdf",
            "chapter": "Chapter 1",
            "page_number": 1
        }
        
        questions = self.quiz_generator.generate_questions(content, 1)
        
        # Verify questions were generated
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["question"], "What is the main concept?")
        self.assertIn("options", questions[0])
        
        # Verify metadata was added
        self.assertEqual(questions[0]["content_id"], "test_1")
        self.assertEqual(questions[0]["content_type"], "text")
    
    def test_generate_quiz_questions_no_content(self):
        """Test quiz question generation with no content."""
        content = {}
        questions = self.quiz_generator.generate_questions(content, 3)
        self.assertEqual(questions, [])
    
    def test_generate_quiz_questions_llm_failure(self):
        """Test quiz question generation when LLM fails."""
        self.llm_model_func.side_effect = Exception("LLM error")
        
        content = {
            "text_content": "This is sample educational content.",
            "content_id": "test_1"
        }
        
        questions = self.quiz_generator.generate_questions(content, 3)
        
        # Should fall back to default quiz questions
        self.assertEqual(len(questions), 3)
        self.assertIn("options", questions[0])


if __name__ == '__main__':
    unittest.main()