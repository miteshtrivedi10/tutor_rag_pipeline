import unittest
from unittest.mock import Mock, patch
from src.rag.questionnaire_generator import QuestionnaireGenerator


class TestQuestionnaireGenerator(unittest.TestCase):
    """Test cases for the QuestionnaireGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_openrouter_client = Mock()
        self.generator = QuestionnaireGenerator(self.mock_openrouter_client)
    
    def test_init(self):
        """Test initialization of QuestionnaireGenerator."""
        self.assertIsInstance(self.generator, QuestionnaireGenerator)
        self.assertEqual(self.generator.openrouter_client, self.mock_openrouter_client)
        self.assertEqual(self.generator.llm_model, "openrouter/sonoma-dusk-alpha")
    
    def test_generate_questionnaire_for_content_with_text(self):
        """Test generating questionnaire for content with text."""
        # Mock the OpenRouter client response
        mock_response = {
            "choices": [{
                "message": {
                    "content": """[
                        {
                            "question": "What is the main topic?",
                            "answer": "The main topic is educational content."
                        },
                        {
                            "question": "What are key concepts?",
                            "answer": "Key concepts include learning and understanding."
                        }
                    ]"""
                }
            }]
        }
        self.mock_openrouter_client.chat_completion.return_value = mock_response
        
        content_item = {
            "text_content": "This is sample educational content for testing.",
            "source_file": "test.pdf",
            "page_id": "1",
            "content_type": "text"
        }
        
        result = self.generator.generate_questionnaire_for_content(content_item)
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["question"], "What is the main topic?")
        self.assertEqual(result[0]["answer"], "The main topic is educational content.")
        self.assertEqual(result[0]["source_file"], "test.pdf")
        self.assertEqual(result[0]["page_id"], "1")
        
        # Verify the OpenRouter client was called with correct parameters
        self.mock_openrouter_client.chat_completion.assert_called_once()
    
    def test_generate_questionnaire_for_content_no_text(self):
        """Test generating questionnaire for content with no text."""
        content_item = {
            "source_file": "test.pdf",
            "page_id": "1"
        }
        
        result = self.generator.generate_questionnaire_for_content(content_item)
        
        # Should return empty list when no text content
        self.assertEqual(result, [])
    
    def test_generate_default_qa(self):
        """Test generating default QA pairs."""
        result = self.generator._generate_default_qa("test content", "test.pdf", "1")
        
        # Should return 2 default QA pairs
        self.assertEqual(len(result), 2)
        self.assertIn("main topic", result[0]["question"])
        self.assertEqual(result[0]["source_file"], "test.pdf")
        self.assertEqual(result[0]["page_id"], "1")


if __name__ == "__main__":
    unittest.main()