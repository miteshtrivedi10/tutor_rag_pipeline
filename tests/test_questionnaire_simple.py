import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestQuestionnaireGenerator(unittest.TestCase):
    """Simple test cases for the QuestionnaireGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the OpenRouterClient
        self.mock_openrouter_client = Mock()
        
        # Import here to avoid dependency issues
        from rag.questionnaire_generator import QuestionnaireGenerator
        self.QuestionnaireGenerator = QuestionnaireGenerator
    
    def test_import(self):
        """Test that we can import the QuestionnaireGenerator."""
        self.assertTrue(hasattr(self, 'QuestionnaireGenerator'))
    
    def test_class_exists(self):
        """Test that QuestionnaireGenerator class exists."""
        # This test just verifies the class can be imported
        generator = self.QuestionnaireGenerator(self.mock_openrouter_client)
        self.assertIsInstance(generator, self.QuestionnaireGenerator)


if __name__ == "__main__":
    unittest.main()