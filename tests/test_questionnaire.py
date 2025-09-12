#!/usr/bin/env python3
"""
Standalone test for the QuestionnaireGenerator class.
This test doesn't require all the project dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_questionnaire_generator():
    """Test that we can import and instantiate the QuestionnaireGenerator."""
    try:
        # Try to import the QuestionnaireGenerator
        from rag.questionnaire_generator import QuestionnaireGenerator
        
        # Create a mock OpenRouterClient
        class MockOpenRouterClient:
            def chat_completion(self, model, messages, max_tokens, temperature):
                return {
                    "choices": [{
                        "message": {
                            "content": """[
                                {
                                    "question": "What is the main topic of this content?",
                                    "answer": "The content discusses educational material."
                                },
                                {
                                    "question": "What are the key concepts presented?",
                                    "answer": "Key concepts include learning and understanding."
                                }
                            ]"""
                        }
                    }]
                }
        
        # Instantiate the QuestionnaireGenerator
        mock_client = MockOpenRouterClient()
        generator = QuestionnaireGenerator(mock_client)
        
        print("✓ QuestionnaireGenerator imported and instantiated successfully")
        print(f"✓ Default LLM model: {generator.llm_model}")
        
        # Test generating a questionnaire
        sample_content = {
            "text_content": "This is sample educational content about science topics.",
            "source_file": "sample.pdf",
            "page_id": "1",
            "content_type": "text"
        }
        
        # This would normally call the LLM, but we're using a mock
        qa_pairs = generator.generate_questionnaire_for_content(sample_content)
        
        print(f"✓ Generated {len(qa_pairs)} QA pairs")
        if qa_pairs:
            print(f"✓ First question: {qa_pairs[0]['question']}")
            print(f"✓ First answer: {qa_pairs[0]['answer']}")
            print(f"✓ Source file: {qa_pairs[0]['source_file']}")
            print(f"✓ Page ID: {qa_pairs[0]['page_id']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing QuestionnaireGenerator...")
    success = test_questionnaire_generator()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)