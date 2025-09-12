import unittest
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the FastAPI app to avoid startup issues
with patch('src.api.main.FastAPI'):
    from src.api.main import initialize_components, rag_processor, qa_manager


class TestAPI(unittest.TestCase):
    """Test cases for API components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize components with mocks
        with patch('src.api.main.RAGProcessor') as mock_rag_processor, \
             patch('src.api.main.QAManager') as mock_qa_manager, \
             patch('src.api.main.OpenRouterEmbeddingGenerator'), \
             patch('src.api.main.MilvusStorage'):
            
            # Create mock instances
            self.mock_rag_instance = Mock()
            self.mock_qa_instance = Mock()
            
            # Configure mock return values
            mock_rag_processor.return_value = self.mock_rag_instance
            mock_qa_manager.return_value = self.mock_qa_instance
            
            # Initialize components
            initialize_components()
    
    def test_initialize_components_success(self):
        """Test successful component initialization."""
        # This test is mainly to ensure the initialization function runs without errors
        # The actual mocking is done in setUp
        self.assertTrue(True)  # If we get here, initialization succeeded
    
    @patch('src.api.main.rag_processor')
    def test_process_document_endpoint(self, mock_rag):
        """Test process document endpoint."""
        from src.api.main import app
        from fastapi.testclient import TestClient
        
        # Mock the RAG processor response
        mock_rag.process_document.return_value = [
            {"type": "text", "text": "Sample content"}
        ]
        
        # Create test client
        client = TestClient(app)
        
        # Test the endpoint
        response = client.post("/process/document?file_path=test.pdf")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["file_path"], "test.pdf")
        self.assertEqual(json_response["content_items_processed"], 1)
        self.assertEqual(json_response["status"], "success")
    
    @patch('src.api.main.rag_processor')
    def test_search_similar_content_endpoint(self, mock_rag):
        """Test search similar content endpoint."""
        from src.api.main import app
        from fastapi.testclient import TestClient
        
        # Mock the RAG processor response
        mock_rag.search_similar_content.return_value = [
            {"content_id": "test_1", "text_content": "Similar content"}
        ]
        
        # Create test client
        client = TestClient(app)
        
        # Test the endpoint
        response = client.get("/search?query=test%20query&top_k=5")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["query"], "test query")
        self.assertEqual(json_response["count"], 1)
        self.assertEqual(json_response["status"], "success")
    
    @patch('src.api.main.qa_manager')
    def test_generate_qa_pairs_endpoint(self, mock_qa):
        """Test generate Q&A pairs endpoint."""
        from src.api.main import app
        from fastapi.testclient import TestClient
        
        # Mock the QA manager response
        mock_qa.generate_learning_material_for_similar_content.return_value = [
            {
                "qa_pairs": [
                    {"question": "What is this?", "answer": "This is an answer."}
                ]
            }
        ]
        
        # Create test client
        client = TestClient(app)
        
        # Test the endpoint
        response = client.get("/qa/generate?query=test%20query&num_questions=5")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["query"], "test query")
        self.assertEqual(json_response["count"], 1)
        self.assertEqual(json_response["status"], "success")
    
    @patch('src.api.main.qa_manager')
    def test_generate_quiz_questions_endpoint(self, mock_qa):
        """Test generate quiz questions endpoint."""
        from src.api.main import app
        from fastapi.testclient import TestClient
        
        # Mock the QA manager response
        mock_qa.generate_learning_material_for_similar_content.return_value = [
            {
                "quiz_questions": [
                    {
                        "question": "What is this?",
                        "options": ["A", "B", "C", "D"]
                    }
                ]
            }
        ]
        
        # Create test client
        client = TestClient(app)
        
        # Test the endpoint
        response = client.get("/quiz/generate?query=test%20query&num_questions=5")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["query"], "test query")
        self.assertEqual(json_response["count"], 1)
        self.assertEqual(json_response["status"], "success")
    
    @patch('src.api.main.qa_manager')
    def test_generate_content_description_endpoint(self, mock_qa):
        """Test generate content description endpoint."""
        from src.api.main import app
        from fastapi.testclient import TestClient
        
        # Mock the QA manager response
        mock_qa.generate_learning_material_for_similar_content.return_value = [
            {
                "description": "This is a meaningful description."
            }
        ]
        
        # Create test client
        client = TestClient(app)
        
        # Test the endpoint
        response = client.get("/description/generate?query=test%20query")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["query"], "test query")
        self.assertEqual(json_response["description"], "This is a meaningful description.")
        self.assertEqual(json_response["status"], "success")
    
    @patch('src.api.main.performance_monitor')
    def test_performance_metrics_endpoint(self, mock_monitor):
        """Test performance metrics endpoint."""
        from src.api.main import app
        from fastapi.testclient import TestClient
        
        # Mock the performance monitor response
        mock_metric = Mock()
        mock_metric.operation_name = "test_operation"
        mock_metric.execution_time = 0.5
        mock_metric.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
        mock_metric.success = True
        mock_metric.error_message = None
        mock_metric.input_size = 100
        mock_metric.output_size = 200
        
        mock_monitor.get_recent_metrics.return_value = [mock_metric]
        
        # Create test client
        client = TestClient(app)
        
        # Test the endpoint
        response = client.get("/performance/metrics?limit=10")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["count"], 1)
        self.assertEqual(json_response["status"], "success")
    
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        from src.api.main import app
        from fastapi.testclient import TestClient
        
        # Create test client
        client = TestClient(app)
        
        # Test the endpoint
        response = client.get("/health")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertEqual(json_response["status"], "healthy")


if __name__ == '__main__':
    unittest.main()