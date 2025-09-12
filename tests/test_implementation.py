#!/usr/bin/env python3
"""Test script for RAG-Anything Phase 1 implementation."""

import logging
from pathlib import Path
from src.rag.processor import RAGProcessor
from src.rag.openrouter import OpenRouterEmbeddingGenerator
from src.rag.storage import MilvusStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_file():
    """Test processing a single file."""
    test_pdfs_dir = Path("test_pdfs")
    if test_pdfs_dir.exists():
        pdf_files = list(test_pdfs_dir.glob("*.pdf"))
        if pdf_files:
            file_path = str(pdf_files[0])
            print(f"Testing single file processing: {file_path}")
            
            # Define model functions (placeholder implementations)
            def vision_model_func(content_item, context=None):
                """Placeholder vision model function."""
                return {
                    "description": "Detailed visual description from vision model",
                    "objects_detected": ["object1", "object2"],
                    "colors_present": ["color1", "color2"],
                    "scene_type": "educational_diagram"
                }
            
            def llm_model_func(content_item, context=None):
                """Placeholder LLM model function."""
                content_type = content_item.get("type", "content")
                return {
                    "summary": f"Summary of the {content_type} from LLM",
                    "key_points": ["Point 1", "Point 2"],
                    "analysis": "Detailed analysis from LLM"
                }
            
            # Initialize embedding generator and storage
            embedding_generator = OpenRouterEmbeddingGenerator()
            storage = MilvusStorage()
            
            processor = RAGProcessor(
                vision_model_func=vision_model_func,
                llm_model_func=llm_model_func,
                embedding_generator=embedding_generator,
                storage=storage
            )
            try:
                content_list = processor.process_document(file_path)
                print(f"Successfully processed {file_path}")
                print(f"Extracted {len(content_list)} content items")
                if content_list:
                    print("First content item:")
                    print(f"  Type: {content_list[0].get('type', 'N/A')}")
                    print(f"  Text: {content_list[0].get('text', 'N/A')[:100]}...")
                    # Print enhanced metadata if available
                    metadata = content_list[0].get('metadata', {})
                    if metadata.get('has_visual_analysis') or metadata.get('has_table_analysis') or metadata.get('has_equation_analysis') or metadata.get('has_content_analysis'):
                        print(f"  Enhanced metadata: {metadata}")
                
                # Test search functionality
                print("\nTesting search functionality...")
                query = "What is the main concept discussed in the document?"
                similar_content = processor.search_similar_content(query, top_k=3)
                print(f"Found {len(similar_content)} similar content items for query: '{query}'")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                return False
            return True
    return False

def test_directory():
    """Test processing a directory."""
    test_pdfs_dir = Path("test_pdfs")
    if test_pdfs_dir.exists():
        print(f"\nTesting directory processing: {test_pdfs_dir}/")
        
        # Define model functions (placeholder implementations)
        def vision_model_func(content_item, context=None):
            """Placeholder vision model function."""
            return {
                "description": "Detailed visual description from vision model",
                "objects_detected": ["object1", "object2"],
                "colors_present": ["color1", "color2"],
                "scene_type": "educational_diagram"
            }
        
        def llm_model_func(content_item, context=None):
            """Placeholder LLM model function."""
            content_type = content_item.get("type", "content")
            return {
                "summary": f"Summary of the {content_type} from LLM",
                "key_points": ["Point 1", "Point 2"],
                "analysis": "Detailed analysis from LLM"
            }
        
        # Initialize embedding generator and storage
        embedding_generator = OpenRouterEmbeddingGenerator()
        storage = MilvusStorage()
        
        processor = RAGProcessor(
            vision_model_func=vision_model_func,
            llm_model_func=llm_model_func,
            embedding_generator=embedding_generator,
            storage=storage
        )
        try:
            results = processor.process_directory(str(test_pdfs_dir))
            
            total_files = len(results)
            total_content_items = sum(len(content) for content in results.values())
            
            print(f"Processing Summary:")
            print(f"- Processed {total_files} files")
            print(f"- Extracted {total_content_items} content items")
            
            for file_path, content_list in results.items():
                print(f"- {file_path}: {len(content_list)} content items")
                # Print enhanced metadata for first content item if available
                if content_list:
                    metadata = content_list[0].get('metadata', {})
                    if metadata.get('has_visual_analysis') or metadata.get('has_table_analysis') or metadata.get('has_equation_analysis') or metadata.get('has_content_analysis'):
                        print(f"  Enhanced metadata: {metadata}")
            
            # Test search functionality
            print("\nTesting search functionality...")
            query = "What is the main concept discussed in the document?"
            similar_content = processor.search_similar_content(query, top_k=3)
            print(f"Found {len(similar_content)} similar content items for query: '{query}'")
                
            return True
        except Exception as e:
            print(f"Error processing directory {test_pdfs_dir}: {e}")
            return False
    return False

def main():
    """Main test function."""
    print("RAG-Anything Phase 1 Implementation Test")
    print("=" * 40)
    
    # Test single file processing
    success1 = test_single_file()
    
    # Test directory processing
    success2 = test_directory()
    
    if success1 and success2:
        print("\nAll tests passed!")
        return True
    else:
        print("\nSome tests failed!")
        return False

if __name__ == "__main__":
    main()
