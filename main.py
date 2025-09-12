import json
import logging
import sys
import base64
from pathlib import Path
from src.config.settings import settings
from src.rag.processor import RAGProcessor
from src.rag.openrouter import OpenRouterClient
from src.rag.nomic_embedding import NomicEmbeddingGenerator
from src.rag.storage import MilvusStorage
from src.utils.exceptions import RAGAnythingError

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the RAG-Anything Phase 3 implementation."""
    print("RAG-Anything Phase 3 - Embedding Generation and Storage Pipeline")

    # Initialize OpenRouter client
    openrouter_client = OpenRouterClient()

    def vision_model_func(content_item, context=None):
        """Real vision model function using Sonoma-Dusk-Alpha via OpenRouter."""


        # Extract image data
        image_bytes = content_item.get("data")
        if image_bytes is None:
            logger.warning("No image data found in content_item")
            return {"description": "No image data available", "scene_type": "unknown"}

        # Ensure image_bytes is bytes
        if not isinstance(image_bytes, bytes):
            if hasattr(image_bytes, "tobytes"):
                # If it's a numpy array, convert to bytes
                image_bytes = image_bytes.tobytes()
            else:
                try:
                    image_bytes = base64.b64decode(image_bytes)
                except Exception:
                    logger.error(
                        f"Could not convert image data to bytes: {type(image_bytes)}"
                    )
                    return {
                        "description": "Could not process image data",
                        "scene_type": "unknown",
                        "objects_detected": [],
                        "error": "Invalid image data format",
                    }

        # Base64 encode image
        try:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image data: {e}")
            return {
                "description": "Failed to encode image data",
                "scene_type": "unknown",
                "objects_detected": [],
                "error": str(e),
            }

        image_mime = content_item.get("mime_type", "image/png")

        # Build context from surrounding content
        context_text = ""
        if context:
            context_texts = [
                item.get("text", "") for item in context if item.get("text")
            ]
            context_text = " ".join(context_texts[:3])[:1000]  # Limit context

        # Prompt for educational image analysis
        prompt = f"""
        Analyze this educational image from a textbook. Provide a detailed description suitable for RAG processing.
        Context from surrounding content: {context_text}
        
        Focus on:
        - Main subject and educational concept
        - Visual elements (diagrams, charts, photos)
        - Text content within the image if readable
        - Scene type (diagram, photo, chart, illustration)
        - Key objects and their relationships
        
        Return structured JSON response only:
        {{
            "description": "Detailed natural language description (2-3 sentences)",
            "scene_type": "diagram|photo|chart|illustration|other",
            "objects_detected": ["list", "of", "key", "objects"],
            "colors_present": ["dominant", "colors"],
            "text_elements": ["any", "readable", "text", "from", "image"],
            "educational_concept": "main learning objective",
            "complexity_level": "simple|medium|advanced"
        }}
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert educational content analyst. Analyze textbook images accurately for RAG systems.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime};base64,{image_base64}"
                        },
                    },
                ],
            },
        ]

        try:
            response = openrouter_client.chat_completion(
                model="openrouter/sonoma-dusk-alpha",
                messages=messages,
                max_tokens=500,
                temperature=0.3,  # Low temperature for factual analysis
            )

            # Parse JSON from response
            content = response["choices"][0]["message"]["content"]


            analysis = json.loads(content)

            logger.info(
                f"Generated vision analysis for image from {content_item.get('source', 'unknown')}"
            )
            return analysis

        except Exception as e:
            logger.error(f"Vision model error: {e}")
            return {
                "description": f"Image analysis failed: {str(e)}",
                "scene_type": "unknown",
                "objects_detected": [],
                "error": str(e),
            }

    def llm_model_func(content_item, context=None):
        """Real LLM model function using Sonoma-Dusk-Alpha via OpenRouter."""
        content_type = content_item.get("type", "text")
        text_content = content_item.get("text", "") or content_item.get(
            "enhanced_text", ""
        )

        # Build context
        context_text = ""
        if context:
            context_items = [
                item.get("text", "") or item.get("enhanced_text", "")
                for item in context
                if item.get("text") or item.get("enhanced_text")
            ]
            context_text = " ".join(context_items[:2])[:1500]  # Limit context length

        # Prompt for content analysis based on type
        if content_type == "table":
            prompt = f"""
            Analyze this table from educational content. Extract key information and provide structured analysis.
            Table content: {text_content}
            Context: {context_text}
            
            Return JSON:
            {{
                "summary": "1-2 sentence summary of table content",
                "key_points": ["bullet", "points", "from", "table", "data"],
                "columns": ["list", "of", "column", "headers"],
                "rows_summary": "brief description of data patterns",
                "educational_value": "what students can learn from this table",
                "complexity": "simple|medium|advanced"
            }}
            """
        elif content_type == "equation":
            prompt = f"""
            Analyze this mathematical equation from educational content. Explain its significance and usage.
            Equation: {text_content}
            Context: {context_text}
            
            Return JSON:
            {{
                "summary": "What the equation represents",
                "components": ["variable", "meanings", "and", "relationships"],
                "application": "Educational context and usage",
                "difficulty_level": "basic|intermediate|advanced",
                "related_concepts": ["other", "equations", "or", "topics"],
                "solved_example": "Simple numerical example if applicable"
            }}
            """
        else:  # Generic text content
            prompt = f"""
            Analyze this educational text content for RAG processing.
            Content: {text_content}
            Context: {context_text}
            Content type: {content_type}
            
            Return JSON:
            {{
                "summary": "Concise summary (1-2 sentences)",
                "key_points": ["3-5", "main", "ideas"],
                "analysis": "Deeper understanding and connections",
                "educational_objectives": ["what", "students", "should", "learn"],
                "vocabulary_terms": ["key", "terms", "with", "definitions"],
                "complexity": "simple|medium|advanced"
            }}
            """

        messages = [
            {
                "role": "system",
                "content": f"You are an expert educational content analyst for {content_type}. Provide structured JSON analysis only.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = openrouter_client.chat_completion(
                model="openrouter/sonoma-dusk-alpha",
                messages=messages,
                max_tokens=800,
                temperature=0.2,  # Low for consistent analysis
            )

            content = response["choices"][0]["message"]["content"]


            analysis = json.loads(content)

            logger.debug(f"Generated LLM analysis for {content_type} content")
            return analysis

        except Exception as e:
            logger.error(f"LLM model error: {e}")
            return {
                "summary": f"Analysis failed: {str(e)}",
                "key_points": [],
                "error": str(e),
            }

    # Initialize embedding generator and storage
    embedding_generator = NomicEmbeddingGenerator()
    storage = MilvusStorage(
        uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN, use_mock_on_failure=False
    )

    # Initialize the RAG processor with model functions
    processor = RAGProcessor(
        storage=storage,
        vision_model_func=vision_model_func,
        llm_model_func=llm_model_func,
    )

    # Process test PDFs if available
    test_pdfs_dir = Path("test_pdfs")
    if test_pdfs_dir.exists() and test_pdfs_dir.is_dir():
        print(f"Processing documents in {test_pdfs_dir}")
        try:
            results = processor.process_directory(str(test_pdfs_dir))

            # Print summary
            total_files = len(results)
            total_content_items = sum(len(content) for content in results.values())

            print("Processing Summary:")
            print(f"- Processed {total_files} files")
            print(f"- Extracted {total_content_items} content items")

            # Print details for each file
            for file_path, content_list in results.items():
                print(f"- {file_path}: {len(content_list)} content items")

        except RAGAnythingError as e:
            logger.error(f"RAG-Anything error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            sys.exit(1)
    else:
        print(f"Test directory {test_pdfs_dir} not found")

    print("Phase 3 implementation completed successfully!")

    # Test search functionality
    # print("Testing search functionality...")
    # query = "What is the main concept discussed in the document?"
    # similar_content = processor.search_similar_content(query, top_k=3)
    # print(f"Found {len(similar_content)} similar content items for query: '{query}'")
    # for i, item in enumerate(similar_content):
    #     print(f"  {i+1}. {item.get('text_content', '')[:100]}...")


if __name__ == "__main__":
    main()
