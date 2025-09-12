from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any, Optional
import base64
import json
import logging
import os
import uvicorn
from src.rag.processor import RAGProcessor
from src.rag.qa_manager import QAManager
from src.rag.openrouter import OpenRouterClient
from src.rag.nomic_embedding import NomicEmbeddingGenerator
from src.rag.storage import MilvusStorage
from src.rag.performance_monitor import get_global_monitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG-Anything API",
    description="API for the RAG-Anything project - Retrieval-Augmented Generation for educational content",
    version="1.0.0",
)

# Global variables for components
rag_processor: Optional[RAGProcessor] = None
qa_manager: Optional[QAManager] = None
performance_monitor = get_global_monitor()


def initialize_components():
    """Initialize the RAG components."""
    global rag_processor, qa_manager

    try:
        # Initialize embedding generator and storage
        embedding_generator = NomicEmbeddingGenerator(
            model_name="nomic-embed-text:latest",
            min_dimensions=768,
            vision_model_func=vision_model_func,  # Pass vision func for image preprocessing
        )
        storage = MilvusStorage()

        # Initialize OpenRouter client
        openrouter_client = OpenRouterClient()

        def vision_model_func(content_item, context=None):
            """Real vision model function using Sonoma-Dusk-Alpha via OpenRouter."""

            # Extract image data
            image_bytes = content_item.get("data")
            if not image_bytes:
                logger.warning("No image data found in content_item")
                return {
                    "description": "No image data available",
                    "scene_type": "unknown",
                }

            if not isinstance(image_bytes, bytes):
                image_bytes = base64.b64decode(image_bytes)

            # Base64 encode image
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
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
                context_text = " ".join(context_items[:2])[
                    :1500
                ]  # Limit context length

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

        # Initialize RAG processor
        rag_processor = RAGProcessor(
            vision_model_func=vision_model_func,
            llm_model_func=llm_model_func,
            embedding_generator=embedding_generator,
            storage=storage,
        )

        # Initialize QA manager
        qa_manager = QAManager(
            embedding_generator=embedding_generator,
            storage=storage,
            llm_model_func=llm_model_func,
        )

        logger.info("RAG components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG components: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    initialize_components()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG-Anything API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "components": ["rag_processor", "qa_manager"]}


@app.post("/process/document")
async def process_document(file_path: str):
    """
    Process a single document file.

    Args:
        file_path: Path to the document file to process

    Returns:
        Processing results
    """
    if not rag_processor:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")

    try:
        content_list = rag_processor.process_document(file_path)
        
        # Generate questionnaires for the processed content
        questionnaire_data = []
        if hasattr(rag_processor, 'questionnaire_generator') and content_list:
            for content_item in content_list:
                qa_pairs = rag_processor.questionnaire_generator.generate_questionnaire_for_content(content_item)
                questionnaire_data.extend(qa_pairs)
        
        return {
            "file_path": file_path,
            "content_items_processed": len(content_list),
            "questionnaires_generated": len(questionnaire_data),
            "questionnaire_data": questionnaire_data,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )


@app.post("/process/documents")
async def process_documents(file_paths: List[str]):
    """
    Process multiple document files.

    Args:
        file_paths: List of paths to document files to process

    Returns:
        Processing results
    """
    if not rag_processor:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")

    try:
        results = rag_processor.process_documents(file_paths)
        total_files = len(results)
        total_content_items = sum(len(content) for content in results.values())
        
        # Generate questionnaires for all processed content
        questionnaire_data = []
        if hasattr(rag_processor, 'questionnaire_generator'):
            for file_path, content_list in results.items():
                for content_item in content_list:
                    qa_pairs = rag_processor.questionnaire_generator.generate_questionnaire_for_content(content_item)
                    questionnaire_data.extend(qa_pairs)
        
        return {
            "total_files_processed": total_files,
            "total_content_items_extracted": total_content_items,
            "questionnaires_generated": len(questionnaire_data),
            "questionnaire_data": questionnaire_data,
            "results": results,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing documents: {str(e)}"
        )


@app.post("/process/directory")
async def process_directory(directory_path: str):
    """
    Process all supported files in a directory.

    Args:
        directory_path: Path to the directory to process

    Returns:
        Processing results
    """
    if not rag_processor:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")

    try:
        results = rag_processor.process_directory(directory_path)
        total_files = len(results)
        total_content_items = sum(len(content) for content in results.values())
        
        # Generate questionnaires for all processed content
        questionnaire_data = []
        if hasattr(rag_processor, 'questionnaire_generator'):
            for file_path, content_list in results.items():
                for content_item in content_list:
                    qa_pairs = rag_processor.questionnaire_generator.generate_questionnaire_for_content(content_item)
                    questionnaire_data.extend(qa_pairs)
        
        return {
            "directory_path": directory_path,
            "total_files_processed": total_files,
            "total_content_items_extracted": total_content_items,
            "questionnaires_generated": len(questionnaire_data),
            "questionnaire_data": questionnaire_data,
            "results": results,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error processing directory {directory_path}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing directory: {str(e)}"
        )


@app.get("/search")
async def search_similar_content(
    query: str, top_k: int = Query(5, ge=1, le=100), filter_expr: Optional[str] = None
):
    """
    Search for similar content based on a text query.

    Args:
        query: Text query to search for similar content
        top_k: Number of similar items to return (1-100)
        filter_expr: Optional filter expression for the search

    Returns:
        List of similar content items
    """
    if not rag_processor:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")

    try:
        similar_items = rag_processor.search_similar_content(query, top_k, filter_expr)
        return {
            "query": query,
            "results": similar_items,
            "count": len(similar_items),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error searching for similar content: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error searching content: {str(e)}"
        )


@app.get("/search/reranked")
async def search_similar_content_with_reranking(
    query: str,
    initial_top_k: int = Query(20, ge=1, le=200),
    final_top_k: int = Query(5, ge=1, le=50),
    filter_expr: Optional[str] = None,
):
    """
    Search for similar content with reranking for better results.

    Args:
        query: Text query to search for similar content
        initial_top_k: Number of initial candidates to retrieve (1-200)
        final_top_k: Number of final results to return (1-50)
        filter_expr: Optional filter expression for the search

    Returns:
        List of similar content items, reranked
    """
    if not rag_processor:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")

    try:
        similar_items = rag_processor.search_similar_content_with_reranking(
            query, initial_top_k, final_top_k, filter_expr
        )
        return {
            "query": query,
            "results": similar_items,
            "count": len(similar_items),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error searching for similar content with reranking: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error searching content: {str(e)}"
        )


@app.get("/search/type")
async def search_similar_content_by_type(
    query: str, content_type: str, top_k: int = Query(5, ge=1, le=100)
):
    """
    Search for similar content filtered by content type.

    Args:
        query: Text query to search for similar content
        content_type: Content type to filter by
        top_k: Number of similar items to return (1-100)

    Returns:
        List of similar content items
    """
    if not rag_processor:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")

    try:
        similar_items = rag_processor.search_similar_content_by_type(
            query, content_type, top_k
        )
        return {
            "query": query,
            "content_type": content_type,
            "results": similar_items,
            "count": len(similar_items),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error searching for similar content by type: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error searching content: {str(e)}"
        )


@app.get("/search/chapter")
async def search_similar_content_by_chapter(
    query: str, chapter: str, top_k: int = Query(5, ge=1, le=100)
):
    """
    Search for similar content filtered by chapter.

    Args:
        query: Text query to search for similar content
        chapter: Chapter to filter by
        top_k: Number of similar items to return (1-100)

    Returns:
        List of similar content items
    """
    if not rag_processor:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")

    try:
        similar_items = rag_processor.search_similar_content_by_chapter(
            query, chapter, top_k
        )
        return {
            "query": query,
            "chapter": chapter,
            "results": similar_items,
            "count": len(similar_items),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error searching for similar content by chapter: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error searching content: {str(e)}"
        )


@app.get("/search/source")
async def search_similar_content_by_source(
    query: str, source_file: str, top_k: int = Query(5, ge=1, le=100)
):
    """
    Search for similar content filtered by source file.

    Args:
        query: Text query to search for similar content
        source_file: Source file to filter by
        top_k: Number of similar items to return (1-100)

    Returns:
        List of similar content items
    """
    if not rag_processor:
        raise HTTPException(status_code=500, detail="RAG processor not initialized")

    try:
        similar_items = rag_processor.search_similar_content_by_source(
            query, source_file, top_k
        )
        return {
            "query": query,
            "source_file": source_file,
            "results": similar_items,
            "count": len(similar_items),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error searching for similar content by source: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error searching content: {str(e)}"
        )


@app.get("/qa/generate")
async def generate_qa_pairs(query: str, num_questions: int = Query(5, ge=1, le=20)):
    """
    Generate Q&A pairs for content similar to the given query.

    Args:
        query: Query to find similar content for
        num_questions: Number of Q&A pairs to generate per content item (1-20)

    Returns:
        Generated Q&A pairs
    """
    if not qa_manager:
        raise HTTPException(status_code=500, detail="QA manager not initialized")

    try:
        learning_materials = qa_manager.generate_learning_material_for_similar_content(
            query, top_k=3, num_qa_pairs=num_questions, num_quiz_questions=0
        )

        # Extract only QA pairs from learning materials
        qa_pairs = []
        for material in learning_materials:
            if "qa_pairs" in material:
                qa_pairs.extend(material["qa_pairs"])

        return {
            "query": query,
            "qa_pairs": qa_pairs,
            "count": len(qa_pairs),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error generating Q&A pairs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating Q&A pairs: {str(e)}"
        )


@app.get("/quiz/generate")
async def generate_quiz_questions(
    query: str, num_questions: int = Query(5, ge=1, le=20)
):
    """
    Generate quiz questions for content similar to the given query.

    Args:
        query: Query to find similar content for
        num_questions: Number of quiz questions to generate per content item (1-20)

    Returns:
        Generated quiz questions
    """
    if not qa_manager:
        raise HTTPException(status_code=500, detail="QA manager not initialized")

    try:
        learning_materials = qa_manager.generate_learning_material_for_similar_content(
            query, top_k=3, num_qa_pairs=0, num_quiz_questions=num_questions
        )

        # Extract only quiz questions from learning materials
        quiz_questions = []
        for material in learning_materials:
            if "quiz_questions" in material:
                quiz_questions.extend(material["quiz_questions"])

        return {
            "query": query,
            "quiz_questions": quiz_questions,
            "count": len(quiz_questions),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error generating quiz questions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating quiz questions: {str(e)}"
        )


@app.get("/description/generate")
async def generate_content_description(query: str):
    """
    Generate a meaningful description for content similar to the given query.

    Args:
        query: Query to find similar content for

    Returns:
        Generated content description
    """
    if not qa_manager:
        raise HTTPException(status_code=500, detail="QA manager not initialized")

    try:
        learning_materials = qa_manager.generate_learning_material_for_similar_content(
            query, top_k=1, num_qa_pairs=0, num_quiz_questions=0
        )

        # Extract description from the first learning material
        description = ""
        if learning_materials and "description" in learning_materials[0]:
            description = learning_materials[0]["description"]

        return {"query": query, "description": description, "status": "success"}
    except Exception as e:
        logger.error(f"Error generating content description: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating content description: {str(e)}"
        )


@app.get("/learning-material/generate")
async def generate_learning_material(
    query: str,
    num_qa_pairs: int = Query(3, ge=1, le=10),
    num_quiz_questions: int = Query(3, ge=1, le=10),
):
    """
    Generate complete learning material for content similar to the given query.

    Args:
        query: Query to find similar content for
        num_qa_pairs: Number of Q&A pairs to generate per content item (1-10)
        num_quiz_questions: Number of quiz questions to generate per content item (1-10)

    Returns:
        Generated learning material
    """
    if not qa_manager:
        raise HTTPException(status_code=500, detail="QA manager not initialized")

    try:
        learning_materials = qa_manager.generate_learning_material_for_similar_content(
            query,
            top_k=3,
            num_qa_pairs=num_qa_pairs,
            num_quiz_questions=num_quiz_questions,
        )

        return {
            "query": query,
            "learning_materials": learning_materials,
            "count": len(learning_materials),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error generating learning material: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating learning material: {str(e)}"
        )


@app.get("/performance/metrics")
async def get_performance_metrics(limit: int = Query(10, ge=1, le=100)):
    """
    Get recent performance metrics.

    Args:
        limit: Number of recent metrics to return (1-100)

    Returns:
        Recent performance metrics
    """
    if not performance_monitor:
        raise HTTPException(
            status_code=500, detail="Performance monitor not initialized"
        )

    try:
        metrics = performance_monitor.get_recent_metrics(limit)
        return {
            "metrics": [
                {
                    "operation_name": m.operation_name,
                    "execution_time": m.execution_time,
                    "timestamp": m.timestamp.isoformat(),
                    "success": m.success,
                    "error_message": m.error_message,
                    "input_size": m.input_size,
                    "output_size": m.output_size,
                }
                for m in metrics
            ],
            "count": len(metrics),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting performance metrics: {str(e)}"
        )


@app.get("/performance/summary")
async def get_performance_summary():
    """
    Get performance summary for all operations.

    Returns:
        Performance summary
    """
    if not performance_monitor:
        raise HTTPException(
            status_code=500, detail="Performance monitor not initialized"
        )

    try:
        summary = performance_monitor.get_operation_summary()
        return {"summary": summary, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting performance summary: {str(e)}"
        )


@app.get("/performance/clear")
async def clear_performance_metrics():
    """
    Clear all performance metrics history.

    Returns:
        Success message
    """
    if not performance_monitor:
        raise HTTPException(
            status_code=500, detail="Performance monitor not initialized"
        )

    try:
        performance_monitor.clear_history()
        return {"message": "Performance metrics history cleared", "status": "success"}
    except Exception as e:
        logger.error(f"Error clearing performance metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error clearing performance metrics: {str(e)}"
        )


@app.get("/questionnaires/generate")
async def generate_questionnaires_for_content(query: str, top_k: int = Query(5, ge=1, le=20)):
    """
    Generate questionnaires for content similar to the given query.

    Args:
        query: Query to find similar content for
        top_k: Number of similar content items to process (1-20)

    Returns:
        Generated questionnaires
    """
    if not rag_processor or not hasattr(rag_processor, 'questionnaire_generator'):
        raise HTTPException(status_code=500, detail="Questionnaire generator not initialized")

    try:
        # Search for similar content
        similar_content = rag_processor.search_similar_content(query, top_k=top_k)
        
        # Generate questionnaires for each content item
        questionnaire_data = []
        for content_item in similar_content:
            qa_pairs = rag_processor.questionnaire_generator.generate_questionnaire_for_content(content_item)
            questionnaire_data.extend(qa_pairs)
        
        return {
            "query": query,
            "questionnaires": questionnaire_data,
            "count": len(questionnaire_data),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error generating questionnaires: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating questionnaires: {str(e)}"
        )


if __name__ == "__main__":


    uvicorn.run(app, host="0.0.0.0", port=8000)
