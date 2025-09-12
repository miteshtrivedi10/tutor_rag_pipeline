from typing import List, Dict, Any, Optional
import logging
from src.rag.question_generator import QAQuestionGenerator, QuizQuestionGenerator
from src.rag.description_generator import ContentDescriptionGenerator
from src.rag.storage import MilvusStorage
from src.rag.openrouter import OpenRouterEmbeddingGenerator

logger = logging.getLogger(__name__)


class QAManager:
    """Manager for handling Q&A generation and content description."""
    
    def __init__(self, embedding_generator: OpenRouterEmbeddingGenerator, storage: MilvusStorage,
                 llm_model_func=None):
        """
        Initialize the Q&A manager.
        
        Args:
            embedding_generator: Embedding generator for processing content
            storage: Storage system for retrieving content
            llm_model_func: Function to call the LLM for content analysis
        """
        self.embedding_generator = embedding_generator
        self.storage = storage
        self.llm_model_func = llm_model_func
        
        # Initialize generators
        self.qa_generator = QAQuestionGenerator(embedding_generator, storage, llm_model_func)
        self.quiz_generator = QuizQuestionGenerator(embedding_generator, storage, llm_model_func)
        self.description_generator = ContentDescriptionGenerator(embedding_generator, storage, llm_model_func)
    
    def generate_qa_pairs(self, content: Dict[str, Any], num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate Q&A pairs for the given content.
        
        Args:
            content: Content to generate Q&A pairs for
            num_questions: Number of Q&A pairs to generate
            
        Returns:
            List of Q&A pairs
        """
        try:
            # Generate questions
            questions = self.qa_generator.generate_questions(content, num_questions)
            
            if not questions:
                logger.warning("No questions generated for content")
                return []
            
            # Generate answers for the questions
            qa_pairs = self.qa_generator.generate_answers(questions, content)
            
            return qa_pairs
        except Exception as e:
            logger.error(f"Error generating Q&A pairs: {e}")
            return []
    
    def generate_quiz_questions(self, content: Dict[str, Any], num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate quiz-style questions for the given content.
        
        Args:
            content: Content to generate quiz questions for
            num_questions: Number of quiz questions to generate
            
        Returns:
            List of quiz questions
        """
        try:
            # Generate quiz questions
            quiz_questions = self.quiz_generator.generate_questions(content, num_questions)
            
            return quiz_questions
        except Exception as e:
            logger.error(f"Error generating quiz questions: {e}")
            return []
    
    def generate_content_description(self, content: Dict[str, Any]) -> str:
        """
        Generate a meaningful description for the given content.
        
        Args:
            content: Content to generate description for
            
        Returns:
            Generated description
        """
        try:
            description = self.description_generator.generate_description(content)
            return description
        except Exception as e:
            logger.error(f"Error generating content description: {e}")
            return "Description not available."
    
    def generate_learning_material(self, content: Dict[str, Any], num_qa_pairs: int = 5, 
                                 num_quiz_questions: int = 5) -> Dict[str, Any]:
        """
        Generate complete learning material including Q&A pairs, quiz questions, and description.
        
        Args:
            content: Content to generate learning material for
            num_qa_pairs: Number of Q&A pairs to generate
            num_quiz_questions: Number of quiz questions to generate
            
        Returns:
            Dictionary containing all generated learning material
        """
        try:
            # Generate Q&A pairs
            qa_pairs = self.generate_qa_pairs(content, num_qa_pairs)
            
            # Generate quiz questions
            quiz_questions = self.generate_quiz_questions(content, num_quiz_questions)
            
            # Generate content description
            description = self.generate_content_description(content)
            
            # Combine all material
            learning_material = {
                "content_id": content.get("content_id", ""),
                "content_type": content.get("content_type", "unknown"),
                "source_file": content.get("source_file", ""),
                "chapter": content.get("chapter", ""),
                "page_number": content.get("page_number", 0),
                "description": description,
                "qa_pairs": qa_pairs,
                "quiz_questions": quiz_questions
            }
            
            return learning_material
        except Exception as e:
            logger.error(f"Error generating learning material: {e}")
            return {
                "content_id": content.get("content_id", ""),
                "error": f"Failed to generate learning material: {str(e)}"
            }
    
    def generate_learning_material_for_similar_content(self, query: str, top_k: int = 5,
                                                     num_qa_pairs: int = 3, 
                                                     num_quiz_questions: int = 3) -> List[Dict[str, Any]]:
        """
        Generate learning material for content similar to the given query.
        
        Args:
            query: Query to find similar content for
            top_k: Number of similar content items to process
            num_qa_pairs: Number of Q&A pairs to generate per content item
            num_quiz_questions: Number of quiz questions to generate per content item
            
        Returns:
            List of learning materials for similar content
        """
        try:
            # Search for similar content
            similar_content = self.storage.search_similar(
                self.embedding_generator.generate_embedding(query, "text"),
                top_k=top_k
            )
            
            if not similar_content:
                logger.warning("No similar content found for query")
                return []
            
            learning_materials = []
            for content in similar_content:
                # Generate learning material for each content item
                learning_material = self.generate_learning_material(
                    content, num_qa_pairs, num_quiz_questions
                )
                learning_materials.append(learning_material)
            
            return learning_materials
        except Exception as e:
            logger.error(f"Error generating learning material for similar content: {e}")
            return []