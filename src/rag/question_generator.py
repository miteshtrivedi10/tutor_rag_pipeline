from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from src.rag.storage import MilvusStorage
from src.rag.nomic_embedding import NomicEmbeddingGenerator

logger = logging.getLogger(__name__)


class BaseQuestionGenerator(ABC):
    """Abstract base class for question generators."""
    
    def __init__(self, embedding_generator: NomicEmbeddingGenerator, storage: MilvusStorage):
        """
        Initialize the question generator.
        
        Args:
            embedding_generator: Embedding generator for processing content
            storage: Storage system for retrieving content
        """
        self.embedding_generator = embedding_generator
        self.storage = storage
    
    @abstractmethod
    def generate_questions(self, content: Dict[str, Any], num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate questions based on the provided content.
        
        Args:
            content: Content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions with metadata
        """
        pass
    
    @abstractmethod
    def generate_answers(self, questions: List[Dict[str, Any]], content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate answers for the provided questions based on the content.
        
        Args:
            questions: List of questions to answer
            content: Content to use for generating answers
            
        Returns:
            List of questions with their answers
        """
        pass


class QAQuestionGenerator(BaseQuestionGenerator):
    """Question generator for creating Q&A pairs for student learning."""
    
    def __init__(self, embedding_generator: NomicEmbeddingGenerator, storage: MilvusStorage, 
                 llm_model_func=None):
        """
        Initialize the QA question generator.
        
        Args:
            embedding_generator: Embedding generator for processing content
            storage: Storage system for retrieving content
            llm_model_func: Function to call the LLM for content analysis
        """
        super().__init__(embedding_generator, storage)
        self.llm_model_func = llm_model_func or self._default_llm_model_func
    
    def generate_questions(self, content: Dict[str, Any], num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate Q&A pairs based on the provided content.
        
        Args:
            content: Content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of generated Q&A pairs with metadata
        """
        try:
            # Extract text content
            text_content = content.get("text_content") or content.get("text", "")
            if not text_content:
                logger.warning("No text content found for question generation")
                return []
            
            # Generate questions using LLM
            questions = self._generate_questions_with_llm(text_content, num_questions)
            
            # Add metadata
            for i, question in enumerate(questions):
                question["content_id"] = content.get("content_id", "")
                question["content_type"] = content.get("content_type", "unknown")
                question["source_file"] = content.get("source_file", "")
                question["chapter"] = content.get("chapter", "")
                question["page_number"] = content.get("page_number", 0)
                question["question_id"] = f"{content.get('content_id', 'content')}_{i}"
            
            return questions
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []
    
    def generate_answers(self, questions: List[Dict[str, Any]], content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate answers for the provided questions based on the content.
        
        Args:
            questions: List of questions to answer
            content: Content to use for generating answers
            
        Returns:
            List of questions with their answers
        """
        try:
            # Extract text content
            text_content = content.get("text_content") or content.get("text", "")
            if not text_content:
                logger.warning("No text content found for answer generation")
                return questions
            
            # Generate answers for each question
            questions_with_answers = []
            for question in questions:
                answer = self._generate_answer_with_llm(question["question"], text_content)
                question_with_answer = question.copy()
                question_with_answer["answer"] = answer
                questions_with_answers.append(question_with_answer)
            
            return questions_with_answers
        except Exception as e:
            logger.error(f"Error generating answers: {e}")
            return questions
    
    def _generate_questions_with_llm(self, text_content: str, num_questions: int) -> List[Dict[str, Any]]:
        """
        Generate questions using the LLM model.
        
        Args:
            text_content: Text content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        try:
            # Create prompt for question generation
            prompt = f"""
            Based on the following educational content, generate {num_questions} questions that would be 
            appropriate for student learning. The questions should test understanding of key concepts.
            
            Content:
            {text_content[:2000]}  # Limit content to 2000 characters
            
            Please provide questions in the following format:
            1. [Question 1]
            2. [Question 2]
            ...
            """
            
            # Call LLM model
            response = self.llm_model_func({
                "type": "text",
                "text": prompt
            })
            
            # Parse response into questions
            questions = []
            if isinstance(response, dict):
                # If response is a dict, try to extract questions
                generated_text = response.get("text", "")
                questions = self._parse_questions_from_text(generated_text)
            elif isinstance(response, str):
                # If response is a string, parse directly
                questions = self._parse_questions_from_text(response)
            else:
                # Default to simple questions
                questions = self._generate_default_questions(text_content, num_questions)
            
            return questions
        except Exception as e:
            logger.error(f"Error generating questions with LLM: {e}")
            # Fallback to default question generation
            return self._generate_default_questions(text_content, num_questions)
    
    def _generate_answer_with_llm(self, question: str, text_content: str) -> str:
        """
        Generate an answer for a question using the LLM model.
        
        Args:
            question: Question to answer
            text_content: Text content to use for generating the answer
            
        Returns:
            Generated answer
        """
        try:
            # Create prompt for answer generation
            prompt = f"""
            Based on the following educational content, answer the question:
            
            Question: {question}
            
            Content:
            {text_content[:2000]}  # Limit content to 2000 characters
            
            Answer:
            """
            
            # Call LLM model
            response = self.llm_model_func({
                "type": "text",
                "text": prompt
            })
            
            # Extract answer from response
            if isinstance(response, dict):
                return response.get("text", "Answer not available")
            elif isinstance(response, str):
                return response
            else:
                return "Answer not available"
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            return "Answer not available"
    
    def _parse_questions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse questions from text response.
        
        Args:
            text: Text containing questions
            
        Returns:
            List of parsed questions
        """
        questions = []
        lines = text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            # Check if line starts with a number followed by a period or parenthesis
            if line and (line[0].isdigit() and (". " in line[:5] or ") " in line[:5])):
                # Extract question text (everything after the number and separator)
                if ". " in line:
                    question_text = line.split(". ", 1)[1]
                else:
                    question_text = line.split(") ", 1)[1]
                
                if question_text:
                    questions.append({
                        "question": question_text,
                        "type": "qa",
                        "difficulty": "medium"
                    })
        
        return questions
    
    def _generate_default_questions(self, text_content: str, num_questions: int) -> List[Dict[str, Any]]:
        """
        Generate default questions when LLM fails.
        
        Args:
            text_content: Text content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of default questions
        """
        # Extract key terms from content (simple approach)
        words = text_content.split()
        key_terms = [word for word in words if len(word) > 5][:5]  # Get first 5 long words
        
        questions = []
        for i in range(min(num_questions, len(key_terms))):
            questions.append({
                "question": f"What is the meaning of '{key_terms[i]}' in this context?",
                "type": "qa",
                "difficulty": "medium"
            })
        
        # Add a general question if we haven't reached the desired number
        if len(questions) < num_questions:
            questions.append({
                "question": "What are the key concepts discussed in this content?",
                "type": "qa",
                "difficulty": "medium"
            })
        
        return questions
    
    def _default_llm_model_func(self, content_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default LLM model function for content analysis.
        
        Args:
            content_item: Content item to process
            
        Returns:
            Analysis results
        """
        # This is a placeholder implementation
        # In a real implementation, this would call an actual LLM
        return {
            "text": "Generated response from LLM",
            "summary": "Summary from LLM",
            "key_points": ["Point 1", "Point 2"]
        }


class QuizQuestionGenerator(BaseQuestionGenerator):
    """Question generator for creating quiz-style questions."""
    
    def __init__(self, embedding_generator: NomicEmbeddingGenerator, storage: MilvusStorage, 
                 llm_model_func=None):
        """
        Initialize the quiz question generator.
        
        Args:
            embedding_generator: Embedding generator for processing content
            storage: Storage system for retrieving content
            llm_model_func: Function to call the LLM for content analysis
        """
        super().__init__(embedding_generator, storage)
        self.llm_model_func = llm_model_func or self._default_llm_model_func
    
    def generate_questions(self, content: Dict[str, Any], num_questions: int = 5) -> List[Dict[str, Any]]:
        """
        Generate quiz-style questions based on the provided content.
        
        Args:
            content: Content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of generated quiz questions with metadata
        """
        try:
            # Extract text content
            text_content = content.get("text_content") or content.get("text", "")
            if not text_content:
                logger.warning("No text content found for quiz question generation")
                return []
            
            # Generate quiz questions using LLM
            questions = self._generate_quiz_questions_with_llm(text_content, num_questions)
            
            # Add metadata
            for i, question in enumerate(questions):
                question["content_id"] = content.get("content_id", "")
                question["content_type"] = content.get("content_type", "unknown")
                question["source_file"] = content.get("source_file", "")
                question["chapter"] = content.get("chapter", "")
                question["page_number"] = content.get("page_number", 0)
                question["question_id"] = f"{content.get('content_id', 'content')}_{i}"
            
            return questions
        except Exception as e:
            logger.error(f"Error generating quiz questions: {e}")
            return []
    
    def generate_answers(self, questions: List[Dict[str, Any]], content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate answers for the provided quiz questions based on the content.
        
        Args:
            questions: List of questions to answer
            content: Content to use for generating answers
            
        Returns:
            List of questions with their answers
        """
        # For quiz questions, answers are typically embedded in the question itself
        # This method is included for interface consistency
        return questions
    
    def _generate_quiz_questions_with_llm(self, text_content: str, num_questions: int) -> List[Dict[str, Any]]:
        """
        Generate quiz-style questions using the LLM model.
        
        Args:
            text_content: Text content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of generated quiz questions
        """
        try:
            # Create prompt for quiz question generation
            prompt = f"""
            Based on the following educational content, generate {num_questions} quiz-style questions.
            Include multiple choice questions with 4 options each. One option should be correct.
            
            Content:
            {text_content[:2000]}  # Limit content to 2000 characters
            
            Please provide questions in the following format:
            1. [Question 1]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct: [Correct Option Letter]
            
            2. [Question 2]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct: [Correct Option Letter]
            ...
            """
            
            # Call LLM model
            response = self.llm_model_func({
                "type": "text",
                "text": prompt
            })
            
            # Parse response into quiz questions
            questions = []
            if isinstance(response, dict):
                # If response is a dict, try to extract questions
                generated_text = response.get("text", "")
                questions = self._parse_quiz_questions_from_text(generated_text)
            elif isinstance(response, str):
                # If response is a string, parse directly
                questions = self._parse_quiz_questions_from_text(response)
            else:
                # Default to simple quiz questions
                questions = self._generate_default_quiz_questions(text_content, num_questions)
            
            return questions
        except Exception as e:
            logger.error(f"Error generating quiz questions with LLM: {e}")
            # Fallback to default quiz question generation
            return self._generate_default_quiz_questions(text_content, num_questions)
    
    def _parse_quiz_questions_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse quiz questions from text response.
        
        Args:
            text: Text containing quiz questions
            
        Returns:
            List of parsed quiz questions
        """
        questions = []
        lines = text.strip().split("\n")
        
        current_question = None
        options = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number followed by a period (new question)
            if line[0].isdigit() and ". " in line[:5]:
                # If we have a previous question, save it
                if current_question:
                    questions.append({
                        "question": current_question,
                        "options": options.copy(),
                        "type": "quiz",
                        "difficulty": "medium"
                    })
                
                # Start new question
                current_question = line.split(". ", 1)[1]
                options = []
            # Check if line is an option (A), B), C), D))
            elif line.startswith(("A) ", "B) ", "C) ", "D) ")):
                options.append(line[3:])  # Remove the letter and parenthesis
            # Check if line indicates the correct answer
            elif line.startswith("Correct: "):
                # We could store the correct answer, but for now we'll just continue
                pass
        
        # Don't forget the last question
        if current_question:
            questions.append({
                "question": current_question,
                "options": options,
                "type": "quiz",
                "difficulty": "medium"
            })
        
        return questions
    
    def _generate_default_quiz_questions(self, text_content: str, num_questions: int) -> List[Dict[str, Any]]:
        """
        Generate default quiz questions when LLM fails.
        
        Args:
            text_content: Text content to generate questions from
            num_questions: Number of questions to generate
            
        Returns:
            List of default quiz questions
        """
        # Extract key terms from content (simple approach)
        words = text_content.split()
        key_terms = [word for word in words if len(word) > 5][:num_questions]  # Get required number of long words
        
        questions = []
        for i, term in enumerate(key_terms):
            questions.append({
                "question": f"What is the meaning of '{term}'?",
                "options": [
                    f"A definition of {term}",
                    f"An opposite of {term}",
                    f"A synonym of {term}",
                    f"Unrelated to {term}"
                ],
                "type": "quiz",
                "difficulty": "medium"
            })
        
        return questions
    
    def _default_llm_model_func(self, content_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default LLM model function for content analysis.
        
        Args:
            content_item: Content item to process
            
        Returns:
            Analysis results
        """
        # This is a placeholder implementation
        # In a real implementation, this would call an actual LLM
        return {
            "text": "Generated response from LLM",
            "summary": "Summary from LLM",
            "key_points": ["Point 1", "Point 2"]
        }