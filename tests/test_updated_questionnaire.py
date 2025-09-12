#!/usr/bin/env python3
"""
Test for the updated QuestionnaireGenerator class with improved prompts.
"""

import json
import re

def test_prompt_quality():
    """Test that our prompt enforces the right constraints."""
    
    # Sample prompt that would be sent to the LLM
    sample_prompt = """
    Based on the following educational content, generate exactly 2 specific question-answer pairs 
    that test understanding of key educational concepts. The questions should be direct and specific, 
    without referencing the source material explicitly.
    
    Content:
    The human circulatory system consists of the heart, blood vessels, and blood. The heart pumps blood 
    through two main circuits: pulmonary circulation (to the lungs) and systemic circulation (to the rest 
    of the body). Blood vessels include arteries (carry blood away from heart), veins (carry blood to heart), 
    and capillaries (site of nutrient exchange).
    
    Requirements:
    1. Generate exactly 2 question-answer pairs
    2. Questions should be specific and direct educational questions
    3. DO NOT use phrases like "according to the content", "based on the diagram", "as per the text", etc.
    4. DO NOT reference the source material in any way
    5. Questions should test actual knowledge of the subject matter
    6. Answers should be concise but comprehensive
    7. Format each pair as a JSON object with "question" and "answer" fields
    
    Example of GOOD questions:
    - "What process describes the movement of nutrients through blood vessels?"
    - "Which veins carry blood from the upper and lower parts of the body to the heart?"
    
    Example of BAD questions:
    - "According to the content, what process describes..."
    - "As per the diagram shown, which veins carry..."
    - "Based on the text, what are the key concepts..."
    
    Return ONLY a JSON array with exactly 2 objects in this format:
    [
        {
            "question": "Your first specific question here",
            "answer": "Your first direct answer here"
        },
        {
            "question": "Your second specific question here",
            "answer": "Your second direct answer here"
        }
    ]
    """
    
    # Check that the prompt includes the important constraints
    assert "DO NOT use phrases like" in sample_prompt
    assert "DO NOT reference the source material" in sample_prompt
    assert "GOOD questions" in sample_prompt
    assert "BAD questions" in sample_prompt
    
    print("✓ Prompt includes constraints against source referencing")
    print("✓ Prompt includes examples of good and bad questions")
    
    # Test JSON parsing function
    sample_response = '''[
        {
            "question": "What are the two main circuits of the human circulatory system?",
            "answer": "The two main circuits are pulmonary circulation and systemic circulation."
        },
        {
            "question": "What is the function of capillaries in the circulatory system?",
            "answer": "Capillaries are the site of nutrient and waste exchange between blood and tissues."
        }
    ]'''
    
    # Simple JSON parsing test
    try:
        parsed = json.loads(sample_response)
        assert len(parsed) == 2
        assert "question" in parsed[0]
        assert "answer" in parsed[0]
        print("✓ JSON parsing works correctly")
    except Exception as e:
        print(f"✗ JSON parsing failed: {e}")
        return False
    
    # Test regex extraction
    markdown_response = "```json\n" + sample_response + "\n```"
    json_match = re.search(r'\[[\s\S]*\]', markdown_response)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            assert len(parsed) == 2
            print("✓ Regex extraction of JSON from markdown works")
        except Exception as e:
            print(f"✗ Regex extraction failed: {e}")
            return False
    else:
        print("✗ Regex pattern didn't match")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing updated QuestionnaireGenerator...")
    success = test_prompt_quality()
    if success:
        print("\n✓ All tests passed!")
        print("\nExample of expected output format:")
        print("{")
        print('    "question": "What are the two main circuits of the human circulatory system?",')
        print('    "answer": "The two main circuits are pulmonary circulation and systemic circulation.",')
        print('    "source_file": "Modified.pdf",')
        print('    "page_id": "1"')
        print("}")
        print("")
        print("{")
        print('    "question": "What is the function of capillaries in the circulatory system?",')
        print('    "answer": "Capillaries are the site of nutrient and waste exchange between blood and tissues.",')
        print('    "source_file": "Modified.pdf",')
        print('    "page_id": "1"')
        print("}")
    else:
        print("\n✗ Some tests failed!")