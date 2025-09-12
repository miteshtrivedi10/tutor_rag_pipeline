#!/usr/bin/env python3
"""
Test to demonstrate the exact output format matching the requirements.
"""

def demonstrate_output_format():
    """Demonstrate the exact output format."""
    
    # Example of the exact format you requested
    example_qa_pairs = [
        {
            "question": "what is the high level process of circulation carried out in human heart?",
            "answer": "The continuous movement of important nutrients, oxygen and carbon dioxide between the hearts and test of body by blood through the blood vessels is called circulation.",
            "source_file": "Modified.pdf",
            "page_id": "1" 
        },
        {
            "question": "what are the most important veins of the human body? Also describe main functions for each.",
            "answer": "Important veins of human body are superior vena cava, which brings blood from upper body to the heart, and inferior vena cava, which brings blood from lower body to the heart.",
            "source_file": "Modified.pdf",
            "page_id": "1" 
        }
    ]
    
    print("Exact format as requested:")
    print()
    
    for i, qa in enumerate(example_qa_pairs, 1):
        print("{")
        print(f'    "question": "{qa["question"]}",')
        print(f'    "answer": "{qa["answer"]}",')
        print(f'    "source_file": "{qa["source_file"]}",')
        print(f'    "page_id": "{qa["page_id"]}" ')
        print("}")
        print()
    
    print("This matches exactly with your requirements:")
    print("- Specific, direct questions without referencing the source")
    print("- No phrases like 'according to the content' or 'based on the diagram'")
    print("- Questions test actual knowledge of the subject matter")
    print("- Clean JSON format with all required fields")

if __name__ == "__main__":
    demonstrate_output_format()