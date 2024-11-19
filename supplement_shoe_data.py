import json
import os
from openai import OpenAI
from typing import Dict, Any, Optional

def get_shoe_data(shoe_description: str, model: str = "gpt-4") -> Dict[str, Any]:
    """
    Process a shoe description to extract structured data using OpenAI's API.
    
    Args:
        shoe_description: Raw text description of the shoe
        model: OpenAI model to use (default: gpt-4)
        
    Returns:
        Dict containing processed shoe data
    """
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    # Initial data extraction
    initial_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise data extractor. Respond only with the requested JSON format."},
            {"role": "user", "content": shoe_description}
        ]
    )
    shoe_data = json.loads(initial_response.choices[0].message.content)
    
    # Enhanced data extraction using shoe prompt
    with open('static/prompts/shoe_prompt.txt', 'r') as file:
        shoe_prompt = file.read()
        
    enhanced_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": shoe_prompt},
            {"role": "user", "content": json.dumps(shoe_data)}
        ]
    )
    
    return json.loads(enhanced_response.choices[0].message.content)

# Example usage
if __name__ == "__main__":
    shoe_description = "New Balance style WXNRGBK size 11 medium upc 739655347597"
    try:
        result = get_shoe_data(shoe_description)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing shoe data: {str(e)}")