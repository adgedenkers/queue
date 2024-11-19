import json
import os
from openai import OpenAI
import argparse
from typing import Dict, Any

def get_shoe_data(shoe_description: str, model: str = "gpt-4") -> Dict[str, Any]:
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    initial_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise data extractor. Respond only with the requested JSON format."},
            {"role": "user", "content": shoe_description}
        ]
    )
    shoe_data = json.loads(initial_response.choices[0].message.content)
    
    with open('../static/prompts/shoe_prompt.txt', 'r') as file:
        shoe_prompt = file.read()
        
    enhanced_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": shoe_prompt},
            {"role": "user", "content": json.dumps(shoe_data)}
        ]
    )
    
    return json.loads(enhanced_response.choices[0].message.content)

def main():
    parser = argparse.ArgumentParser(description='Process shoe description data')
    parser.add_argument('description', type=str, help='Shoe description to process')
    args = parser.parse_args()
    
    try:
        result = get_shoe_data(args.description)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error processing shoe data: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()