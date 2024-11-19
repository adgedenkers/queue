# shoe = "DV by dolche vida size 11 Elaine in Rose Gold"
# shoe="Adidas Samba Classic Black White Mens Size 12"
# shoe = "Birkenstock sandals bostonian size 9 womens"
shoe = "New Balance style WXNRGBK size 11 medium upc 739655347597"


import json
import openai
import os

from openai import OpenAI, OpenAIError, AsyncOpenAI

GPT_MODEL = "gpt-4"
openai.api_key = os.environ['OPENAI_API_KEY']

client = openai.OpenAI()

def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

shoe = chat_completion_request([
    {"role": "system", "content": "You are a precise data extractor. Respond only with the requested JSON format."},
    {"role": "user", "content": f"{shoe}"},])

def chat_enhance_shoe_data(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
shoe_data = json.loads(shoe.choices[0].message.content)

# get shoe prompt
with open('shoe_prompt.txt', 'r') as file: shoe_prompt = file.read()

enhanced_shoe_data = chat_completion_request([
    {"role": "system", "content": f"{shoe_prompt}"},
    {"role": "user", "content": f"{shoe_data}"}]
)

print(enhanced_shoe_data.choices[0].message.content)
