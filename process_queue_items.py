import os
import json
import openai
import asyncio
from datetime import datetime, timezone
import aiohttp
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


# Configuration
API_BASE_URL="https://api.denkers.co"
USER_ID="1"
AUTH_TOKEN="6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b"


class ObjectType(str, Enum):
    SHOES_FOR_SALE = "shoes-for-sale"
    MEDICAL_VITALS = "medical-vitals"
    UNKNOWN = "unknown"

@dataclass
class ProcessedQueueItem:
    id: int
    raw_text: str
    options: Dict[str, Any]
    properties: Dict[str, Any]

class QueueProcessor:
    def __init__(self, api_base_url: str, user_id: int, auth_token: str, openai_api_key: str):
        """Initialize the processor with API credentials."""
        self.api_base_url = api_base_url
        self.user_id = user_id
        self.auth_token = auth_token
        self.headers = {
            "user-id": str(user_id),
            "x-token": auth_token,
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        openai.api_key = openai_api_key

    async def fetch_pending_items(self) -> List[Dict[str, Any]]:
        """Fetch queue items with status 'pending'."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.api_base_url}/queue/"
            params = {"status": "pending", "active": True}
            
            async with session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to fetch queue items: {await response.text()}")

    async def detect_object_type(self, text: str) -> ObjectType:
        """Use GPT to determine what type of data the text represents."""
        prompt = f"""
        Analyze this text and determine if it's describing:
        1. Shoes for sale (look for brand names, sizes, colors)
        2. Medical vitals (look for bp, pulse, patient info, heart rate, pulse ox, etc.)
        3. Everything else is a Log Entry
        
        Text: "{text}"
        
        Respond with exactly one of these:
        - shoes-for-sale
        - medical-vitals
        - log-entry
        """
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise data classifier."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content.strip().lower()
        return ObjectType(result)

    async def process_shoes_data(self, text: str) -> Dict[str, Any]:
        """Extract shoe details from text using GPT."""
        prompt = f"""
        Extract shoe details from this text: "{text}"
        
        Return a JSON object with these fields:
        - brand: The shoe brand name
        - model: The shoe model name
        - size: Numeric shoe size
        - color: Color description
        - gender: One of [mens, womens, unisex, youth, kids]
        - shoe_type: Type of shoe (sneaker, boot, sandal, etc.)
        - style: Style description
        
        Provide only the JSON object, no other text.
        """
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise data extractor."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)

    async def process_vitals_data(self, text: str) -> Dict[str, Any]:
        """Extract medical vitals from text using GPT."""
        prompt = f"""
        Extract vital signs from this text: "{text}"
        
        Return a JSON object with these fields:
        - patient_name: Name if provided, otherwise null
        - systolic_bp: Systolic blood pressure number
        - diastolic_bp: Diastolic blood pressure number
        - pulse: Heart rate/pulse number
        - timestamp: Current UTC timestamp
        
        Provide only the JSON object, no other text.
        """
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise data extractor."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)

    async def update_queue_item(self, item_id: int, options: Dict[str, Any], properties: Dict[str, Any]):
        """Update a queue item with processed data."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.api_base_url}/queue/{item_id}"
            data = {
                "options": options,
                "properties": properties,
                "status": "processed"
            }
            
            async with session.patch(url, headers=self.headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"Failed to update queue item: {await response.text()}")

    async def create_shoe_record(self, properties: Dict[str, Any]):
        """Create a new shoe record from processed properties."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.api_base_url}/shoes/"
            
            # Map properties to shoe record format
            shoe_data = {
                "brand": properties["brand"],
                "model": properties["model"],
                "size": float(properties["size"]),
                "color": properties["color"],
                "gender": properties["gender"],
                "shoe_type": properties["shoe_type"],
                "style": properties["style"],
                "condition": "Used - Excellent"  # Default value, adjust as needed
            }
            
            async with session.post(url, headers=self.headers, json=shoe_data) as response:
                if response.status != 201:
                    raise Exception(f"Failed to create shoe record: {await response.text()}")

    async def process_queue_items(self):
        """Main processing function."""
        items = await self.fetch_pending_items()
        
        for item in items:
            try:
                # Detect type of data
                object_type = await self.detect_object_type(item["raw_text"])
                
                options = {"object_type": object_type.value}
                properties = {}
                
                # Process based on type
                if object_type == ObjectType.SHOES_FOR_SALE:
                    properties = await self.process_shoes_data(item["raw_text"])
                    await self.create_shoe_record(properties)
                elif object_type == ObjectType.MEDICAL_VITALS:
                    properties = await self.process_vitals_data(item["raw_text"])
                
                # Update queue item with processed data
                await self.update_queue_item(item["id"], options, properties)
                
            except Exception as e:
                print(f"Error processing item {item['id']}: {str(e)}")
                continue

async def main():
    # Load configuration from environment variables
    api_base_url = os.getenv("API_BASE_URL", "https://api.denkers.co")
    user_id = int(os.getenv("USER_ID", "1"))
    auth_token = os.getenv("AUTH_TOKEN", "6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    processor = QueueProcessor(api_base_url, user_id, auth_token, openai_api_key)
    await processor.process_queue_items()

if __name__ == "__main__":
    asyncio.run(main())