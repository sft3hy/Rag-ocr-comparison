import os
import requests
from typing import List, Dict, Any
import json

SANCTUARY_API_KEY = os.environ.get("SANCTUARY_API_KEY")


class SanctuaryClient:
    def __init__(
        self,
        api_key: str = SANCTUARY_API_KEY,
        base_url: str = "https://api-sanctuary.i2cv.io",
        model_name: str = "bedrock-claude-3-5-sonnet-v1",
    ):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        )

    def create_chat_completion(
        self, messages: List[Dict], temperature: float = 0.3
    ) -> Any:
        """Make a chat completion request using Sanctuary API."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions", json=payload
            )

            # Enhanced error handling with response details
            if not response.ok:
                print(f"API Error {response.status_code}: {response.reason}")
                try:
                    error_body = response.json()
                    print(f"Error response body: {json.dumps(error_body, indent=2)}")
                except:
                    print(f"Raw error response: {response.text}")
                response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise
