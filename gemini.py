# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import requests
from cache import cache  # new import replacing duplicate cache initialization

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Decorate the function to cache its results indefinitely.
@cache.memoize()
def gemini_get_text_response(prompt: str,
                                    stop_sequences: list = None,
                                    temperature: float = 0.1,
                                    max_output_tokens: int = 4000,
                                    top_p: float = 0.8,
                                    top_k: int = 10):
    """
    Makes a text generation request to the Gemini API.
    """

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "stopSequences": stop_sequences or ["Title"],
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": top_p,
            "topK": top_k
        }
    }

    response = requests.post(api_url, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]