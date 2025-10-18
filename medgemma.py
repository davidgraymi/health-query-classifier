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

# MedGemma endpoint
import requests
from auth import create_credentials, get_access_token_refresh_if_needed
import os
from cache import cache

_endpoint_url = os.environ.get('GCP_MEDGEMMA_ENDPOINT')

# Create credentials
secret_key_json = os.environ.get('GCP_MEDGEMMA_SERVICE_ACCOUNT_KEY')
medgemma_credentials = create_credentials(secret_key_json)

# https://cloud.google.com/vertex-ai/docs/reference/rest/v1beta1/projects.locations.endpoints.chat/completions
@cache.memoize()
def medgemma_get_text_response(
    messages: list,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    stream: bool = False,
    top_p: float | None = None,
    seed: int | None = None,
    stop: list[str] | str | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    model: str="tgi"
):
    """
    Makes a chat completion request to the configured LLM API (OpenAI-compatible).
    """
    headers = {
        "Authorization": f"Bearer {get_access_token_refresh_if_needed(medgemma_credentials)}",
        "Content-Type": "application/json",
    }

    # Based on the openai format
    payload = {
                "messages": messages,
                "max_tokens": max_tokens
              }


    if temperature is not None: payload["temperature"] = temperature
    if top_p is not None: payload["top_p"] = top_p
    if seed is not None: payload["seed"] = seed
    if stop is not None: payload["stop"] = stop
    if frequency_penalty is not None: payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None: payload["presence_penalty"] = presence_penalty


    response = requests.post(_endpoint_url, headers=headers, json=payload, stream=stream, timeout=60)
    try:
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.JSONDecodeError:
        # Log the problematic response for easier debugging in the future.
        print(f"Error: Failed to decode JSON from MedGemma. Status: {response.status_code}, Response: {response.text}")
        # Re-raise the exception so the caller knows something went wrong.
        raise
