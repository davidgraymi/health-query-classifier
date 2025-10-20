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
from cache import cache
from transformers import pipeline

pipe = pipeline("text-generation", model="google/medgemma-4b-it")

@cache.memoize()
def medgemma_generate(
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
    return pipe(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        top_p=top_p,
        seed=seed,
        stop=stop,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
