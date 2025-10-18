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

import google.generativeai as genai
import os
import struct
import re
import logging
from cache import cache

# Add these imports for MP3 conversion
from pydub import AudioSegment
import io

# --- Constants ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GENERATE_SPEECH = os.environ.get("GENERATE_SPEECH", "false").lower() == "true"
TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_RAW_AUDIO_MIME = "audio/L16;rate=24000"

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

genai.configure(api_key=GEMINI_API_KEY)

class TTSGenerationError(Exception):
    """Custom exception for TTS generation failures."""
    pass


# --- Helper functions for audio processing ---
def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """
    Parses bits per sample and rate from an audio MIME type string.
    e.g., "audio/L16;rate=24000" -> {"bits_per_sample": 16, "rate": 24000}
    """
    bits_per_sample = 16  # Default
    rate = 24000          # Default

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip().lower()
        if param.startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass # Keep default if parsing fails
        elif re.match(r"audio/l\d+", param): # Matches audio/L<digits>
             try:
                bits_str = param.split("l",1)[1]
                bits_per_sample = int(bits_str)
             except (ValueError, IndexError):
                pass # Keep default
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """
    Generates a WAV file header for the given raw audio data and parameters.
    Assumes mono audio.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1  # Mono
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ",
        16, 1, num_channels, sample_rate, byte_rate, block_align,
        bits_per_sample, b"data", data_size
    )
    return header + audio_data
# --- End of helper functions ---

def _synthesize_gemini_tts_impl(text: str, gemini_voice_name: str) -> tuple[bytes, str]:
    """
    Synthesizes English text using the Gemini API via the google-genai library.
    Returns a tuple: (processed_audio_data_bytes, final_mime_type).
    Raises TTSGenerationError on failure.
    """
    if not GENERATE_SPEECH:
        # This should ideally not be hit if the logic outside this function is correct,
        # but as a safeguard, we raise an error.
        raise TTSGenerationError(
            "GENERATE_SPEECH is not set. Please set it in your environment variables to generate speech."
        )

    try:
        model = genai.GenerativeModel(TTS_MODEL)

        generation_config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": gemini_voice_name
                    }
                }
            }
        }

        response = model.generate_content(
            contents=[text],
            generation_config=generation_config,
        )

        audio_part = response.candidates[0].content.parts[0]
        audio_data_bytes = audio_part.inline_data.data
        final_mime_type = audio_part.inline_data.mime_type
    except Exception as e:
        error_message = f"An unexpected error occurred with google-genai: {e}"
        logging.error(error_message)
        raise TTSGenerationError(error_message) from e

    if not audio_data_bytes:
        error_message = "No audio data was successfully retrieved or decoded."
        logging.error(error_message)
        raise TTSGenerationError(error_message)

    # --- Audio processing ---
    if final_mime_type:
        final_mime_type_lower = final_mime_type.lower()
        needs_wav_conversion = any(p in final_mime_type_lower for p in ("audio/l16", "audio/l24", "audio/l8")) or \
                               not final_mime_type_lower.startswith(("audio/wav", "audio/mpeg", "audio/ogg", "audio/opus"))

        if needs_wav_conversion:
            processed_audio_data = convert_to_wav(audio_data_bytes, final_mime_type)
            processed_audio_mime = "audio/wav"
        else:
            processed_audio_data = audio_data_bytes
            processed_audio_mime = final_mime_type
    else:
        logging.warning("MIME type not determined. Assuming raw audio and attempting WAV conversion (defaulting to %s).", DEFAULT_RAW_AUDIO_MIME)
        processed_audio_data = convert_to_wav(audio_data_bytes, DEFAULT_RAW_AUDIO_MIME)
        processed_audio_mime = "audio/wav"

    # --- MP3 compression ---
    if processed_audio_data:
        try:
            # Load audio into AudioSegment
            audio_segment = AudioSegment.from_file(io.BytesIO(processed_audio_data), format="wav")
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3")
            mp3_bytes = mp3_buffer.getvalue()
            return mp3_bytes, "audio/mpeg"
        except Exception as e:
            logging.warning("MP3 compression failed: %s. Falling back to WAV.", e)
            # Fallback to WAV if MP3 conversion fails
            return processed_audio_data, processed_audio_mime
    else:
        error_message = "Audio processing failed."
        logging.error(error_message)
        raise TTSGenerationError(error_message)

# Always create the memoized function first, so we can access its .key() method
_memoized_tts_func = cache.memoize()(_synthesize_gemini_tts_impl)

if GENERATE_SPEECH:
    def synthesize_gemini_tts_with_error_handling(*args, **kwargs) -> tuple[bytes | None, str | None]:
        """
        A wrapper for the memoized TTS function that catches errors and returns (None, None).
        This makes the audio generation more resilient to individual failures.
        """
        try:
            # Attempt to get the audio from the cache or by generating it.
            return _memoized_tts_func(*args, **kwargs)
        except TTSGenerationError as e:
            # If generation fails, log the error and return None, None.
            logging.error("Handled TTS Generation Error: %s. Continuing without audio for this segment.", e)
            return None, None

    synthesize_gemini_tts = synthesize_gemini_tts_with_error_handling
else:
    # When not generating speech, create a read-only function that only
    # checks the cache and does not generate new audio.
    def read_only_synthesize_gemini_tts(*args, **kwargs):
        """
        Checks cache for a result, but never calls the underlying TTS function.
        This is a 'read-only' memoization check.
        """
        # Generate the cache key using the memoized function's key method.
        key = _memoized_tts_func.__cache_key__(*args, **kwargs)

        # Check the cache directly using the generated key.
        _sentinel = object()
        result = cache.get(key, default=_sentinel)

        if result is not _sentinel:
            return result  # Cache hit

        # Cache miss
        logging.info("GENERATE_SPEECH is false and no cached result found for key: %s", key)
        return None, None

    synthesize_gemini_tts = read_only_synthesize_gemini_tts