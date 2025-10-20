from transformers import pipeline

pipe = pipeline("text-generation", model="google/medgemma-4b-it")

def medgemma_generate_response(messages: list):
    return pipe(messages)

if __name__ == "__main__":
    test_messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"""You are a medical assistant summarizing the EHR (FHIR) records for the patient David Gray.
                    Provide a concise summary of the patient's medical history, including any existing conditions, medications, and relevant past treatments.
                    Do not include personal opinions or assumptions, only factual information."""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hi there!"
                }
            ]
        }
    ]
    print(medgemma_generate_response(test_messages))
