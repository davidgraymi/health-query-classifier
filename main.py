import re

from medgemma import medgemma_generate
from interview import interviewer_roleplay_instructions, write_report
from cache import create_cache_zip

def start_interview():
    print("\n************************************************************\n" +
          "Welcome to our health patient portal! This is a chat \n" +
          "interface designed to triage all inqueries to your medical \n" +
          "provider and staff.\n" +
          "************************************************************\n\n" +
          "One moment while we start your session...\n")
    
    interviewer_instructions = interviewer_roleplay_instructions()
    
    dialog = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": interviewer_instructions
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "start interview"
                }
            ]
        }
    ]

    write_report_text = ""
    full_interview_q_a = ""
    number_of_questions_limit = 21
    for i in range(number_of_questions_limit):
        # Get the next interviewer question from MedGemma
        interviewer_question_text = medgemma_generate(
            messages=dialog,
            temperature=0.1,
            max_tokens=2048,
        )
        # Process optional "thinking" text (if present in the LLM output)
        thinking_search = re.search('<unused94>(.+?)<unused95>', interviewer_question_text, re.DOTALL)
        if thinking_search:
            thinking_text = thinking_search.group(1)
            interviewer_question_text = interviewer_question_text.replace(f'<unused94>{thinking_text}<unused95>', "")
            print("[Interviewer is thinking...]\n" + thinking_text + "\n")

        # Clean up the text for display
        clean_interviewer_text = interviewer_question_text.replace("End interview.", "").strip()

        dialog.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": interviewer_question_text
            }]
        })
        if "End interview" in interviewer_question_text:
            # End the interview loop if the LLM signals completion
            break

        # Get the patient's response
        patient_response_text = input(f"\nAssistant: {clean_interviewer_text}\n\nMe: ")

        dialog.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": patient_response_text
            }]
        })
        # Track the full Q&A for context in future LLM calls
        most_recent_q_a = f"Q: {interviewer_question_text}\nA: {patient_response_text}\n"
        full_interview_q_a_with_new_q_a = "PREVIOUS Q&A:\n" + full_interview_q_a + "\nNEW Q&A:\n" + most_recent_q_a
        # Update the report after each Q&A
        write_report_text = write_report(full_interview_q_a_with_new_q_a, write_report_text)
        full_interview_q_a += most_recent_q_a

    print(f"""Interview complete.\nHere is the final report for your review:\n\n{write_report_text}""")

if __name__ == "__main__":
    start_interview()
