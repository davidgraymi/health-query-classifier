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

import json
import re
import os
import base64

from gemini import gemini_get_text_response
from medgemma import medgemma_get_text_response
from gemini_tts import synthesize_gemini_tts

INTERVIEWER_VOICE = "Aoede"

def read_symptoms_json():
    # Load the list of symptoms for each condition from a JSON file
    with open("symptoms.json", 'r') as f:
        return json.load(f)

def read_patient_and_conditions_json():
    # Load all patient and condition data from the frontend assets
    with open(os.path.join(os.environ.get("FRONTEND_BUILD", "frontend/build"), "assets", "patients_and_conditions.json"), 'r') as f:
        return json.load(f)

def get_patient(patient_name):
    """Helper function to locate a patient record by name. Raises StopIteration if not found."""
    return next(p for p in PATIENTS if p["name"] == patient_name)

def read_fhir_json(patient):
    # Load the FHIR (EHR) JSON file for a given patient
    with open(os.path.join(os.environ.get("FRONTEND_BUILD", "frontend/build"), patient["fhirFile"].lstrip("/")), 'r') as f:
        return json.load(f)

def get_ehr_summary_per_patient(patient_name):
    # Returns a concise EHR summary for the patient, using LLM if not already cached
    patient = get_patient(patient_name)
    if patient.get("ehr_summary"):
        return patient["ehr_summary"]
    # Use MedGemma to summarize the EHR for the patient
    ehr_summary = medgemma_get_text_response([
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"""You are a medical assistant summarizing the EHR (FHIR) records for the patient {patient_name}.
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
                    "text": json.dumps(read_fhir_json(patient))
                }
            ]
        }
    ])
    patient["ehr_summary"] = ehr_summary
    return ehr_summary

PATIENTS = read_patient_and_conditions_json()["patients"]
SYMPTOMS = read_symptoms_json()
   
def patient_roleplay_instructions(patient_name, condition_name, previous_answers):
    """
    Generates structured instructions for the LLM to roleplay as a patient, including persona, scenario, and symptom logic.
    """
    # This assumes SYMPTOMS is a globally available dictionary as in the user's example
    patient = get_patient(patient_name)
    symptoms = "\n".join(SYMPTOMS[condition_name])

    return f"""
        SYSTEM INSTRUCTION: Before the interview begins, silently review the optional symptoms and decide which ones you have.

        ### Your Persona ###
        - **Name:** {patient_name}
        - **Age:** {patient["age"]}
        - **Gender:** {patient["gender"]}
        - **Your Role:** You are to act as this patient. Behave naturally and realistically.

        ### Scenario ###
        You are at home, participating in a remote pre-visit interview with a clinical assistant. You recently booked an appointment with your doctor because you've been feeling unwell. You are now answering the assistant's questions about your symptoms.

        ### Your Medical History ###
        You have a known history of **{patient["existing_condition"]}**. You should mention this if asked about your medical history, but you do not know if it is related to your current problem.

        ### Your Current Symptoms ###
        This is how you have been feeling. Base all your answers on these facts. Do not invent new symptoms.
        ---
        {symptoms}
        ---

        ### Critical Rules of Roleplay ###
        - **Handle Optional Symptoms:** Your symptom list may contain optional symptoms (e.g., "I might have..."). Before the interview starts, you MUST silently decide 'yes' or 'no' for each optional symptom. A 50% chance for each is a good approach. Remember your choices and be consistent throughout the entire interview.
        - **Act as the Patient:** Your entire response must be ONLY what the patient would say. Do not add external comments, notes, or clarifications (e.g., do not write "[I am now describing the headache]").
        - **No Guessing:** You DO NOT know your diagnosis or the name of your condition. Do not guess or speculate about it.
        - **Answer Only What Is Asked:** Do not volunteer your entire list of symptoms at once. Respond naturally to the specific question asked by the interviewer.

        ### Your previous health history ###
        {patient["ehr_summary"]}

        ### Your previous answers ###
        ---
        {previous_answers}
        ---
    """

def interviewer_roleplay_instructions(patient_name):
    # Returns detailed instructions for the LLM to roleplay as the interviewer/clinical assistant
    return f"""
        SYSTEM INSTRUCTION: Always think silently before responding.

        ### Persona & Objective ###
        You are a clinical assistant. Your objective is to interview a patient, {patient_name.split(" ")[0]}, and build a comprehensive and detailed report for their PCP.

        ### Critical Rules ###
        - **No Assessments:** You are NOT authorized to provide medical advice, diagnoses, or express any form of assessment to the patient.
        - **Question Format:** Ask only ONE question at a time. Do not enumerate your questions.
        - **Question Length:** Each question must be 20 words or less.
        - **Question Limit:** You have a maximum of 20 questions.

        ### Interview Strategy ###
        - **Clinical Reasoning:** Based on the patient's responses and EHR, actively consider potential diagnoses.
        - **Differentiate:** Formulate your questions strategically to help differentiate between these possibilities.
        - **Probe Critical Clues:** When a patient's answer reveals a high-yield clue (e.g., recent travel, a key symptom like rapid breathing), ask one or two immediate follow-up questions to explore that clue in detail before moving to a new line of questioning.
        - **Exhaustive Inquiry:** Your goal is to be thorough. Do not end the interview early. Use your full allowance of questions to explore the severity, character, timing, and context of all reported symptoms.
        - **Fact-Finding:** Focus exclusively on gathering specific, objective information.

        ### Context: Patient EHR ###
        You MUST use the following EHR summary to inform and adapt your questioning. Do not ask for information already present here unless you need to clarify it.
        EHR RECORD START
        {get_ehr_summary_per_patient(patient_name)}
        EHR RECORD END

        ### Procedure ###
        1.  **Start Interview:** Begin the conversation with this exact opening: "Thank you for booking an appointment with your primary doctor. I am an assistant here to ask a few questions to help your doctor prepare for your visit. To start, what is your main concern today?"
        2.  **Conduct Interview:** Proceed with your questioning, following all rules and strategies above.
        3.  **End Interview:** You MUST continue the interview until you have asked 20 questions OR the patient is unable to provide more information. When the interview is complete, you MUST conclude by printing this exact phrase: "Thank you for answering my questions. I have everything needed to prepare a report for your visit. End interview."
    """

def report_writer_instructions(patient_name: str) -> str:
    """
    Generates the system prompt with clear instructions, role, and constraints for the LLM.
    """
    ehr_summary = get_ehr_summary_per_patient(patient_name)

    return f"""<role>
You are a highly skilled medical assistant with expertise in clinical documentation.
</role>

<task>
Your task is to generate a concise yet clinically comprehensive medical intake report for a Primary Care Physician (PCP). This report will be based on a patient interview and their Electronic Health Record (EHR).
</task>

<guiding_principles>
To ensure the report is both brief and useful, you MUST adhere to the following two principles:

1.  **Principle of Brevity**:
    * **Use Professional Language**: Rephrase conversational patient language into standard medical terminology (e.g., "it hurts when I breathe deep" becomes "reports pleuritic chest pain").
    * **Omit Filler**: Do not include conversational filler, pleasantries, or repeated phrases from the interview.

2.  **Principle of Clinical Relevance (What is "Critical Information")**:
    * **Prioritize the HPI**: The History of Present Illness is the most important section. Include key details like onset, duration, quality of symptoms, severity, timing, and modifying factors.
    * **Include "Pertinent Negatives"**: This is critical. You MUST include symptoms the patient **denies** if they are relevant to the chief complaint. For example, if the chief complaint is a cough, denying "fever" or "shortness of breath" is critical information and must be included in the report.
    * **Filter History**: Only include historical EHR data that could reasonably be related to the patient's current complaint. For a cough, a history of asthma or smoking is relevant; a past appendectomy is likely not.
</guiding_principles>

<instructions>
1.  **Primary Objective**: Synthesize the interview and EHR into a clear, organized report, strictly following the <guiding_principles>.
2.  **Content Focus**:
    * **Main Concern**: State the patient's chief complaint.
    * **Symptoms**: Detail the History of Present Illness, including pertinent negatives.
    * **Relevant History**: Include only relevant information from the EHR.
3.  **Constraints**:
    * **Factual Information Only**: Report only the facts. No assumptions.
    * **No Diagnosis or Assessment**: Do not provide a diagnosis.
</instructions>

<ehr_data>
<ehr_record_start>
{ehr_summary}
<ehr_record_end>
</ehr_data>

<output_format>
The final output MUST be ONLY the full, updated Markdown medical report.
DO NOT include any introductory phrases, explanations, or any text other than the report itself.
</output_format>"""

def write_report(patient_name: str, interview_text: str, existing_report: str = None) -> str:
    """
    Constructs the full prompt, sends it to the LLM, and processes the response.
    This function handles both the initial creation and subsequent updates of a report.
    """
    # Generate the detailed system instructions
    instructions = report_writer_instructions(patient_name)

    # If no existing report is provided, load a default template from a string.
    if not existing_report:
        with open("report_template.txt", 'r') as f:
            existing_report = f.read()

    # Construct the user prompt with the specific task and data
    user_prompt = f"""<interview_start>
{interview_text}
<interview_end>

<previous_report>
{existing_report}
</previous_report>

<task_instructions>
Update the report in the `<previous_report>` tags using the new information from the `<interview_start>` section.
1.  **Integrate New Information**: Add new symptoms or details from the interview into the appropriate sections.
2.  **Update Existing Information**: If the interview provides more current information, replace outdated details.
3.  **Maintain Conciseness**: Remove any information that is no longer relevant.
4.  **Preserve Critical Data**: Do not remove essential historical data (like Hypertension) that could be vital for diagnosis, but ensure it is presented concisely under "Relevant Medical History".
5.  **Adhere to Section Titles**: Do not change the existing Markdown section titles.
</task_instructions>

Now, generate the complete and updated medical report based on all system and user instructions. Your response should be the Markdown text of the report only."""

    # Assemble the full message payload for the LLM API
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": instructions}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}]
        }
    ]

    report = medgemma_get_text_response(messages)
    cleaned_report = re.sub(r'<unused94>.*?</unused95>', '', report, flags=re.DOTALL)
    cleaned_report = cleaned_report.strip()

    # The LLM sometimes wraps the markdown report in a markdown code block.
    # This regex checks if the entire string is a code block and extracts the content.
    match = re.match(r'^\s*```(?:markdown)?\s*(.*?)\s*```\s*$', cleaned_report, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned_report = match.group(1)

    return cleaned_report.strip()



def stream_interview(patient_name, condition_name):
    print(f"Starting interview simulation for patient: {patient_name}, condition: {condition_name}")
    # Prepare roleplay instructions and initial dialog (using existing helper functions)
    interviewer_instructions = interviewer_roleplay_instructions(patient_name)
    
    # Determine voices for TTS
    patient = get_patient(patient_name)
    patient_voice = patient["voice"]
    
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
    number_of_questions_limit = 30
    for i in range(number_of_questions_limit):
        # Get the next interviewer question from MedGemma
        interviewer_question_text = medgemma_get_text_response(
            messages=dialog,
            temperature=0.1,
            max_tokens=2048,
            stream=False
        )
        # Process optional "thinking" text (if present in the LLM output)
        thinking_search = re.search('<unused94>(.+?)<unused95>', interviewer_question_text, re.DOTALL)
        if thinking_search:
            thinking_text = thinking_search.group(1)
            interviewer_question_text = interviewer_question_text.replace(f'<unused94>{thinking_text}<unused95>', "")
            if i == 0:
                # Only yield the "thinking" summary for the first question
                thinking_text = gemini_get_text_response(
                    f"""Provide a summary of up to 100 words containing only the reasoning and planning from this text,
                    do not include instructions, use first person: {thinking_text}""")
                yield json.dumps({
                        "speaker": "interviewer thinking",
                    "text": thinking_text
                })

        # Clean up the text for TTS and display
        clean_interviewer_text = interviewer_question_text.replace("End interview.", "").strip()

        # Generate audio for the interviewer's question using Gemini TTS
        audio_data, mime_type = synthesize_gemini_tts(f"Speak in a slightly upbeat and brisk manner, as a friendly clinician: {clean_interviewer_text}", INTERVIEWER_VOICE)
        audio_b64 = None
        if audio_data and mime_type:
            audio_b64 = f"data:{mime_type};base64,{base64.b64encode(audio_data).decode('utf-8')}"

        # Yield interviewer message (text and audio)
        yield json.dumps({
            "speaker": "interviewer",
            "text": clean_interviewer_text,
            "audio": audio_b64
        })
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

        # Get the patient's response from Gemini (roleplay LLM)
        patient_response_text = gemini_get_text_response(f"""
        {patient_roleplay_instructions(patient_name, condition_name, full_interview_q_a)}\n\n
        Question: {interviewer_question_text}""")

        # Generate audio for the patient's response
        audio_data, mime_type = synthesize_gemini_tts(f"Say this in faster speed, using a sick tone: {patient_response_text}", patient_voice)
        audio_b64 = None
        if audio_data and mime_type:
            audio_b64 = f"data:{mime_type};base64,{base64.b64encode(audio_data).decode('utf-8')}"

        # Yield patient message (text and audio)
        yield json.dumps({
            "speaker": "patient",
            "text": patient_response_text,
            "audio": audio_b64
        })
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
        write_report_text = write_report(patient_name, full_interview_q_a_with_new_q_a, write_report_text)
        full_interview_q_a += most_recent_q_a
        yield json.dumps({
            "speaker": "report",
            "text": write_report_text
        })

    print(f"""Interview simulation completed for patient: {patient_name}, condition: {condition_name}.
          Patient profile used:
          {patient_roleplay_instructions(patient_name, condition_name, full_interview_q_a)}""")
    # Add this at the end to signal end of stream
    yield json.dumps({"event": "end"})