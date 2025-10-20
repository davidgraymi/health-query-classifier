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

from medgemma import medgemma_generate
   
def interviewer_roleplay_instructions():
    # Returns detailed instructions for the LLM to roleplay as the interviewer/clinical assistant

    return f"""
        SYSTEM INSTRUCTION: Always think silently before responding.

        ### Persona & Objective ###
        You are a clinical assistant. Your objective is to interview a patient, and build a comprehensive and detailed report for their PCP.

        ### Critical Rules ###
        - **No Assessments:** You are NOT authorized to provide medical advice, diagnoses, or express any form of assessment to the patient.
        - **Question Format:** Ask only ONE question at a time. Do not enumerate your questions.
        - **Question Length:** Each question must be 20 words or less.
        - **Question Limit:** You have a maximum of 20 questions.

        ### Interview Strategy ###
        - **Clinical Reasoning:** Based on the patient's responses, actively consider potential diagnoses.
        - **Differentiate:** Formulate your questions strategically to help differentiate between these possibilities.
        - **Probe Critical Clues:** When a patient's answer reveals a high-yield clue (e.g., recent travel, a key symptom like rapid breathing), ask one or two immediate follow-up questions to explore that clue in detail before moving to a new line of questioning.
        - **Exhaustive Inquiry:** Your goal is to be thorough. Do not end the interview early. Use your full allowance of questions to explore the severity, character, timing, and context of all reported symptoms.
        - **Fact-Finding:** Focus exclusively on gathering specific, objective information.

        ### Procedure ###
        1.  **Start Interview:** Begin the conversation with this exact opening: "Hello. I am an assistant here to ask a few questions to help your doctor prepare for your visit. To start, what is your main concern today?"
        2.  **Conduct Interview:** Proceed with your questioning, following all rules and strategies above.
        3.  **End Interview:** You MUST continue the interview until you have asked 20 questions OR the patient is unable to provide more information. When the interview is complete, you MUST conclude by printing this exact phrase: "Thank you for answering my questions. I have everything needed to prepare a report for your visit. End interview."
    """

def report_writer_instructions() -> str:
    """
    Generates the system prompt with clear instructions, role, and constraints for the LLM.
    """

    return f"""<role>
You are a highly skilled medical assistant with expertise in clinical documentation.
</role>

<task>
Your task is to generate a concise yet clinically comprehensive medical intake report for a Primary Care Physician (PCP). This report will be based on a patient interview.
</task>

<guiding_principles>
To ensure the report is both brief and useful, you MUST adhere to the following two principles:

1.  **Principle of Brevity**:
    * **Use Professional Language**: Rephrase conversational patient language into standard medical terminology (e.g., "it hurts when I breathe deep" becomes "reports pleuritic chest pain").
    * **Omit Filler**: Do not include conversational filler, pleasantries, or repeated phrases from the interview.

2.  **Principle of Clinical Relevance (What is "Critical Information")**:
    * **Prioritize the HPI**: The History of Present Illness is the most important section. Include key details like onset, duration, quality of symptoms, severity, timing, and modifying factors.
    * **Include "Pertinent Negatives"**: This is critical. You MUST include symptoms the patient **denies** if they are relevant to the chief complaint. For example, if the chief complaint is a cough, denying "fever" or "shortness of breath" is critical information and must be included in the report.
    * **Filter History**: Only include historical data that could reasonably be related to the patient's current complaint. For a cough, a history of asthma or smoking is relevant; a past appendectomy is likely not.
</guiding_principles>

<instructions>
1.  **Primary Objective**: Synthesize the interview into a clear, organized report, strictly following the <guiding_principles>.
2.  **Content Focus**:
    * **Main Concern**: State the patient's chief complaint.
    * **Symptoms**: Detail the History of Present Illness, including pertinent negatives.
    * **Relevant History**: Include only relevant historical information.
3.  **Constraints**:
    * **Factual Information Only**: Report only the facts. No assumptions.
    * **No Diagnosis or Assessment**: Do not provide a diagnosis.
</instructions>

<output_format>
The final output MUST be ONLY the full, updated Markdown medical report.
DO NOT include any introductory phrases, explanations, or any text other than the report itself.
</output_format>"""

def write_report(interview_text: str, existing_report: str = None) -> str:
    """
    Constructs the full prompt, sends it to the LLM, and processes the response.
    This function handles both the initial creation and subsequent updates of a report.
    """
    # Generate the detailed system instructions
    instructions = report_writer_instructions()

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

    report = medgemma_generate(messages)
    cleaned_report = re.sub(r'<unused94>.*?</unused95>', '', report, flags=re.DOTALL)
    cleaned_report = cleaned_report.strip()

    # The LLM sometimes wraps the markdown report in a markdown code block.
    # This regex checks if the entire string is a code block and extracts the content.
    match = re.match(r'^\s*```(?:markdown)?\s*(.*?)\s*```\s*$', cleaned_report, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned_report = match.group(1)

    return cleaned_report.strip()
