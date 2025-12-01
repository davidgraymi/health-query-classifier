# Health Query Classifier & Research Retriever

## Team Members
*   **David Gray**
*   **Tarak Jha**
*   **Sravani Segireddy**
*   **Riley Millikan**
*   **Kent R. Spillner**

## Project Description
This project is a classifier that triages patient queries. If a query is identified as medical, the system retrieves relevant research and presents it to the user.

## Workflow
The system operates in two main stages to optimize patient care and provider efficiency:

1.  **Classification (Triage)**:
    The tool analyzes the user's input to determine if it is a medical query (requiring clinical attention) or an administrative query (scheduling, billing, etc.).

2.  **Research Retrieval**:
    If the query is classified as medical, the system searches through indexed medical databases (like PubMed and Miriad) to retrieve relevant research articles and Q/A pairs. This empowers the patient with trustworthy information and provides the doctor with context.

## Running the System Locally

### Prerequisites
*   Git
*   Python 3

### Setup & Configuration

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/davidgraymi/health-query-classifier.git
    cd health-query-classifier
    ```

2.  **Configure environment variables:**
    This project uses an `env.list` file for configuration. Create this file in the root directory.
    ```ini
    # env.list
    HF_TOKEN="your-huggingface-token"
    ```
    *   **HF_TOKEN**: Access token can be generated via [huggingface](https://huggingface.co/settings/tokens). The token must have read permissions.

3.  **Create a python virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Setup
To run the demo, you must populate the `data/corpora` directory with the required datasets.

1.  **Create the directory:**
    ```bash
    mkdir -p data/corpora
    ```

2.  **Add Corpora Files:**
    Place the following files in `data/corpora/`:
    *   `medical_qa.jsonl`: Dataset of medical questions and answers.
    *   `miriad_text.jsonl`: Dataset for relevant research/text.
    *   `unidoc_qa.jsonl` (Optional): Additional QA dataset.

    *Note: Ensure these files are formatted as JSONL (JSON Lines).*

3.  **PubMed Data (Optional):**
    To download and index PubMed data:
    ```bash
    python scripts/pubmed.py
    ```

### Execution

```bash
python3 main.py
```
