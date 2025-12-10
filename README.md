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

### Training Script

```bash
python3 -m classifier.train
```

## Running the System Locally

### Prerequisites
*   Git
*   Python 3

### Setup & Configuration

1.  **Clone the repository**

    ```bash
    git clone https://github.com/davidgraymi/health-query-classifier.git
    cd health-query-classifier
    ```

2.  **Configure environment variables**

    This project uses an `env.list` file for configuration. Create this file in the root directory.
    ```ini
    # env.list
    HF_TOKEN="your-huggingface-token"
    ```
    *   **HF_TOKEN**: Access token can be generated via [huggingface](https://huggingface.co/settings/tokens). The token must have read permissions.

3.  **Create a python virtual environment**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

### Data Setup

```bash
python3 adapters/build_corpora.py
```

### Execution

```bash
python3 main.py
```
### Open UI
https://huggingface.co/spaces/taraky/Medical_Document_Retrieval
