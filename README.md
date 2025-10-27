## Table of Contents
- [Demo Description](#demo-description)
- [Technical Architecture](#technical-architecture)
- [Running the Demo Locally](#running-the-demo-locally)
- [Models used](#models-used)
- [Caching](#caching)
- [Disclaimer](#disclaimer)

# Healthcare Portal: Pre-visit Intake and Triaging

Healthcare providers often seek efficient ways to gather comprehensive patient information before appointments. This demo illustrates how AI could be used in an application to streamline pre-visit information collection and utilization. 

The demonstration first asks questions to gather pre-visit information.
After it has identified and collected relevant information, the demo application generates a pre-visit report using collected information (and could be easily expanded to using EHR FHIR resources). This type of intelligent pre-visit report can help providers be more efficient and effective while also providing an improved experience for patients compared to traditional intake forms.

At the conclusion of the demo, you can view an evaluation of the pre-visit report which provides additional insights into the quality of the demonstrated capabilities.

## Running the Demo Locally

### Prerequisites
*   Git
*   Python

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

    HF_TOKEN: Access token can be generated via [huggingface](https://huggingface.co/settings/tokens). The token must have read permissions.

3.  **Create a python virtual environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements-gemma.txt
    ```

### Execution

```bash
python3 main.py
```

# Models used
This demo uses the following models:

* MedGemma 4b-it: https://huggingface.co/google/medgemma-4b-it

 

## Caching
This demo is functional, and results are persistently cached to reduce environmental impact.

## Disclaimer
This demonstration is for illustrative purposes only and does not represent a finished or approved
product. It is not representative of compliance to any regulations or standards for
quality, safety or efficacy. Any real-world application would require additional development,
training, and adaptation. The experience highlighted in this demo shows MedGemma's baseline
capability for the displayed task and is intended to help developers and users explore possible
applications and inspire further development.
