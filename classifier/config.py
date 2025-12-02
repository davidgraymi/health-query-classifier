import os

def load_env():
    if os.path.exists("env.list"):
        with open("env.list", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key] = value

load_env()
HF_TOKEN = os.getenv("HF_TOKEN")
