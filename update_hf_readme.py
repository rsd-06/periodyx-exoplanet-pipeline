import os
from pathlib import Path
from huggingface_hub import HfApi

# Load token from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN not found.")

api = HfApi(token=token)
repo_id = "rsd-06/periodyx-exoplanet-classifier-v5"

print(f"Uploading README.md to {repo_id}...")
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model"
)
print("Model card updated!")
