import os
from pathlib import Path
from huggingface_hub import HfApi

# Load token from .env file (never hardcode tokens in source files).
# .env is listed in .gitignore and will never be committed to GitHub.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to env var

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError(
        "HF_TOKEN not found. Add it to your .env file:\n"
        "  HF_TOKEN=hf_your_token_here\n"
        "Or set it as an environment variable before running."
    )

api = HfApi(token=token)
repo_id = "rsd-06/periodyx-exoplanet-classifier-v4"

print(f"Creating repo {repo_id} (skipped if already exists)...")
try:
    api.create_repo(repo_id=repo_id, repo_type="model", private=False)
except Exception as e:
    print("Repo already exists, continuing.")

print("Uploading model file...")
api.upload_file(
    path_or_fileobj="models/exoplanet_classifier.joblib",
    path_in_repo="exoplanet_classifier_v4.joblib",
    repo_id=repo_id,
    repo_type="model"
)
print("Upload complete!")
