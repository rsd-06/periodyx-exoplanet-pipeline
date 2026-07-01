import os
from huggingface_hub import HfApi

# Token is read from the HF_TOKEN environment variable (set before running)
# or from `huggingface-cli login`. Never hardcode tokens in source files.
api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "rsd-06/periodyx-exoplanet-classifier-v1"

print(f"Creating repo {repo_id}...")
try:
    api.create_repo(repo_id=repo_id, repo_type="model", private=False)
except Exception as e:
    print("Repo might already exist:", e)

print("Uploading model file...")
api.upload_file(
    path_or_fileobj="models/exoplanet_classifier.joblib",
    path_in_repo="exoplanet_classifier_v1.joblib",
    repo_id=repo_id,
    repo_type="model"
)
print("Upload complete!")
