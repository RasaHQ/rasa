from huggingface_hub import snapshot_download
import sys

print(f"Downloading model files for {sys.argv[1]}...")
snapshot_download(
    repo_id=sys.argv[1], allow_patterns=["*.txt", "*.json", "*.h5", "*.model"]
)
