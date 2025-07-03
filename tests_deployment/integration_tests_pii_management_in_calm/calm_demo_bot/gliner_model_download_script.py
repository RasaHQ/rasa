from pathlib import Path

from gliner import GLiNER
import os

def download_model(model_path: Path, model_name: str) -> None:
    """Download a Gliner model to the specified directory."""
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)
    model = GLiNER.from_pretrained(model_name)
    model.save_pretrained(model_path)


if __name__ == "__main__":
    local_model_path = Path("./gliner_model").resolve()
    download_model(
        model_path=local_model_path,
        model_name="urchade/gliner_multi_pii-v1"
    )
