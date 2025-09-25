import os

from pathlib import Path
from gliner import GLiNER
from transformers import AutoTokenizer


def download_model(model_path: Path, model_name: str) -> None:
    """Download a Gliner model to the specified directory."""
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    # The default tokenizer for Gliner is DeBERTa v2, which results in
    # protobuf issues as we are using protobuf 5.29.5.
    # Use BERT tokenizer instead of DeBERTa v2.
    print(f"Downloading GLiNER model: {model_name}")
    print("Using BERT tokenizer to avoid protobuf issues...")

    try:
        # Load a BERT tokenizer that's compatible
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Load GLiNER model with custom tokenizer
        model = GLiNER.from_pretrained(model_name, tokenizer=tokenizer)
        model.save_pretrained(model_path)
        print(f"Successfully downloaded and saved model to {model_path}")

    except Exception as e:
        print(f"Error with BERT tokenizer: {e}")
        print("Trying with default tokenizer...")
        # Fallback to default tokenizer
        model = GLiNER.from_pretrained(model_name)
        model.save_pretrained(model_path)
        print(f"Successfully downloaded and saved model to {model_path}")


if __name__ == "__main__":
    local_model_path = Path("./gliner_model").resolve()
    download_model(
        model_path=local_model_path,
        model_name="urchade/gliner_multi_pii-v1"
    )
