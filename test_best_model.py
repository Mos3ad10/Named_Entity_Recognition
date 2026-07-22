"""Test the best saved NER model on a sentence."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
BERT_MODEL_DIR = CHECKPOINT_DIR / "transformer_ner_model"
BERT_TOKENIZER_DIR = CHECKPOINT_DIR / "transformer_ner_tokenizer"
BERT_CHECKPOINT = CHECKPOINT_DIR / "BERT_best.pt"


def load_best_model():
    if not BERT_MODEL_DIR.exists() or not BERT_TOKENIZER_DIR.exists():
        raise FileNotFoundError(
            "Trained BERT model was not found. Expected "
            f"{BERT_MODEL_DIR} and {BERT_TOKENIZER_DIR}."
        )

    tokenizer = AutoTokenizer.from_pretrained(BERT_TOKENIZER_DIR)
    model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_DIR)
    best_valid_loss = None

    if BERT_CHECKPOINT.exists():
        checkpoint = torch.load(BERT_CHECKPOINT, map_location="cpu", weights_only=False)
        best_valid_loss = checkpoint.get("best_valid_loss")

    return model, tokenizer, best_valid_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "text",
        nargs="?",
        default="Elon Musk met with officials from the United Nations in New York during the climate summit.",
        help="Sentence to tag.",
    )
    args = parser.parse_args()

    model, tokenizer, best_valid_loss = load_best_model()
    device = 0 if torch.cuda.is_available() else -1
    ner = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )

    print("Best model: BERT / bert-base-cased fine-tuned checkpoint")
    if best_valid_loss is not None:
        print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Input: {args.text}")
    print("\nPredictions:")

    entities = ner(args.text)
    if not entities:
        print("No named entities found.")
        return

    for entity in entities:
        word = entity["word"]
        label = entity["entity_group"]
        score = entity["score"]
        print(f"{word} -> {label} ({score:.4f})")


if __name__ == "__main__":
    main()
