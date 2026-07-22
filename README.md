# Named Entity Recognition System

Named Entity Recognition System is an NLP sequence-labeling project that identifies and classifies named entities in text. The project uses CoNLL-2003 style IOB tags, compares recurrent deep learning models, and fine-tunes a Hugging Face transformer for token classification.

## Project Goal

The goal is to tag every token in a sentence as either outside an entity or part of one of four named entity types:

| Entity type | Meaning |
|---|---|
| `PER` | Person names |
| `ORG` | Organizations |
| `LOC` | Locations |
| `MISC` | Miscellaneous named entities |

The project uses the full CoNLL-2003 NER tag set:

```text
O
B-PER, I-PER
B-ORG, I-ORG
B-LOC, I-LOC
B-MISC, I-MISC
```

## Files

| File | Purpose |
|---|---|
| `Named Entity Recognition System.ipynb` | Main notebook for dataset checks, preprocessing, token/label alignment, model training, evaluation, and testing |
| `ner_tagger_app.py` | Gradio app for highlighting entities in user text |
| `requirements.txt` | Python dependencies |
| `conll2003_csv/` | Local CSV export of the CoNLL-2003 train, validation, and test splits |
| `conll2003_dataset/` | Local Hugging Face dataset saved to disk |
| `embeddings/` | Local GloVe vectors for custom sequence models |
| `checkpoints/` | Local model checkpoints and tokenizer files |
| `README_assets/` | Charts and images generated from the notebook for this README |

## Notebook Workflow

The notebook follows this order:

1. Import all libraries in the first cell.
2. Load the local CoNLL-2003 dataset from `conll2003_dataset/` or fall back to Hugging Face datasets.
3. Inspect split sizes, columns, token lengths, missing values, and entity-tag distribution.
4. Decode numeric `ner_tags` into IOB labels.
5. Apply and validate the IOB tagging scheme.
6. Tokenize text and align word-level entity labels to subword tokens.
7. Build vocabularies and data loaders for custom sequence models.
8. Handle OOV words with character-level embeddings for recurrent models.
9. Train and compare LSTM, BiLSTM, and BiLSTM + CRF architectures.
10. Fine-tune a Hugging Face transformer for token classification.
11. Evaluate each model with per-entity precision, recall, and F1 using `seqeval`.
12. Save best and last checkpoints with model weights, optimizer state, label mappings, and hyperparameters.
13. Compare models and test the best model on new text.

## Dataset Checks

The local Hugging Face dataset metadata reports:

| Split | Examples |
|---|---:|
| Train | 14,041 |
| Validation | 3,250 |
| Test | 3,453 |

The dataset columns are:

| Column | Purpose |
|---|---|
| `id` | Sentence identifier |
| `tokens` | Tokenized sentence |
| `pos_tags` | Part-of-speech tags |
| `chunk_tags` | Syntactic chunk tags |
| `ner_tags` | Named entity labels |

## Tagging Scheme

The project uses IOB2 tagging:

| Tag pattern | Meaning |
|---|---|
| `O` | Token is outside any named entity |
| `B-TYPE` | Token begins an entity span |
| `I-TYPE` | Token continues an entity span of the same type |

Example:

| Token | Label |
|---|---|
| `Peter` | `B-PER` |
| `Blackburn` | `I-PER` |
| `works` | `O` |
| `in` | `O` |
| `Brussels` | `B-LOC` |

## Main Hyperparameters

The custom LSTM, BiLSTM, and BiLSTM + CRF models use the same core hyperparameters as the previous Consumer Complaint project:

| Hyperparameter | Value |
|---|---:|
| Embedding dimension | Inferred from the GloVe file |
| Pre-trained word embeddings | GloVe |
| Default embedding path | `embeddings/glove.6B.300d.txt` |
| Freeze embeddings | False |
| Character embedding dimension | 32 |
| Character hidden size | 64 |
| Hidden size | 512 |
| Number of recurrent layers | 2 |
| Batch size | 128 |
| Learning rate | 5e-4 |
| Max epochs | 50 |
| Dropout | 0.3 |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| Early stopping patience | 3 |
| Minimum improvement | 0.001 |
| Bidirectional recurrent layers | True for BiLSTM and BiLSTM + CRF |

To use pre-trained GloVe embeddings, place the vector file in `embeddings/` and keep this value in the first notebook cell:

```python
CUSTOM_HYPERPARAMS['pretrained_embedding_path'] = str(EMBEDDINGS_DIR / 'glove.6B.300d.txt')
```

The notebook infers the vector dimension from the local GloVe file and uses it for the custom model embedding layer.

## Model Plan

### LSTM

The LSTM baseline uses:

- Pre-trained GloVe word embeddings.
- Optional character-level embeddings for OOV handling.
- Unidirectional `nn.LSTM`.
- Token-level linear classifier.
- Cross-entropy loss with ignored padding labels.

### BiLSTM

The BiLSTM model uses:

- Word embeddings.
- Character-level embeddings or subword fallback features.
- Bidirectional `nn.LSTM`.
- Token-level linear classifier.
- Cross-entropy loss with ignored padding labels.

### BiLSTM + CRF

The BiLSTM + CRF model uses:

- Word embeddings.
- Character-level embeddings.
- Bidirectional `nn.LSTM`.
- Linear emission layer.
- CRF decoding to model valid tag transitions across the whole sentence.

### Transformer

The transformer model fine-tunes a Hugging Face model for token classification.

| Hyperparameter | Value |
|---|---:|
| Base model | `bert-base-cased` |
| Max token length | 256 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Max epochs | 5 |
| Early stopping patience | 2 |
| Warmup ratio | 0.1 |
| Mixed precision AMP | True on CUDA |
| Optimizer | AdamW |
| Metric | Entity-level F1 |

`bert-base-cased` is a good first transformer choice because capitalization is useful for named entity recognition.

## Evaluation

Use `seqeval` for entity-level metrics:

| Metric | Why it matters |
|---|---|
| Precision | How many predicted entity spans are correct |
| Recall | How many true entity spans are found |
| F1 | Balance between precision and recall |
| Per-entity F1 | Shows whether the model is better at `PER`, `ORG`, `LOC`, or `MISC` |

Each model produces a `seqeval` classification report, and the notebook compares the models using precision, recall, F1, and accuracy.

Models to compare:

- LSTM
- BiLSTM
- BiLSTM + CRF
- BERT token classifier

### Model Comparison

| Model | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| BERT | 0.8912 | 0.9069 | 0.8990 | 0.9713 |
| BiLSTM + CRF | 0.8486 | 0.8407 | 0.8446 | 0.9693 |
| BiLSTM | 0.8197 | 0.8201 | 0.8199 | 0.9678 |
| LSTM | 0.7341 | 0.7587 | 0.7462 | 0.9581 |

### Where the CRF Improves Boundary Detection

The BiLSTM + CRF achieved a higher test F1 score (0.8446) than the standard BiLSTM (0.8199), an improvement of about 2.47 percentage points. A standard BiLSTM predicts each token label independently, while the CRF learns transition scores between consecutive IOB labels and decodes the best label sequence for the complete sentence. This can discourage inconsistent transitions, such as `I-ORG` appearing without a suitable `B-ORG`, and helps produce more coherent entity boundaries.

The transformer model used for the BERT token classifier is:

```text
bert-base-cased
```

The notebook also saves a full comparison checkpoint:

```text
checkpoints/lstm_bilstm_crf_bert_ner_comparison_checkpoint.pt
```

## Gradio App

The project includes `ner_tagger_app.py`.

The app:

- Accepts raw user text.
- Runs token classification.
- Groups predicted entity spans.
- Highlights entities by type.
- Uses a local fine-tuned checkpoint when available.
- Falls back to a Hugging Face demo model when local checkpoints are not present.

## Example App Input

```text
Peter Blackburn works for the European Commission in Brussels.
```

Expected entities:

| Entity | Type |
|---|---|
| `Peter Blackburn` | `PER` |
| `European Commission` | `ORG` |
| `Brussels` | `LOC` |

## Notes

- Keep the notebook as the source of truth for training and evaluation.
- Save model weights, optimizer states, scheduler states, label mappings, tokenizer files, histories, and hyperparameters in checkpoints.
- Upload large checkpoints with Git LFS or release assets instead of normal Git commits.
