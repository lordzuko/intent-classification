from __future__ import annotations

import torch
from transformers import BertTokenizerFast

from config import CHECKPOINT_PATH
from config import LABEL_COLUMNS
from config import ML_BERT_MODEL_NAME
from src.intent_classifier import IntentClassifier


def get_model_and_tokenizer(eval=True):
    """Get the model and tokenizer for usage

    Args:
        eval (bool, optional): Set model in evaluation mode. Defaults to True.

    Returns:
        model, tokenizer: return the loaded model and tokenizer
    """
    tokenizer = BertTokenizerFast.from_pretrained(ML_BERT_MODEL_NAME)
    model = IntentClassifier.load_from_checkpoint(
        CHECKPOINT_PATH,
        n_classes=len(LABEL_COLUMNS),
    )
    if eval:
        model.eval()
        model.freeze()
    if torch.cuda.is_available():
        model.to("cuda")

    return model, tokenizer
