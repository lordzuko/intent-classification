from __future__ import annotations

import logging

import torch
from flask import Flask
from flask import jsonify
from flask import request

from config import LABEL_COLUMNS
from config import MAX_TOKEN_COUNT
from config import PORT
from src.utils import get_model_and_tokenizer

app = Flask(__name__)
MODEL, TOKENIZER = get_model_and_tokenizer(eval=True)


@app.route("/ready")
def ready():
    if MODEL:
        return "OK", 200
    else:
        return "Not ready", 423


@app.route("/intent", methods=["POST"])
def intent():
    try:
        request_data = request.get_json()

        if not request_data:
            return (
                jsonify(label="BODY_MISSING", message="Request doesn't have a body."),
                400,
            )

        if "text" not in request_data:
            return (
                jsonify(
                    label="TEXT_MISSING",
                    message='"text" missing from request body.',
                ),
                400,
            )

        text = request_data["text"]

        if not isinstance(text, str):
            return jsonify(label="INVALID_TYPE", message='"text" is not a string.'), 400

        if not text:
            return jsonify(label="TEXT_EMPTY", message='"text" is empty.'), 400

        encoding = TOKENIZER.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_TOKEN_COUNT,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        _, test_prediction = MODEL(encoding["input_ids"], encoding["attention_mask"])
        test_prediction = test_prediction.flatten()
        test_logits = torch.sigmoid(test_prediction).detach().cpu()

        values = torch.topk(test_logits, 3).values
        indices = torch.topk(test_logits, 3).indices

        pred_intents = []
        for i, v in zip(indices, values):
            pred_intents.append(
                {"label": LABEL_COLUMNS[i], "confidence": round(v.item(), 3)},
            )
        return jsonify({"intents": pred_intents}), 200

    except Exception:
        error_message = "An error occurred while processing the request."
        logging.exception("Exception in /intent endpoint:")
        return jsonify(label="INTERNAL_ERROR", message=error_message), 500


def main():
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
