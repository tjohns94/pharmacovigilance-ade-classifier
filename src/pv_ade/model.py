"""Model + tokenizer construction for the ablation.

Each model gets its own tokenizer (they're not interchangeable across checkpoints),
but the downstream classification head is identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def build_model(
    checkpoint: str,
    num_labels: int = 2,
) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase"]:
    """Load a pretrained encoder and wrap it with a sequence classification head."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    )
    return model, tokenizer
