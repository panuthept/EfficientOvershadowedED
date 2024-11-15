from typing import List, Any
from efficient_overshadowed_ed.data_types.basic_types import Passage


def texts_to_passages(texts: List[str]|str = None, passages: List[Passage]|Passage = None) -> List[Passage]:
    # Cast `texts` to `passages` if `passages` is not provided
    if passages is None:
        assert texts is not None, "Either `text` or `passages` must be provided."
        if isinstance(texts, list):
            passages = [Passage(text=t) for t in texts]
        else:
            passages = [Passage(text=texts)]
    # Ensure that `passages` is a list of `Passage` objects
    if not isinstance(passages, list):
        passages = [passages]
    return passages


def pad_1d_sequence(sequences: List[List[Any]], pad_value: Any, pad_length: int) -> List[Any]:
    padded_sequences = []
    for sequence in sequences:
        assert len(sequence) <= pad_length, f"Length of `sequence` ({len(sequence)}) must be less than or equal to `pad_length` ({pad_length})."
        padded_sequences.append(sequence + [pad_value] * (pad_length - len(sequence)))
    return padded_sequences