import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from efficient_overshadowed_ed.cft_refined_package.utilities.general_utils import get_tokenizer

NER_TAG_TO_IX = {
    "O": 0,
    "B-MENTION": 1,
    "I-MENTION": 2
}


@dataclass
class ModelConfig:
    data_dir: str
    transformer_name: str
    max_seq: int = 510
    learning_rate: float = 5e-5
    num_train_epochs: int = 2
    freeze_all_bert_layers: bool = False
    gradient_accumulation_steps: int = 1
    per_gpu_batch_size: int = 12
    freeze_embedding_layers: bool = False
    freeze_layers: List[str] = field(default_factory=lambda: [])
    n_gpu: int = 4
    lr_ner_scale: int = 100
    ner_layer_dropout: float = 0.10  # was 0.15
    ed_layer_dropout: float = 0.05  # should add pem specific dropout
    max_candidates: int = 30
    warmup_steps: int = 10000  # 5000 could be used when restoring model
    logging_steps: int = 500
    save_steps: int = 500
    detach_ed_layer: bool = True
    only_ner: bool = False
    only_ed: bool = False
    md_layer_dropout: float = 0.1
    debug: bool = False

    beta_desc: float = 0.0
    beta_et: float = 0.0
    beta_ed: float = 0.0

    sep_token_id: Optional[int] = None
    cls_token_id: Optional[int] = None
    mask_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    vocab_size: Optional[int] = None

    ner_tag_to_ix: Dict[str, int] = field(default_factory=lambda: NER_TAG_TO_IX)

    def __post_init__(self):
        tokenizer = get_tokenizer(transformer_name=self.transformer_name, data_dir=self.data_dir)
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size

    @classmethod
    def from_file(cls, filename: str, data_dir: str):
        with open(filename, "r") as f:
            cfg = json.load(f)
            transformer_name = cfg["transformer_name"]
            cfg = {k: v for k, v in cfg.items() if k in cls.__dict__}
            cfg["data_dir"] = data_dir
            cfg["transformer_name"] = transformer_name
            if cfg["transformer_name"] == "roberta-base":
                cfg["transformer_name"] = "roberta_base_model"
            return cls(**cfg)

    def to_file(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.__dict__, f)
