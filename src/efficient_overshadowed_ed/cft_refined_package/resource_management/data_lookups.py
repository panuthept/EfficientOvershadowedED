import pickle
from typing import Mapping, List, Tuple, Dict, Any, Set

import numpy as np
import torch
import ujson as json
from nltk import PunktSentenceTokenizer
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedTokenizer, PreTrainedModel

from efficient_overshadowed_ed.cft_refined_package.resource_management.resource_manager import ResourceManager, get_mmap_shape
from efficient_overshadowed_ed.cft_refined_package.resource_management.aws import S3Manager
from efficient_overshadowed_ed.cft_refined_package.resource_management.lmdb_wrapper import LmdbImmutableDict
from efficient_overshadowed_ed.cft_refined_package.resource_management.loaders import load_human_qcode
import os


class LookupsInferenceOnly:

    def __init__(
            self, 
            entity_set: str, 
            data_dir: str, 
            use_precomputed_description_embeddings: bool = True,
            return_titles: bool = False,
            transformer_name: str = "roberta_base_model"
    ):
        self.entity_set = entity_set
        self.data_dir = data_dir
        self.use_precomputed_description_embeddings = use_precomputed_description_embeddings

        # Download wikipedia data
        self.wikipedia_data_dir = {
            "nltk_sentence_splitter_english": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/nltk_sentence_splitter_english.pickle", cache_dir=data_dir),
            "qcode_idx_to_class_idx": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/qcode_to_class_tns_6269457-138.np", cache_dir=data_dir),
            "descriptions_tns": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/descriptions_tns.pt", cache_dir=data_dir),
            "class_to_label": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/class_to_label.json", cache_dir=data_dir),
            "qcode_to_wiki": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/qcode_to_wiki.lmdb", cache_dir=data_dir),
            "human_qcodes": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/human_qcodes.json", cache_dir=data_dir),
            "qcode_to_idx": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/qcode_to_idx.lmdb", cache_dir=data_dir),
            "class_to_idx": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/class_to_idx.json", cache_dir=data_dir),
            "subclasses": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/subclasses.lmdb", cache_dir=data_dir),
            "wiki_pem": hf_hub_download(repo_id="panuthept/CFT-ReFinED", filename="wikipedia_data/pem.lmdb", cache_dir=data_dir),
        }

        # resource_manager = ResourceManager(entity_set=entity_set,
        #                                    data_dir=data_dir,
        #                                    model_name=None,
        #                                    s3_manager=S3Manager(),
        #                                    load_descriptions_tns=not use_precomputed_description_embeddings,
        #                                    load_qcode_to_title=return_titles
        #                                    )
        # resource_to_file_path = resource_manager.get_data_files()
        # self.resource_to_file_path = resource_to_file_path

        # replace all get_file and download_if needed
        # always use resource names that are provided instead of relying on same data_dirs
        # shape = (num_ents, max_num_classes)
        self.qcode_idx_to_class_idx = np.memmap(
            self.wikipedia_data_dir["qcode_idx_to_class_idx"],
            shape=get_mmap_shape(self.wikipedia_data_dir["qcode_idx_to_class_idx"]),
            mode="r",
            dtype=np.int16,
        )

        if not self.use_precomputed_description_embeddings:
            with open(self.wikipedia_data_dir["descriptions_tns"], "rb") as f:
                # (num_ents, desc_len)
                self.descriptions_tns = torch.load(f)
        else:
            # TODO: convert to numpy memmap to save space during training with multiple workers
            self.descriptions_tns = None

        self.pem: Mapping[str, List[Tuple[str, float]]] = LmdbImmutableDict(self.wikipedia_data_dir["wiki_pem"])

        with open(self.wikipedia_data_dir["class_to_label"], "r") as f:
            self.class_to_label: Dict[str, Any] = json.load(f)

        self.human_qcodes: Set[str] = load_human_qcode(self.wikipedia_data_dir["human_qcodes"])

        self.subclasses: Mapping[str, List[str]] = LmdbImmutableDict(self.wikipedia_data_dir["subclasses"])

        self.qcode_to_idx: Mapping[str, int] = LmdbImmutableDict(self.wikipedia_data_dir["qcode_to_idx"])

        with open(self.wikipedia_data_dir["class_to_idx"], "r") as f:
            self.class_to_idx = json.load(f)

        self.index_to_class = {y: x for x, y in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())
        self.max_num_classes_per_ent = self.qcode_idx_to_class_idx.shape[1]
        self.num_classes = len(self.class_to_idx)

        if return_titles:
            self.qcode_to_wiki: Mapping[str, str] = LmdbImmutableDict(self.wikipedia_data_dir["qcode_to_wiki"])
        else:
            self.qcode_to_wiki = None

        with open(self.wikipedia_data_dir["nltk_sentence_splitter_english"], 'rb') as f:
            self.nltk_sentence_splitter_english: PunktSentenceTokenizer = pickle.load(f)

        # can be shared
        self.tokenizers: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            # os.path.dirname(resource_to_file_path[transformer_name]),
            "roberta-base",
            # add_special_tokens=False,
            add_prefix_space=False,
            use_fast=True,
        )

        self.transformer_model_config = AutoConfig.from_pretrained(
            # os.path.dirname(resource_to_file_path[transformer_name])
            "roberta-base"
        )

    def get_transformer_model(self, transformer_name) -> PreTrainedModel:
        # cannot be shared so create a copy
        return AutoModel.from_pretrained(
            "roberta-base"
        )
