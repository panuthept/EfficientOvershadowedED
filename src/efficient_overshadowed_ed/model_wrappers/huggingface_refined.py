from typing import List
from huggingface_hub import hf_hub_download
from efficient_overshadowed_ed.data_types.basic_types import Passage, Span, Entity
from efficient_overshadowed_ed.cft_refined_package.inference.processor import Refined
from efficient_overshadowed_ed.cft_refined_package.data_types.doc_types import Doc as ReFinED_Doc
from efficient_overshadowed_ed.cft_refined_package.data_types.base_types import Span as ReFinED_Span


class HuggingfaceReFinED:
    def __init__(self, *args, **kwargs):
        self.model = Refined(*args, **kwargs)

    def __call__(
            self, 
            passages: List[Passage]|Passage,
            batch_size: int = 8,
            threshold: float = 0.25,
    ) -> List[Passage]:
        passages = [passages] if isinstance(passages, Passage) else passages

        refined_docs: List[ReFinED_Doc] = self.model.process_text_batch(
            texts=[d.text for d in passages],
            spanss=[[ReFinED_Span(text=span.surface_form, start=span.start, ln=span.end - span.start) for span in d.entities] for d in passages],
            max_batch_size=batch_size,
        )

        for refined_doc, passage in zip(refined_docs, passages):
            passage.entities = [
                Span(
                    start=refined_span.start, 
                    end=refined_span.start + refined_span.ln,
                    surface_form=refined_span.text,
                    pred_entity=Entity(
                        identifier="Q0" if refined_span.predicted_entity.wikidata_entity_id is None or refined_span.entity_linking_model_confidence_score < threshold or refined_span.predicted_entity.wikidata_entity_id == "Q-1" else refined_span.predicted_entity.wikidata_entity_id,
                        confident=refined_span.entity_linking_model_confidence_score,
                    ),
                    cand_entities=[
                        Entity(
                            identifier=wikidata_entity_id,
                            confident=pem,
                        ) for wikidata_entity_id, pem in refined_span.candidate_entities
                    ]
                ) for refined_span in refined_doc.spans
            ]
        return passages
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cache_dir: str = "./data",
        **kwargs,
    ):
        # Download model files
        model_file_or_model = hf_hub_download(repo_id=model_name_or_path, filename="model.pt", cache_dir=cache_dir, **kwargs)
        model_config_file_or_model_config = hf_hub_download(repo_id=model_name_or_path, filename="config.json", cache_dir=cache_dir, **kwargs)
        model_description_embeddings_file = hf_hub_download(repo_id=model_name_or_path, filename="precomputed_entity_descriptions_emb_wikipedia_6269457-300.np", cache_dir=cache_dir, **kwargs)
        return cls(
            model_file_or_model=model_file_or_model,
            model_config_file_or_model_config=model_config_file_or_model_config,
            model_description_embeddings_file=model_description_embeddings_file,
            data_dir=cache_dir,
            entity_set="wikipedia",
        )
    

if __name__ == "__main__":
    model = HuggingfaceReFinED.from_pretrained("panuthept/CFT-ReFinED")

    passages = [
        Passage(
            text='I absolutely love the MCU movies, but Spider-Man said it best in Civil War when he saw Cap throwing his shield and said, "That thing doesn’t obey the laws of physics at all."',
            entities=[
                Span(surface_form="shield"),
            ]
        ),
    ]

    passages = model(passages)
    print(passages)