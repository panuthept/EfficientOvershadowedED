import os
from pprint import pprint
from typing import Iterable, Optional, Dict

from refined_package.dataset_reading.entity_linking.dataset_factory import Datasets
from refined_package.data_types.doc_types import Doc
from refined_package.doc_preprocessing.preprocessor import Preprocessor
from refined_package.resource_management.resource_manager import ResourceManager
from refined_package.evaluation.metrics import Metrics, ClassificationMetrics, MultilabelClassificationMetrics
from refined_package.resource_management.aws import S3Manager
from tqdm.auto import tqdm
from refined_package.inference.processor import Refined
from refined_package.doc_preprocessing.wikidata_mapper import WikidataMapper
from refined_package.utilities.general_utils import get_logger

LOG = get_logger(__name__)


def process_annotated_document(
        refined: Refined,
        doc: Doc,
        el: bool = False,
        ed_threshold: float = 0.0,
        force_prediction: bool = False,
        apply_class_check: bool = False,
        filter_nil: bool = False,  # filter_nil is False for our papers results as this is consistent with previous
        # work. But `filter_nil=False` unfairly penalises predicting new (or unlabelled) entities.
        return_special_spans: bool = False  # only set to True if the dataset has special spans (e.g. dates)
) -> Metrics:
    if force_prediction:
        assert ed_threshold == 0.0, "ed_threshold must be set to 0 to force predictions"

    # optionally filter NIL gold spans
    # nil_spans is a set of mention spans that are annotated as mentions in the dataset but are not linked to a KB
    # many nil_spans in public datasets should have been linked to an entity but due to the annotation creation
    # method many entity were missed. Furthermore, when some datasets were built the correct entity
    # did not exist in the KB at the time but do exist now. This means models are unfairly penalized for predicting
    # entities for nil_spans.
    nil_spans = set()
    if doc.md_spans is not None:
        for span in doc.md_spans:
            # gold_entity id will be added to md_spans when md_spans overlaps withs spans in merge_spans() method
            if span.gold_entity is None or span.gold_entity.wikidata_entity_id is None:
                nil_spans.add((span.text, span.start))

    predicted_spans = refined.process_text(
        text=doc.text,
        spans=doc.spans if not el else None,
        apply_class_check=apply_class_check,
        prune_ner_types=False,
        return_special_spans=return_special_spans  # only set to True if the dataset has special spans (e.g. dates)
    )

    num_cands = []
    gold_entity_spans = set()
    gold_entity_types = dict()
    gold_entity_ranks = []
    for span in doc.spans:
        if not span.is_eval:
            continue
        
        if (span.gold_entity is None or span.gold_entity.wikidata_entity_id is None
            # only include entity spans that have been annotated as an entity in a KB
                or span.gold_entity.wikidata_entity_id == "Q0"):
            continue

        gold_entity_spans.add((span.text, span.start, span.gold_entity.wikidata_entity_id))
        gold_entity_types[(span.text, span.start)] = set(
            refined.preprocessor.index_to_class[class_idx] for class_idx in refined.preprocessor.get_classes_idx_for_qcode_batch([span.gold_entity.wikidata_entity_id])[0].tolist() if class_idx != 0
        )
        num_cands.append(len([qcode for qcode, _ in span.candidate_entities if qcode != "Q0"]))
        if span.gold_entity.wikidata_entity_id in {qcode for qcode, _ in span.candidate_entities}:
            gold_entity_ranks.append([qcode for qcode, _ in span.candidate_entities].index(span.gold_entity.wikidata_entity_id))
        else:
            gold_entity_ranks.append(None)

    pred_entity_scores = dict()
    pred_entity_spans = set()
    pred_entity_spans_2 = set()
    pred_topk_entity_spans = dict()
    pred_topk_entity_spans_2 = dict()
    pred_all_entity_spans = dict()
    pred_all_entity_scores = dict()
    pred_entity_types = dict()
    sorted_pred_entity_types = dict()
    for span in predicted_spans:
        if not span.is_eval:
            continue
        # skip dates and numbers, only consider entities that are linked to a KB
        # pred_entity_spans is used for linkable mentions only
        if span.coarse_type != "MENTION":
            continue
        # Entity prediction using combined scores
        if (
                span.predicted_entity.wikidata_entity_id is None
                or span.entity_linking_model_confidence_score < ed_threshold
                or span.predicted_entity.wikidata_entity_id == 'Q-1'
        ):
            qcode = "Q0"
        else:
            qcode = span.predicted_entity.wikidata_entity_id
        if force_prediction and qcode == "Q0":
            if len(span.top_k_predicted_entities) >= 2:
                qcode = span.top_k_predicted_entities[1][0].wikidata_entity_id
        pred_entity_spans.add((span.text, span.start, qcode))
        pred_entity_scores[(span.text, span.start)] = {"qcode": qcode, "score": span.entity_linking_model_confidence_score}
        pred_topk_entity_spans[(span.text, span.start)] = [entity.wikidata_entity_id for entity, _ in span.top_k_predicted_entities]
        pred_all_entity_spans[(span.text, span.start)] = [{"qcode": entity.wikidata_entity_id, "score": score} for entity, score in span.all_predicted_entities]
        pred_all_entity_scores[(span.text, span.start)] = {entity.wikidata_entity_id: score for entity, score in span.top_k_predicted_entities}
        pred_all_entity_spans[(span.text, span.start)] = [{"qcode": entity.wikidata_entity_id, "score": score} for entity, score in span.all_predicted_entities]
        # Entity prediction using entity description scores
        if (
                span.predicted_entity_2.wikidata_entity_id is None
                or span.entity_linking_model_confidence_score_2 < ed_threshold
                or span.predicted_entity_2.wikidata_entity_id == 'Q-1'
        ):
            qcode = "Q0"
        else:
            qcode = span.predicted_entity_2.wikidata_entity_id
        if force_prediction and qcode == "Q0":
            if len(span.top_k_predicted_entities_2) >= 2:
                qcode = span.top_k_predicted_entities_2[1][0].wikidata_entity_id
        pred_entity_spans_2.add((span.text, span.start, qcode))
        pred_topk_entity_spans_2[(span.text, span.start)] = [entity.wikidata_entity_id for entity, _ in span.top_k_predicted_entities_2]

        pred_entity_types[(span.text, span.start)] = set(
            class_id for class_id, class_label, conf in span.predicted_entity_types
        )
        sorted_pred_entity_types[(span.text, span.start)] = [
            class_id for class_id, class_label, conf in span.sorted_predicted_entity_types
        ]

    tp_et = []
    fp_et = []
    fn_et = []
    fp_errors_et = []
    fn_errors_et = []
    for (text, start), gold_class_ids in gold_entity_types.items():
        if len(gold_class_ids) > 0 and (text, start) in pred_entity_types:
            pred_class_ids = pred_entity_types[(text, start)]
            tp_et.append(len(pred_class_ids & gold_class_ids))
            fp_et.append(len(pred_class_ids - gold_class_ids))
            fn_et.append(len(gold_class_ids - pred_class_ids))
            if len(pred_class_ids - gold_class_ids) > 0:
                fp_errors_et.append(pred_class_ids - gold_class_ids)
            if len(gold_class_ids - pred_class_ids) > 0:
                fn_errors_et.append(gold_class_ids - pred_class_ids)
    et_metrics = MultilabelClassificationMetrics(
        tp=tp_et,
        fp=fp_et,
        fn=fn_et,
    )

    et_metrics_k = {}
    for k in [3, 5, 10, 20]:
        tp_et = []
        fp_et = []
        fn_et = []
        for (text, start), gold_class_ids in gold_entity_types.items():
            if len(gold_class_ids) > 0 and (text, start) in sorted_pred_entity_types:
                k_pred_class_ids = set(sorted_pred_entity_types[(text, start)][:k])
                tp_et.append(len(k_pred_class_ids & gold_class_ids))
                fp_et.append(len(k_pred_class_ids - gold_class_ids))
                fn_et.append(len(gold_class_ids - k_pred_class_ids))
        et_metrics_k[k] = MultilabelClassificationMetrics(
            tp=tp_et,
            fp=fp_et,
            fn=fn_et,
        )

    pred_entity_spans = {(text, start, qcode) for text, start, qcode in pred_entity_spans if qcode != "Q0"}
    pred_entity_spans_2 = {(text, start, qcode) for text, start, qcode in pred_entity_spans_2 if qcode != "Q0"}
    if filter_nil:
        # filters model predictions that align with NIL spans in the dataset. See above for more information.
        # Note that this `Doc.md_spans` must include spans with wikidata_entity_id set to None,
        # so the data reader must not filter them out for this argument to work.
        pred_entity_spans = {
            (text, start, qcode)
            for text, start, qcode in pred_entity_spans
            if (text, start) not in nil_spans
        }
        pred_entity_spans_2 = {
            (text, start, qcode)
            for text, start, qcode in pred_entity_spans_2
            if (text, start) not in nil_spans
        }

    num_gold_spans = len(gold_entity_spans)
    tp_el = len(pred_entity_spans & gold_entity_spans)
    fp_el = len(pred_entity_spans - gold_entity_spans)
    fn_el = len(gold_entity_spans - pred_entity_spans)
    el_metrics = ClassificationMetrics(
        tp=tp_el,
        fp=fp_el,
        fn=fn_el,
        num_samples=num_gold_spans,
    )
    tp_el = len(pred_entity_spans_2 & gold_entity_spans)
    fp_el = len(pred_entity_spans_2 - gold_entity_spans)
    fn_el = len(gold_entity_spans - pred_entity_spans_2)
    el_metrics_2 = ClassificationMetrics(
        tp=tp_el,
        fp=fp_el,
        fn=fn_el,
        num_samples=num_gold_spans,
    )

    entity_hit_indices = []
    entity_hit_indices_2 = []
    for text, start, gold_qcode in gold_entity_spans:
        if (text, start) in pred_topk_entity_spans:
            hit_index = None
            if gold_qcode in pred_topk_entity_spans[(text, start)]:
                hit_index = pred_topk_entity_spans[(text, start)].index(gold_qcode)
            entity_hit_indices.append(hit_index)
        if (text, start) in pred_topk_entity_spans_2:
            hit_index = None
            if gold_qcode in pred_topk_entity_spans_2[(text, start)]:
                hit_index = pred_topk_entity_spans_2[(text, start)].index(gold_qcode)
            entity_hit_indices_2.append(hit_index)

    # ignore which entity is linked to (consider just the mention detection (NER) prediction)
    pred_spans_md = {(span.text, span.start, span.coarse_type) for span in predicted_spans}
    gold_spans_md = {(span.text, span.start, span.coarse_type) for span in doc.md_spans
                     if return_special_spans or span.coarse_type == "MENTION"}
    tp_md = len(pred_spans_md & gold_spans_md)
    fp_md = len(pred_spans_md - gold_spans_md)
    fn_md = len(gold_spans_md - pred_spans_md)
    md_metrics = ClassificationMetrics(
        tp=tp_md,
        fp=fp_md,
        fn=fn_md,
        num_samples=len(gold_spans_md),
    )

    fp_errors_et = sorted(list(fp_errors_et), key=lambda x: len(x))[:5]
    fn_errors_et = sorted(list(fn_errors_et), key=lambda x: len(x))[:5]

    fp_errors_el = sorted(list(pred_entity_spans - gold_entity_spans), key=lambda x: x[1])[:5]
    fn_errors_el = sorted(list(gold_entity_spans - pred_entity_spans), key=lambda x: x[1])[:5]

    fp_errors_md = sorted(list(pred_spans_md - gold_spans_md), key=lambda x: x[1])[:5]
    fn_errors_md = sorted(list(gold_spans_md - pred_spans_md), key=lambda x: x[1])[:5]

    answers = [{
        "text": doc.text,
        "answers": [
            {
                "text": text, 
                "start": start, 
                "gold_qcode": gold_qcode, 
                "pred_qcode": pred_entity_scores[(text, start)]["qcode"], 
                "pred_score": pred_entity_scores[(text, start)]["score"],
                "cand_scores": pred_all_entity_spans[(text, start)],
                "mse": 1.0 - pred_all_entity_scores[(text, start)][gold_qcode] if gold_qcode in pred_all_entity_scores[(text, start)] else 1.0
            } 
            for (text, start, gold_qcode) in gold_entity_spans if (text, start) in pred_entity_scores]
    }]

    metrics = Metrics(
        el=el,
        num_gold_spans=num_gold_spans,
        entity_hit_indices=entity_hit_indices,
        entity_hit_indices_2=entity_hit_indices_2,
        et_metrics=et_metrics,
        et_metrics_3=et_metrics_k[3],
        et_metrics_5=et_metrics_k[5],
        et_metrics_10=et_metrics_k[10],
        et_metrics_20=et_metrics_k[20],
        el_metrics=el_metrics,
        el_metrics_2=el_metrics_2,
        md_metrics=md_metrics,
        num_cands=num_cands,
        gold_entity_ranks=gold_entity_ranks,
        num_docs=1,
        example_errors_et=[{'doc_title': doc.text[:20], 'fp_errors_et': fp_errors_et, 'fn_errors_et': fn_errors_et}],
        example_errors_el=[{'doc_title': doc.text[:20], 'fp_errors_el': fp_errors_el, 'fn_errors_el': fn_errors_el}],
        example_errors_md=[{'doc_title': doc.text[:20], 'fp_errors_md': fp_errors_md, 'fn_errors_md': fn_errors_md}],
        answers=answers,
    )
    return metrics


def evaluate_on_docs(
        refined,
        docs: Iterable[Doc],
        progress_bar: bool = True,
        dataset_name: str = "dataset",
        ed_threshold: float = 0.0,
        apply_class_check: bool = False,
        el: bool = False,
        sample_size: Optional[int] = None,
        filter_nil_spans: bool = False,
        return_special_spans: bool = False
):
    overall_metrics = Metrics.zeros(el=el)
    for doc_idx, doc in tqdm(
            enumerate(list(docs)), disable=not progress_bar, desc=f"Evaluating on {dataset_name}"
    ):
        doc_metrics = process_annotated_document(
            refined=refined,
            doc=doc,
            force_prediction=False,
            ed_threshold=ed_threshold,
            apply_class_check=apply_class_check,
            el=el,
            filter_nil=filter_nil_spans,
            return_special_spans=return_special_spans
        )
        overall_metrics += doc_metrics
        if sample_size is not None and doc_idx > sample_size:
            break
    return overall_metrics


def eval_all(
        refined,
        data_dir: Optional[str] = None,
        datasets_dir: Optional[str] = None,
        additional_data_dir: Optional[str] = None,
        include_spans: bool = True,
        filter_not_in_kb: bool = True,
        ed_threshold: float = 0.15,
        el: bool = False,
        download: bool = True,
        apply_class_check: bool = False,
        filter_nil_spans: bool = False
):
    datasets = get_datasets_obj(preprocessor=refined.preprocessor,
                                data_dir=data_dir,
                                datasets_dir=datasets_dir,
                                additional_data_dir=additional_data_dir,
                                download=download)
    dataset_name_to_docs = get_standard_datasets(datasets, el, filter_not_in_kb, include_spans)
    return evaluate_on_datasets(refined=refined,
                                dataset_name_to_docs=dataset_name_to_docs,
                                el=el,
                                apply_class_check=apply_class_check,
                                ed_threshold=ed_threshold,
                                filter_nil_spans=filter_nil_spans
                                )


def get_standard_datasets(datasets: Datasets,
                          el: bool,
                          filter_not_in_kb: bool = True,
                          include_spans: bool = True) -> Dict[str, Iterable[Doc]]:
    if not el:
        dataset_name_to_docs = {
            "AIDA": datasets.get_aida_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "MSNBC": datasets.get_msnbc_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "AQUAINT": datasets.get_aquaint_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "ACE2004": datasets.get_ace2004_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "CWEB": datasets.get_cweb_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "WIKI": datasets.get_wiki_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
        }
    else:
        dataset_name_to_docs = {
            "AIDA": datasets.get_aida_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
            "MSNBC": datasets.get_msnbc_docs(
                split="test",
                include_gold_label=True,
                filter_not_in_kb=filter_not_in_kb,
                include_spans=include_spans,
            ),
        }
    return dataset_name_to_docs


def evaluate_on_datasets(refined: Refined,
                         dataset_name_to_docs: Dict[str, Iterable[Doc]],
                         el: bool,
                         apply_class_check: bool = False,
                         ed_threshold: float = 0.15,
                         return_special_spans: bool = False,  # only set to True if the dataset has special spans (
                         # e.g. dates)
                         filter_nil_spans: bool = False
                         ):
    dataset_name_to_metrics = dict()
    for dataset_name, dataset_docs in dataset_name_to_docs.items():
        metrics = evaluate_on_docs(
            refined=refined,
            docs=dataset_docs,
            dataset_name=dataset_name,
            ed_threshold=ed_threshold,
            el=el,
            apply_class_check=apply_class_check,
            filter_nil_spans=filter_nil_spans,  # filter model predictions that align with md_spans that have no
            # gold_entity_id but are annotated/labelled as mentions in the dataset.
            return_special_spans=return_special_spans,
        )
        dataset_name_to_metrics[dataset_name] = metrics
        print("*****************************\n\n")
        print(f"Dataset name: {dataset_name}")
        print(metrics.get_summary())
        print("*****************************\n\n")
    return dataset_name_to_metrics


def get_datasets_obj(preprocessor: Preprocessor,
                     download: bool = True,
                     data_dir: Optional[str] = None,
                     datasets_dir: Optional[str] = None,
                     additional_data_dir: Optional[str] = None,
                     ) -> Datasets:
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser('~'), '.cache', 'refined')
    if datasets_dir is None:
        datasets_dir = os.path.join(data_dir, 'datasets')
    if additional_data_dir is None:
        additional_data_dir = os.path.join(data_dir, 'additional_data')

    resource_manager = ResourceManager(S3Manager(),
                                       data_dir=datasets_dir,
                                       datasets_dir=datasets_dir,
                                       additional_data_dir=additional_data_dir,
                                       entity_set=None,
                                       model_name=None
                                       )
    if download:
        resource_manager.download_datasets_if_needed()
        resource_manager.download_additional_files_if_needed()

    wikidata_mapper = WikidataMapper(resource_manager=resource_manager)
    return Datasets(preprocessor=preprocessor,
                    resource_manager=resource_manager,
                    wikidata_mapper=wikidata_mapper)


def evaluate(evaluation_dataset_name_to_docs: Dict[str, Iterable[Doc]],
             refined: Refined,
             ed_threshold: float = 0.15,
             el: bool = True,
             ed: bool = True,
             print_errors: bool = True,
             return_special_spans: bool = True) -> Dict[str, Metrics]:
    dataset_name_to_metrics = dict()
    if el:
        LOG.info("Running entity linking evaluation")
        el_results = evaluate_on_datasets(
            refined=refined,
            dataset_name_to_docs=evaluation_dataset_name_to_docs,
            el=True,
            ed_threshold=ed_threshold,
            return_special_spans=return_special_spans,
            filter_nil_spans=True  # makes EL evaluation more fair
        )
        for dataset_name, metrics in el_results.items():
            dataset_name_to_metrics[f"{dataset_name}-EL"] = metrics
            if print_errors:
                LOG.info("Printing EL errors")
                pprint(metrics.example_errors_el[:5])
                LOG.info("Printing ET errors")
                pprint(metrics.example_errors_et[:5])
                LOG.info("Printing MD errors")
                pprint(metrics.example_errors_md[:5])

    if ed:
        LOG.info("Running entity disambiguation evaluation")
        ed_results = evaluate_on_datasets(
            refined=refined,
            dataset_name_to_docs=evaluation_dataset_name_to_docs,
            el=False,
            ed_threshold=ed_threshold,
            return_special_spans=False
        )
        for dataset_name, metrics in ed_results.items():
            dataset_name_to_metrics[f"{dataset_name}-ED"] = metrics
            if print_errors:
                LOG.info("Printing ED errors")
                pprint(metrics.example_errors_el[:5])
                LOG.info("Printing ET errors")
                pprint(metrics.example_errors_et[:5])

    return dataset_name_to_metrics
