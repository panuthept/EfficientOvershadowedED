import os
import json
import numpy as np
from dataclasses import dataclass, field

# add weak match for QA and EL
from typing import List, Any


def cal_precision(tp: int, fp: int) -> float:
    return tp / (tp + fp + 1e-8 * 1.0)

def cal_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn + 1e-8 * 1.0)

def cal_f1(tp: int, fp: int, fn: int) -> float:
    p = cal_precision(tp, fp)
    r = cal_recall(tp, fn)
    return 2.0 * p * r / (p + r + 1e-8)

def cal_accuracy(tp: int, num_samples: int) -> float:
    return 1.0 * tp / (num_samples + 1e-8)


@dataclass
class ClassificationMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    num_samples: int = 0

    def __add__(self, other: 'ClassificationMetrics'):
        return ClassificationMetrics(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            num_samples=self.num_samples + other.num_samples
        )
    
    def get_recall(self):
        return cal_recall(self.tp, self.fn)
    
    def get_precision(self):
        return cal_precision(self.tp, self.fp)
    
    def get_f1(self):
        return cal_f1(self.tp, self.fp, self.fn)
    
    def get_accuracy(self):
        return cal_accuracy(self.tp, self.num_samples)
    
    @classmethod
    def zeros(cls):
        return ClassificationMetrics(tp=0, fp=0, fn=0, num_samples=0)
    

@dataclass
class MultilabelClassificationMetrics:
    tp: List[int] = field(default_factory=list)
    fp: List[int] = field(default_factory=list)
    fn: List[int] = field(default_factory=list)

    def __add__(self, other: 'MultilabelClassificationMetrics'):
        return MultilabelClassificationMetrics(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )
    
    def get_recall(self):
        return np.mean([cal_recall(tp, fn) for tp, fn in zip(self.tp, self.fn)])
    
    def get_precision(self):
        return np.mean([cal_precision(tp, fp) for tp, fp in zip(self.tp, self.fp)])
    
    def get_f1(self):
        return np.mean([cal_f1(tp, fp, fn) for tp, fp, fn in zip(self.tp, self.fp, self.fn)])

    def get_hit(self):
        return np.mean([1.0 if tp > 0 else 0.0 for tp in self.tp])
    
    @classmethod
    def zeros(cls):
        return MultilabelClassificationMetrics(
            tp=field(default_factory=list), 
            fp=field(default_factory=list), 
            fn=field(default_factory=list), 
        )


@dataclass
class Metrics:
    el: bool  # flags whether the metrics are for entity linking (EL) or entity disambiguation (ED)
    num_gold_spans: int = 0
    entity_hit_indices: List[int] = field(default_factory=list)
    entity_hit_indices_2: List[int] = field(default_factory=list)
    et_metrics: MultilabelClassificationMetrics = field(default_factory=MultilabelClassificationMetrics)
    et_metrics_3: MultilabelClassificationMetrics = field(default_factory=MultilabelClassificationMetrics)
    et_metrics_5: MultilabelClassificationMetrics = field(default_factory=MultilabelClassificationMetrics)
    et_metrics_10: MultilabelClassificationMetrics = field(default_factory=MultilabelClassificationMetrics)
    et_metrics_20: MultilabelClassificationMetrics = field(default_factory=MultilabelClassificationMetrics)
    el_metrics: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    el_metrics_2: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    md_metrics: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    num_cands: List[int] = field(default_factory=list)
    gold_entity_ranks: List[int] = field(default_factory=list)
    num_docs: int = 0
    example_errors_et: List[Any] = field(default_factory=list)
    example_errors_el: List[Any] = field(default_factory=list)
    example_errors_md: List[Any] = field(default_factory=list)
    answers: List[Any] = field(default_factory=list)

    def __add__(self, other: 'Metrics'):
        return Metrics(
            el=self.el,
            num_gold_spans=self.num_gold_spans + other.num_gold_spans,
            entity_hit_indices=self.entity_hit_indices + other.entity_hit_indices,
            entity_hit_indices_2=self.entity_hit_indices_2 + other.entity_hit_indices_2,
            et_metrics=self.et_metrics + other.et_metrics,
            et_metrics_3=self.et_metrics_3 + other.et_metrics_3,
            et_metrics_5=self.et_metrics_5 + other.et_metrics_5,
            et_metrics_10=self.et_metrics_10 + other.et_metrics_10,
            et_metrics_20=self.et_metrics_20 + other.et_metrics_20,
            el_metrics=self.el_metrics + other.el_metrics,
            el_metrics_2=self.el_metrics_2 + other.el_metrics_2,
            md_metrics=self.md_metrics + other.md_metrics,
            num_cands=self.num_cands + other.num_cands,
            gold_entity_ranks=self.gold_entity_ranks + other.gold_entity_ranks,
            num_docs=self.num_docs + other.num_docs,
            example_errors_et=self.example_errors_et + other.example_errors_et,
            example_errors_el=self.example_errors_el + other.example_errors_el,
            example_errors_md=self.example_errors_md + other.example_errors_md,
            answers=self.answers + other.answers
        )
    
    def save_answers(self, save_path: str):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, "w") as f:
            for answer in self.answers:
                f.write(f"{json.dumps(answer)}\n")

    def get_summary(self):
        # Mention span and entity must correctly match
        mrr_el = self.get_mrr_el()
        f1_el = self.el_metrics.get_f1()
        r_el = self.el_metrics.get_recall()
        p_el = self.el_metrics.get_precision()
        mrr_el_2 = self.get_mrr_el_2()
        f1_el_2 = self.el_metrics_2.get_f1()
        r_el_2 = self.el_metrics_2.get_recall()
        p_el_2 = self.el_metrics_2.get_precision()
        result = f"\n********** Entity Prediction ***********\n" \
                 f"MRR: {round(mrr_el * 100, 1)} ({round(mrr_el_2 * 100, 1)})\n" \
                 f"F1: {round(f1_el * 100, 1)} ({round(f1_el_2 * 100, 1)})\n" \
                 f"Recall: {round(r_el * 100, 1)} ({round(r_el_2 * 100, 1)})\n" \
                 f"Precision: {round(p_el * 100, 1)} ({round(p_el_2 * 100, 1)})\n" \
                 f"num_gold_spans: {self.num_gold_spans}\n" \
                 f"****************************************\n"
        cand_mrr = self.get_cand_mrr()
        cand_recall = self.get_cand_recall()
        cand_precision = self.get_cand_precision()
        result += f"********** Candidate Generation ********\n" \
                  f"MRR: {round(cand_mrr * 100, 1)}\n" \
                  f"Recall: {round(cand_recall * 100, 1)}\n" \
                  f"Precision: {round(cand_precision * 100, 1)}\n" \
                  f"****************************************\n"
        if not self.el:
            f1_et = self.et_metrics.get_f1()
            r_et = self.et_metrics.get_recall()
            p_et = self.et_metrics.get_precision()
            hit_3_et = self.et_metrics_3.get_hit()
            hit_5_et = self.et_metrics_5.get_hit()
            hit_10_et = self.et_metrics_10.get_hit()
            hit_20_et = self.et_metrics_20.get_hit()
            r_3_et = self.et_metrics_3.get_recall()
            r_5_et = self.et_metrics_5.get_recall()
            r_10_et = self.et_metrics_10.get_recall()
            r_20_et = self.et_metrics_20.get_recall()
            p_3_et = self.et_metrics_3.get_precision()
            p_5_et = self.et_metrics_5.get_precision()
            p_10_et = self.et_metrics_10.get_precision()
            p_20_et = self.et_metrics_20.get_precision()
            result += f"************** Entity Type *************\n" \
                      f"F1: {round(f1_et * 100, 1)}\n" \
                      f"Recall: {round(r_et * 100, 1)}\n" \
                      f"Precision: {round(p_et * 100, 1)}\n" \
                      f"Hit@3/5/10/20: {round(hit_3_et * 100, 1)}/{round(hit_5_et * 100, 1)}/{round(hit_10_et * 100, 1)}/{round(hit_20_et * 100, 1)}\n" \
                      f"Recall@3/5/10/20: {round(r_3_et * 100, 1)}/{round(r_5_et * 100, 1)}/{round(r_10_et * 100, 1)}/{round(r_20_et * 100, 1)}\n" \
                      f"Precision@3/5/10/20: {round(p_3_et * 100, 1)}/{round(p_5_et * 100, 1)}/{round(p_10_et * 100, 1)}/{round(p_20_et * 100, 1)}\n" \
                      f"****************************************\n"
        if self.el:
            f1_md = self.md_metrics.get_f1()
            r_md = self.md_metrics.get_recall()
            p_md = self.md_metrics.get_precision()
            result += f"*********** Mention Detection **********\n" \
                      f"F1: {round(f1_md * 100, 1)}\n" \
                      f"Recall: {round(r_md * 100, 1)}\n" \
                      f"Precision: {round(p_md * 100, 1)}\n" \
                      f"****************************************\n"
        return result
    
    def get_f1(self):
        return self.el_metrics.get_f1()

    def get_mrr_el(self):
        mrr = []
        for hit_index in self.entity_hit_indices:
            if hit_index is None:
                mrr.append(0.0)
            else:
                mrr.append(1.0 / (hit_index + 1))
        return np.mean(mrr)
    
    def get_mrr_el_2(self):
        mrr = []
        for hit_index in self.entity_hit_indices_2:
            if hit_index is None:
                mrr.append(0.0)
            else:
                mrr.append(1.0 / (hit_index + 1))
        return np.mean(mrr)

    def get_cand_recall(self):
        return np.mean([1.0 if rank is not None else 0.0 for rank in self.gold_entity_ranks])
    
    def get_cand_precision(self):
        return np.mean([1.0 / (num_cand + 1e-8) if rank is not None else 0.0 for rank, num_cand in zip(self.gold_entity_ranks, self.num_cands)])

    def get_cand_mrr(self):
        return np.mean([1.0 / (rank + 1) if rank is not None else 0.0 for rank in self.gold_entity_ranks])

    @classmethod
    def zeros(cls, el: bool):
        return Metrics(el=el)
