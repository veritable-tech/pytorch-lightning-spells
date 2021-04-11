import warnings
from typing import Optional, Any

import torch
import numpy as np
from scipy.stats import spearmanr
from pytorch_lightning.metrics import Metric
from pytorch_lightning.utilities import rank_zero_warn
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning


class GlobalMetric(Metric):
    def __init__(
        self,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            'This Metric will save all targets and predictions in buffer.'
            ' For large datasets this may lead to large memory footprint.'
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        self.preds.append(preds)
        self.target.append(target)


class AUC(GlobalMetric):
    """AUC: Area Under the ROC Curve


    **Binary mode**

    >>> auc = AUC()
    >>> auc(torch.tensor([0.3, 0.8, 0.2]), torch.tensor([0, 1, 0]))
    >>> auc(torch.tensor([0.3, 0.3, 0.9]), torch.tensor([1, 1, 0]))
    >>> round(auc.compute().item(), 2)
    0.56

    **Multi-class mode**

    This will use the first column as the negative case, and the rest collectively as the positive case.

    >>> auc = AUC()
    >>> auc(torch.tensor([[0.3, 0.8, 0.2], [0.2, 0.1, 0.1], [0.5, 0.1, 0.7]]).t(), torch.tensor([0, 1, 0]))
    >>> auc(torch.tensor([[0.3, 0.3, 0.8], [0.2, 0.6, 0.1], [0.5, 0.1, 0.1]]).t(), torch.tensor([1, 1, 0]))
    >>> round(auc.compute().item(), 2)
    0.39
    """
    def compute(self):
        target = torch.cat(self.target, dim=0).cpu().long().numpy()
        preds = torch.cat(self.preds, dim=0).float().cpu().numpy()
        if len(preds.shape) > 1:
            preds = 1 - preds[:, 0]
            target = (target != 0).astype(int)
        if len(np.unique(target)) == 1:
            return torch.tensor(0, device=self.preds[0].device)
        return torch.tensor(roc_auc_score(target, preds), device=self.preds[0].device)


class SpearmanCorrelation(GlobalMetric):
    def __init__(
        self,
        sigmoid: bool = False,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.sigmoid = sigmoid

    def compute(self):
        preds = torch.cat(self.preds, dim=0).float()
        if self.sigmoid:
            preds = torch.sigmoid(preds)
        preds = preds.cpu().numpy()
        target = torch.cat(self.target, dim=0).cpu().float().numpy()
        spearman_score = spearmanr(
            target, preds
        ).correlation
        if len(np.unique(target)) == 1:
            return torch.tensor(0, device=self.preds[0].device)
        return torch.tensor(spearman_score, device=self.preds[0].device)


class FBeta(GlobalMetric):
    """The F-beta score is the weighted harmonic mean of precision and recall

    **Binary mode**

    >>> fbeta = FBeta()
    >>> fbeta(torch.tensor([0.3, 0.8, 0.2]), torch.tensor([0, 1, 0]))
    >>> fbeta(torch.tensor([0.3, 0.3, 0.9]), torch.tensor([1, 1, 0]))
    >>> round(fbeta.compute().item(), 2)
    0.88

    **Multi-class mode**

    This will use the first column as the negative case, and the rest collectively as the positive case.

    >>> fbeta = FBeta()
    >>> fbeta(torch.tensor([[0.8, 0.3, 0.7], [0.1, 0.1, 0.1], [0.1, 0.6, 0.2]]).t(), torch.tensor([0, 1, 0]))
    >>> fbeta(torch.tensor([[0.3, 0.7, 0.8], [0.2, 0.2, 0.1], [0.5, 0.1, 0.1]]).t(), torch.tensor([1, 1, 0]))
    >>> round(fbeta.compute().item(), 4)
    0.9375
    """
    def __init__(
        self,
        step: float = 0.02,
        beta: int = 2,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.step = step
        self.beta = beta

    def find_best_fbeta_threshold(self, truth, probs):
        best, best_thres = 0, -1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            for thres in np.arange(self.step, 1, self.step):
                current = fbeta_score(
                    truth, (probs >= thres).astype("int8"),
                    beta=self.beta, average="binary")
                if current > best:
                    best = current
                    best_thres = thres
        return best, best_thres

    def compute(self):
        preds = torch.cat(self.preds, dim=0).float().cpu().numpy()
        target = torch.cat(self.target, dim=0).cpu().long().numpy()
        if len(preds.shape) > 1:
            preds = 1 - preds[:, 0]
            target = (target != 0).astype(int)
        if len(np.unique(target)) == 1:
            return torch.tensor(0, device=self.preds[0].device)
        best_fbeta, best_thres = self.find_best_fbeta_threshold(
            target, preds)
        return torch.tensor(best_fbeta, device=self.preds[0].device)
