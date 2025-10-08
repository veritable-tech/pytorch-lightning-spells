import warnings
from typing import Any, cast

import torch
import numpy as np
from torchmetrics import Metric
from scipy.stats import spearmanr
from pytorch_lightning.utilities import rank_zero_warn
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning


class GlobalMetric(Metric):
    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Any = None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

        rank_zero_warn(
            "This metric will save all targets and predictions in buffer. For large datasets this may lead to large memory footprint."
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        cast(list[torch.Tensor], cast(object, self.preds)).append(preds)
        cast(list[torch.Tensor], cast(object, self.targets)).append(target)


class AUC(GlobalMetric):
    """AUC: Area Under the ROC Curve


    **Binary mode**

    >>> auc = AUC()
    >>> _ = auc(torch.tensor([0.3, 0.8, 0.2]), torch.tensor([0, 1, 0]))
    >>> _ = auc(torch.tensor([0.3, 0.3, 0.9]), torch.tensor([1, 1, 0]))
    >>> round(auc.compute().item(), 2)
    0.56

    **Multi-class mode**

    This will use the first column as the negative case, and the rest collectively as the positive case.

    >>> auc = AUC()
    >>> _ = auc(torch.tensor([[0.3, 0.8, 0.2], [0.2, 0.1, 0.1], [0.5, 0.1, 0.7]]).t(), torch.tensor([0, 1, 0]))
    >>> _ = auc(torch.tensor([[0.3, 0.3, 0.8], [0.2, 0.6, 0.1], [0.5, 0.1, 0.1]]).t(), torch.tensor([1, 1, 0]))
    >>> round(auc.compute().item(), 2)
    0.39
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        message = (
            "This is a specialized metric that coalesces all classes except the first into a single positive class. "
            "Please use BinaryAUROC from torchmetrics if you have a binary classification problem: "
            "'from torchmetrics.classification import BinaryAUROC'."
        )
        rank_zero_warn(message)

    def compute(self):
        preds_list = cast(list[torch.Tensor], cast(object, self.preds))
        target_list = cast(list[torch.Tensor], cast(object, self.targets))
        targets_np = torch.cat(target_list, dim=0).cpu().long().numpy()
        preds_np = torch.nan_to_num(torch.cat(preds_list, dim=0).float().cpu()).numpy()
        if len(preds_np.shape) > 1:
            preds_np = 1 - preds_np[:, 0]
            targets_np = (targets_np != 0).astype(int)
        if len(np.unique(targets_np)) == 1:
            return torch.tensor(0, device=preds_list[0].device)
        return torch.tensor(roc_auc_score(targets_np, preds_np), device=preds_list[0].device)


class SpearmanCorrelation(GlobalMetric):
    def __init__(
        self,
        sigmoid: bool = False,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Any = None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.sigmoid = sigmoid

    def compute(self):
        preds_list = cast(list[torch.Tensor], cast(object, self.preds))
        target_list = cast(list[torch.Tensor], cast(object, self.targets))
        preds_tensor = torch.cat(preds_list, dim=0).float()
        if self.sigmoid:
            preds_tensor = torch.sigmoid(preds_tensor)
        preds_np = preds_tensor.cpu().numpy()
        targets_np = torch.cat(target_list, dim=0).cpu().float().numpy()
        spearman_score = spearmanr(targets_np, preds_np).correlation  # pyright: ignore[reportAttributeAccessIssue]
        if len(np.unique(targets_np)) == 1:
            return torch.tensor(0, device=preds_list[0].device)
        return torch.tensor(spearman_score, device=preds_list[0].device)


class FBeta(GlobalMetric):
    """The F-beta score is the weighted harmonic mean of precision and recall

    This is a specialized implementation.

    -

    **Binary mode**

    >>> fbeta = FBeta()
    >>> _ = fbeta(torch.tensor([0.3, 0.8, 0.2]), torch.tensor([0, 1, 0]))
    >>> _ = fbeta(torch.tensor([0.3, 0.3, 0.9]), torch.tensor([1, 1, 0]))
    >>> round(fbeta.compute().item(), 2)
    0.88

    **Multi-class mode**

    This will **use the first column as the negative case, and the rest collectively as the positive case**.

    Use MulticlassFBetaScore from torchmetrics for the true multiclass F-score metric.

    >>> fbeta = FBeta()
    >>> _ = fbeta(torch.tensor([[0.8, 0.3, 0.7], [0.1, 0.1, 0.1], [0.1, 0.6, 0.2]]).t(), torch.tensor([0, 1, 0]))
    >>> _ = fbeta(torch.tensor([[0.3, 0.7, 0.8], [0.2, 0.2, 0.1], [0.5, 0.1, 0.1]]).t(), torch.tensor([1, 1, 0]))
    >>> round(fbeta.compute().item(), 4)
    0.9375
    """

    def __init__(
        self,
        step: float = 0.02,
        beta: int = 2,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Any = None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.step = step
        self.beta = beta
        rank_zero_warn(
            (
                "This is a specialized metric that coalesces all classes except the first into the positive class and automatically optimizes the classification threshold. "
                "Please use FBeta from torchmetrics for general use cases."
            )
        )

    def find_best_fbeta_threshold(self, truth, probs):
        best, best_thres = 0, -1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            for thres in np.arange(self.step, 1, self.step):
                current = fbeta_score(
                    truth,
                    (probs >= thres).astype("int8"),
                    beta=self.beta,
                    average="binary",
                )
                if current > best:
                    best = current
                    best_thres = thres
        return best, best_thres

    def compute(self):
        preds_list = cast(list[torch.Tensor], cast(object, self.preds))
        target_list = cast(list[torch.Tensor], cast(object, self.targets))
        preds_np = torch.cat(preds_list, dim=0).float().cpu().numpy()
        targets_np = torch.cat(target_list, dim=0).cpu().long().numpy()
        if len(preds_np.shape) > 1:
            preds_np = 1 - preds_np[:, 0]
            targets_np = (targets_np != 0).astype(int)
        if len(np.unique(targets_np)) == 1:
            return torch.tensor(0, device=preds_list[0].device)
        best_fbeta, best_thres = self.find_best_fbeta_threshold(targets_np, preds_np)
        return torch.tensor(best_fbeta, device=preds_list[0].device)
