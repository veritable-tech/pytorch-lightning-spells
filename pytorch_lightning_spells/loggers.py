from typing import Dict, Optional
from pytorch_lightning.loggers import Logger


class ScreenLogger(Logger):
    """A logger that prints metrics to the screen.

    Suitable in situation where you want to check the training progress directly in the console.
    """

    def __init__(self):
        super().__init__()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if any((key.startswith("val_") for key in metrics.keys())):
            print("")
        for key, val in metrics.items():
            if key.startswith("val_"):
                print(key, "%.4f" % val)

    def log_hyperparams(self, params):
        pass

    @property
    def experiment(self):
        return None

    @property
    def name(self):
        return "screen_logger"

    @property
    def version(self) -> int:
        return 0
