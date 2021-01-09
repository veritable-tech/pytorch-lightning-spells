class EMATracker:
    """Keeps the exponential moving average for a single series.

    Parameters
    ----------
    alpha : float, optional
        the weight of the new value, by default 0.05
    """       

    def __init__(self, alpha: float = 0.05):     
        super().__init__()
        self.alpha = alpha
        self._value = None

    def update(self, new_value):
        new_value = new_value.detach()
        if self._value is None:
            self._value = new_value
        else:
            self._value = (
                new_value * self.alpha +
                self._value * (1-self.alpha)
            )

    @property
    def value(self):
        return self._value