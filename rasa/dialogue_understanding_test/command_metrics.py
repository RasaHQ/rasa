from typing import Dict

from pydantic import BaseModel


class CommandMetrics(BaseModel):
    tp: int
    fp: int
    fn: int
    total_count: int

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Safely perform division, returning 0.0 if the denominator is zero."""
        return numerator / denominator if denominator > 0 else 0.0

    def get_precision(self) -> float:
        return self._safe_divide(self.tp, self.tp + self.fp)

    def get_recall(self) -> float:
        return self._safe_divide(self.tp, self.tp + self.fn)

    def get_f1_score(self) -> float:
        precision = self.get_precision()
        recall = self.get_recall()

        return self._safe_divide(2 * precision * recall, precision + recall)

    def as_dict(self) -> Dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.get_precision(),
            "recall": self.get_recall(),
            "f1_score": self.get_f1_score(),
            "total_count": self.total_count,
        }
