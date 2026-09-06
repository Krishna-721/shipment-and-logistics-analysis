"""Baseline models for benchmarking."""

import pandas as pd
import numpy as np
from typing import Dict, Any

from src.config import QUEUE_THRESHOLD


class QueueOnlyBaseline:
    """
    Simple baseline that predicts congestion based on queue pressure.
    
    Rule: Predict congestion if queue_length > QUEUE_THRESHOLD * total_berths
    """
    
    def __init__(self, threshold: float = QUEUE_THRESHOLD):
        self.threshold = threshold
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'QueueOnlyBaseline':
        """Fit the baseline (no actual fitting needed for rule-based model)."""
        if "queue_length" not in X.columns or "total_berths" not in X.columns:
            raise ValueError("QueueOnlyBaseline requires 'queue_length' and 'total_berths' columns")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict congestion using queue pressure rule."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        queue_pressure = X["queue_length"] / X["total_berths"]
        predictions = (queue_pressure > self.threshold).astype(int)
        return predictions.values
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return prediction probabilities.
        
        For rule-based model:
        - If queue > threshold: [0.01, 0.99] (high confidence congestion)
        - If queue <= threshold: [0.99, 0.01] (high confidence no congestion)
        """
        predictions = self.predict(X)
        
        # Create probability array
        probas = np.zeros((len(predictions), 2))
        probas[predictions == 0] = [0.99, 0.01]  # High confidence no congestion
        probas[predictions == 1] = [0.01, 0.99]  # High confidence congestion
        
        return probas
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {"threshold": self.threshold}
    
    def __repr__(self) -> str:
        return f"QueueOnlyBaseline(threshold={self.threshold})"
