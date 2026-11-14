"""Bayes-TS real-time recommender for the RawTree Datathon FME 2026.

The package is import-agnostic: ``starter.recommender`` is the operational
entry point, while the other modules can be used independently by the offline
builder and the unit tests.
"""

__all__ = ["config", "rawtree", "text", "state", "priors", "features",
           "scoring", "workers", "recommender"]
