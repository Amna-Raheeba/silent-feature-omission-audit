"""Silent Feature Omission Audit.

Scores post-hoc explainers on the features they fail to mention --
features the data-generating process actually used -- against known
structural ground truth, per usage type, swept over planted redundancy.
"""

__version__ = "0.1.0"
