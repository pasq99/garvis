"""Shared, dependency-free observation contract for the Pareto ranker."""

ROUTE_FEATURES = (
    "path_ratio_scaled",
    "horizontal_error_scaled",
    "bearing_error_scaled",
)
PADDING_VALUE = -1.0
MAX_PARETO_SIZE = 100
