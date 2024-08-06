from dataclasses import dataclass


@dataclass
class Metric:
    map50: float = -1
    map: float = -1
    recall: float = -1
    precision: float = -1
    f1: float = -1
