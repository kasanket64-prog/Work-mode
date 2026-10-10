"""Strict TOML configuration; inference dependencies are loaded only on demand."""

import tomllib
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path

UPSTREAM_REVISION = "06766fdf56752ca535fc9b971fca99ce563676d0"


@dataclass(frozen=True)
class Settings:
    model_id: str = "maelic/relsgg-vits16plus"
    model_revision: str = "2db90096be5217bdc7a9003c042950f45723d105"
    device: str = "cpu"
    topk: int = 5
    max_regions: int = 60
    min_score: float = 0.0
    vocabulary: tuple[str, ...] = ("riding", "carrying a rider", "beside", "in front of")

    def __post_init__(self):
        if not isinstance(self.device, str) or self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda, or auto")
        for name in ("model_id", "model_revision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name, minimum in (("topk", 1), ("max_regions", 2)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            type(self.min_score) not in (int, float)
            or not isfinite(self.min_score)
            or not 0 <= self.min_score <= 1
        ):
            raise ValueError("min_score must be in [0, 1]")
        if (
            not isinstance(self.vocabulary, (tuple, list))
            or not self.vocabulary
            or any(not isinstance(v, str) or not v.strip() for v in self.vocabulary)
            or len(set(self.vocabulary)) != len(self.vocabulary)
        ):
            raise ValueError("vocabulary must contain unique nonempty strings")


def load_settings(path: Path) -> Settings:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = set(data) - {f.name for f in fields(Settings)}
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    return Settings(**data)
