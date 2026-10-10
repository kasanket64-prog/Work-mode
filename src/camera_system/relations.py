"""Thin adapter around the pinned upstream public API. No copied model code."""

from dataclasses import asdict
from math import isfinite
from typing import Protocol

import numpy as np
from PIL import Image

from .config import Settings
from .contracts import Region, Relation


def screen_relations(relations, regions):
    """Conservative class checks; unknown labels remain open vocabulary candidates."""
    labels = {r.id: (r.label or "").strip().lower() for r in regions}
    vehicles = {"car", "bus", "truck", "train", "airplane", "boat"}
    nonwearable = vehicles | {
        "bicycle",
        "motorcycle",
        "chair",
        "bench",
        "couch",
        "bed",
        "dining table",
        "refrigerator",
        "toilet",
        "person",
    }
    accepted, rejected = [], []
    # ponytail: explicit class exclusions only; extend using reviewed false positives, not guesses.
    for relation in relations:
        subject, obj = labels.get(relation.subject_id), labels.get(relation.object_id)
        predicate = relation.predicate.strip().lower()
        reason = None
        if subject is None or obj is None:
            reason = "unknown_region"
        elif not isfinite(relation.score) or not 0 <= relation.score <= 1:
            reason = "invalid_score"
        elif relation.subject_id == relation.object_id:
            reason = "self_relation"
        elif predicate == "wearing" and obj in nonwearable:
            reason = "nonwearable_object_class"
        elif (
            predicate in {"wearing", "holding", "sitting on", "standing on"} and subject in vehicles
        ):
            reason = "incompatible_subject_class"
        if reason:
            rejected.append(
                {
                    **asdict(relation),
                    "score": relation.score if isfinite(relation.score) else None,
                    "reason": reason,
                }
            )
        else:
            accepted.append(relation)
    return accepted, rejected


class RelationPredictor(Protocol):
    def predict(self, image: Image.Image, regions: list[Region]) -> list[Relation]: ...


class RelateAnythingPredictor:
    def __init__(self, settings: Settings):
        try:
            import torch
            from relsgg import RelateAnything
        except ImportError as exc:
            raise RuntimeError(
                'Install inference dependencies: pip install -e ".[inference]"'
            ) from exc
        self.settings = settings
        self.device = settings.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; set device = 'cpu'")
        self.model = RelateAnything.from_pretrained(
            settings.model_id,
            revision=settings.model_revision,
            predicates=list(settings.vocabulary),
            device=self.device,
        )

    def predict(self, image: Image.Image, regions: list[Region]) -> list[Relation]:
        # Reject instead of letting upstream truncate/reindex silently. IDs must stay stable.
        if len(regions) > self.settings.max_regions:
            raise ValueError(f"at most {self.settings.max_regions} regions are supported")
        triplets = self.model.predict(
            image,
            np.asarray([r.box for r in regions], dtype=np.float32),
            box_labels=[r.label or r.id for r in regions],
            topk=self.settings.topk,
            max_boxes=self.settings.max_regions,
        )
        candidates = [
            Relation(
                regions[t.subject_idx].id, t.predicate, regions[t.object_idx].id, float(t.score)
            )
            for t in triplets
            if float(t.score) >= self.settings.min_score
        ]
        accepted, self.rejected_relations = screen_relations(candidates, regions)
        return accepted
