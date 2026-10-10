from camera_system.contracts import Region, Relation
from camera_system.relations import screen_relations


def test_screen_retains_valid_and_unknown_candidates_with_rejection_evidence():
    regions = [
        Region(label, (0, 0, 10, 10), label)
        for label in ("person", "car", "backpack", "bicycle", "custom garment")
    ]
    candidates = [
        Relation("person", "wearing", "car", 0.9),
        Relation("car", "standing on", "person", 0.8),
        Relation("person", "wearing", "backpack", 0.7),
        Relation("person", "holding", "bicycle", 0.6),
        Relation("person", "wearing", "custom garment", 0.5),
        Relation("car", "beside", "bicycle", 0.4),
        Relation("car", "beside", "car", 0.9),
        Relation("missing", "beside", "car", 0.9),
        Relation("person", "beside", "car", float("inf")),
    ]
    accepted, rejected = screen_relations(candidates, regions)
    assert accepted == candidates[2:6]
    assert [r["reason"] for r in rejected] == [
        "nonwearable_object_class",
        "incompatible_subject_class",
        "self_relation",
        "unknown_region",
        "invalid_score",
    ]
    assert rejected[0]["score"] == 0.9
