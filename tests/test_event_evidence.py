from camera_system.review import link_event_frames


def test_evidence_requires_both_tracks_and_interval_without_mutation():
    event = {
        "timestamp_seconds": 2,
        "started_seconds": 0,
        "object_track_id": "car",
        "subject_track_id": "person",
    }
    rows = [{"type": "session"}]
    for i, tracks in enumerate((["car", "person"], ["car"], ["car", "person"], ["car", "person"])):
        rows.append(
            {
                "type": "frame",
                "frame_index": i * 10,
                "timestamp_seconds": i,
                "regions": [{"track_id": t} for t in tracks],
                "events": [event] if i == 2 else [],
            }
        )
    result = link_event_frames(rows)
    assert result[3]["events"][0]["review_frame_indices"] == [0, 20]
    assert "review_frame_indices" not in event
    event["started_seconds"] = float("nan")
    assert link_event_frames(rows)[3]["events"][0]["review_frame_indices"] == []
