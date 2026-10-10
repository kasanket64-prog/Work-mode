# Prototype verification — 6 October 2026

Windows, Python 3.12.14, existing isolated environment and pinned model snapshots from Phases 1–3.

- Simulated event sequence: one alert at timestamp 33, after 30 seconds of observed absence, with no duplicates. This is a software fixture, not CCTV accuracy.
- Real-model regression: all three image/video integration tests passed, including stable track IDs, track-referenced relation endpoints, temporal confirmation and no bag alerts on the horse fixture.
- Final full suite: **58 passed in 42.62 seconds**, including all three real-model integration tests and the localhost API test. Lint and dependency consistency checks also passed.
- Browser review: the frame slider was exercised and the simulated candidate event at timestamp 33 was visibly verified. The localhost snapshot demo is available while its server is running.
- A sandboxed model attempt failed on upstream temporary-file permissions. Real-model verification then ran with approved non-sandboxed execution and Hugging Face offline flags.

Tests also cover timestamp expiry, reordered detections, short misses, temporal evidence expiry, resets after gaps/occlusion/movement, duplicate-event scoring, annotation mappings, HTML escaping, output protection and local API responses.

Physical webcam capture and annotated real unattended-bag footage were not tested. No GPU performance or real-world accuracy claims are made. Nothing has been pushed or deployed.
