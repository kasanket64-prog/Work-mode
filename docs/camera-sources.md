# Camera source directory

The review dashboard includes a searchable subset of 158 entries explicitly
labeled SCDOT in baywolf88's [seeallthethings South Carolina catalog](https://github.com/baywolf88/seeallthethings/blob/master/South%20Carolina).
Names, areas and coordinates are retained; historical stream URLs and unrelated
webcam discovery queries are excluded. Source file blob SHA:
`f69747618ceab11e8522820c889dbce71ef41e9a` (retrieved 2026-10-07).

Search by road or location in Camera sources. Use the current official
[511SC portal](https://www.511sc.org/) to locate a camera by its catalog name.
The portal was checked on 2026-10-07; individual locations and stream availability
were not verified. The catalog is historical reference data, not a live status
list. SCDOT states it does not record traffic-camera footage for past retrieval.

This integration is source discovery only. It does not ingest, record or analyze
remote feeds. A live feed
integration needs a current publisher-supported stream and verified permission
for the intended recording/analysis. No third-party camera is contacted when
opening or searching this local directory.

## Explicit network input

The scene analysis CLI accepts a direct authorized video URL:

```powershell
.\.venv\Scripts\python.exe -m camera_system.video_cli --stream "https://YOUR-PROVIDER/live.m3u8" --output outputs/stream-session.jsonl --max-frames 10 --stride 15 --include-preview
```

Replace the placeholder with an actual stream, not a webpage. HTTP, HTTPS and
RTSP are supported through OpenCV/FFmpeg; embedded usernames/passwords are
rejected. No SCDOT endpoint has been verified or auto-connected. The directory
does not supply playable URLs. The plate-only scanner still takes local files.

Connection and read timeouts are 10 seconds each; the frame limit bounds the
number of analyzed frames, not total wall time including model inference.
Disconnects fail the run with a summary while retaining completed JSONL frames.
No automatic retries are made. Network timestamps are host receipt times, not
camera capture times. Backend buffering may add latency; this is not a realtime
guarantee. URLs are omitted from session metadata; do not put secrets in URLs
because the video backend or shell may log them. Review remains a saved snapshot.
