"""Record each session's video resolution.

MediaPipe writes x as a fraction of frame *width* and y as a fraction of
frame *height*, so the stored coordinates are anisotropic whenever the video
is not square: the same physical hand motion produces different dx and dy
depending on the aspect ratio. Nine of the own-recording sessions are off the
640x480 spec (portrait phone captures, 832x464, 480x512 and so on), which
puts every one of their features on a different scale from the rest.

Correcting that needs the resolution, which the collector does not record in
metadata.json, so it is read back off the video here and cached. Run with a
Python that has OpenCV — the trainer venv does not:

    ../data_collector/venv/bin/python scan_resolutions.py

Writes session_resolutions.json, which `preprocessing.aspect_correct` reads.
Sessions missing from the file are left uncorrected and reported as such.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
OUT = PROJECT_ROOT / "session_resolutions.json"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def main():
    try:
        import cv2
    except ImportError:
        sys.exit("OpenCV not importable. Run this with the collector venv:\n"
                 "    ../data_collector/venv/bin/python scan_resolutions.py")

    dataset_dirs = sys.argv[1:] or ["dataset", "dataset_WITA", "dataset_IPN"]
    out, missing = {}, []
    for name in dataset_dirs:
        root = PROJECT_ROOT / name
        if not root.is_dir():
            continue
        for session in sorted(root.glob("P*/S*")):
            videos = [p for p in session.iterdir()
                      if p.suffix.lower() in VIDEO_SUFFIXES]
            key = f"{name}/{session.parent.name}/{session.name}"
            if not videos:
                missing.append(key)
                continue
            cap = cv2.VideoCapture(str(videos[0]))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if w <= 0 or h <= 0:
                missing.append(key)
                continue
            out[key] = {"width": w, "height": h,
                        "aspect": round(w / h, 6),
                        "fps": round(fps, 3) if fps > 0 else None,
                        "frames": frames if frames > 0 else None,
                        "video": videos[0].name}

    OUT.write_text(json.dumps({"sessions": out, "missing": missing}, indent=2))
    shapes = {}
    for v in out.values():
        shapes.setdefault(f"{v['width']}x{v['height']}", []).append(v)
    print(f"{len(out)} session(s) -> {OUT.name}"
          + (f", {len(missing)} without a readable video" if missing else ""))
    for shape, vs in sorted(shapes.items(), key=lambda kv: -len(kv[1])):
        fpss = sorted({v["fps"] for v in vs if v["fps"]})
        print(f"  {shape:12s} x{len(vs):3d}  aspect {vs[0]['aspect']:.3f}  "
              f"fps {', '.join(f'{f:g}' for f in fpss) or '?'}")


if __name__ == "__main__":
    main()
