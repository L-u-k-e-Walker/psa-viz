"""Gate 4: check the final MP4/GIF outputs with ffprobe (env viz). Writes docs/gate4_qc.json.

Run from the project root: conda run -n viz python -I tools/gate4_qc.py
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = {"column": ROOT / "column" / "render", "particles": ROOT / "particle" / "render"}
FRAMES = 600


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,nb_read_frames,codec_name,pix_fmt,r_frame_rate:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    s = d["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return dict(width=s["width"], height=s["height"], frames=int(s["nb_read_frames"]), codec=s["codec_name"],
                pix_fmt=s.get("pix_fmt"), fps=num / den, duration_s=float(d["format"]["duration"]),
                size_MB=round(path.stat().st_size / 1e6, 2))


res, ok = {}, True
for vis, d in OUT.items():
    stem = "psa_column" if vis == "column" else "psa_particles"
    frames_dir = d / "frames"
    for theme in ("dark", "white"):
        n_png = len(list((frames_dir / theme).glob("f_*.png"))) if (frames_dir / theme).exists() else None
        for ext in ("mp4", "gif"):
            p = d / f"{stem}_{theme}.{ext}"
            if not p.exists():
                res[p.name] = "missing"
                ok = False
                continue
            info = probe(p)
            if ext == "mp4":
                checks = dict(size=(info["width"], info["height"]) == (1920, 1080), frames=info["frames"] == FRAMES,
                              duration=19.0 <= info["duration_s"] <= 21.0, h264=info["codec"] == "h264",
                              yuv420p=info["pix_fmt"] == "yuv420p", all_png_frames=n_png == FRAMES)
            else:
                checks = dict(size=info["width"] in (1920, 1280, 960) and abs(info["width"] / info["height"] - 16 / 9) < 0.01,
                              duration=19.0 <= info["duration_s"] <= 21.0, under_50MB=info["size_MB"] <= 50.0)
            info["checks"] = checks
            ok &= all(checks.values())
            res[p.name] = info
res["labels"] = ("legends in both visuals read 'A  heavy · extract' and 'B  light · raffinate' "
                 "(psaviz.style.SPECIES_LABEL); column flow arrows name Raffinate/Extract/Purge")
res["pass"] = bool(ok)
(ROOT / "docs" / "gate4_qc.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
