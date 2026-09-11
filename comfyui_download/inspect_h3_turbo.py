"""Inspect generated H3 MP4 streams and save representative frames for review."""

import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def inspect(path):
    wanted = {0, 1, 5, 10, 20, 60, 120}
    selected = {}
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        result = {"path": str(path.resolve()), "bytes": path.stat().st_size,
                  "video_codec": stream.codec_context.name,
                  "width": stream.width, "height": stream.height,
                  "fps": float(stream.average_rate),
                  "container_duration_s": container.duration / av.time_base}
        for index, frame in enumerate(container.decode(stream)):
            count += 1
            if index in wanted:
                selected[index] = frame.to_ndarray(format="rgb24")
    result["decoded_frames"] = count
    result["frames"] = {
        str(index): {"std": float(frame.std()),
                     "unique_colors": len(np.unique(frame.reshape(-1, 3), axis=0)),
                     "mean_rgb": frame.mean(axis=(0, 1)).tolist()}
        for index, frame in selected.items() if index in (10, 60, 120)}
    result["motion_mad"] = {
        str(gap): float(np.abs(selected[gap].astype(np.float32) - selected[0]).mean())
        for gap in (1, 5, 20) if gap in selected}
    with av.open(str(path)) as container:
        audio = container.streams.audio[0]
        samples = np.concatenate([frame.to_ndarray() for frame in container.decode(audio)], axis=1)
        result["audio"] = {
            "codec": audio.codec_context.name, "sample_rate": audio.rate,
            "channels": len(audio.layout.channels), "decoded_samples": samples.shape[1],
            "duration_s": samples.shape[1] / audio.rate,
            "finite": bool(np.isfinite(samples).all()),
            "rms": float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))),
            "peak": float(np.max(np.abs(samples)))}
    sheet = Image.new("RGB", (1296, 270), "#202020")
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate((10, 60, 120)):
        if index in selected:
            sheet.paste(Image.fromarray(selected[index]).resize((432, 240)), (column * 432, 30))
            draw.text((column * 432 + 8, 8), f"frame {index}", fill="white")
    contact = Path(__file__).resolve().parent / (path.stem + "_contact.jpg")
    sheet.save(contact, quality=90)
    result["contact_sheet"] = str(contact)
    return result


if __name__ == "__main__":
    results = [inspect(Path(path)) for path in sys.argv[1:]]
    target = Path(__file__).resolve().parent / "h3_turbo_v4_8step_quality.json"
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
