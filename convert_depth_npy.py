#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import time

import numpy as np

try:
    from imageio.v2 import imwrite  # type: ignore[reportMissingImports]
except Exception:  # noqa: BLE001
    imwrite = None

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None


def find_npy_files(root: Path) -> list[Path]:
    return sorted(root.rglob("depth/*.npy"))


def render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[{}] 0/0 0.0%".format(" " * width)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * current / total
    return "[{}] {}/{} {:5.1f}%".format(bar, current, total, pct)


def print_progress(current: int, total: int) -> None:
    msg = render_progress(current, total)
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def prepare_png_array(
    array: np.ndarray,
    scale: float | None,
    auto_scale: bool,
    clip_min: float | None,
    clip_max: float | None,
) -> np.ndarray:
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {array.shape}")

    if clip_min is not None or clip_max is not None:
        array = np.clip(
            array,
            clip_min if clip_min is not None else np.min(array),
            clip_max if clip_max is not None else np.max(array),
        )

    if array.dtype == np.uint8 or array.dtype == np.uint16:
        return np.ascontiguousarray(array)

    if scale is not None:
        scaled = array.astype(np.float64) * scale
        scaled = np.clip(scaled, 0, 65535)
        return scaled.astype(np.uint16)

    if auto_scale:
        min_val = float(np.min(array))
        max_val = float(np.max(array))
        if max_val <= min_val:
            return np.zeros_like(array, dtype=np.uint16)
        normalized = (array.astype(np.float64) - min_val) / (max_val - min_val)
        scaled = np.clip(normalized * 65535.0, 0, 65535)
        return scaled.astype(np.uint16)

    raise ValueError(
        "Unsupported dtype; provide --scale or enable auto-scale."
    )


def write_png(path: Path, array: np.ndarray) -> None:
    if imwrite is not None:
        imwrite(path, array)
        return
    if Image is not None:
        Image.fromarray(array).save(path)
        return
    raise RuntimeError("No PNG writer available.")


def convert_file(
    npy_path: Path,
    output_suffix: str,
    scale: float | None,
    auto_scale: bool,
    clip_min: float | None,
    clip_max: float | None,
    delete_source: bool,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[bool, str | None]:
    out_path = npy_path.with_suffix(output_suffix)
    if skip_existing and out_path.exists():
        return False, None

    if dry_run:
        return True, None

    tmp_path = out_path.with_name(out_path.name + ".tmp.png")
    try:
        array = np.load(npy_path, allow_pickle=False)
        png_array = prepare_png_array(
            array=array,
            scale=scale,
            auto_scale=auto_scale,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        write_png(tmp_path, png_array)
        tmp_path.replace(out_path)
        if delete_source:
            npy_path.unlink()
        return True, None
    except Exception as exc:  # noqa: BLE001
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:  # noqa: BLE001
                pass
        return False, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert depth .npy files to PNG with progress."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/colosseum_wrist_data"),
        help="Root datasets folder (default: datasets)",
    )
    parser.add_argument(
        "--output-suffix",
        default=".png",
        help="Output file suffix (default: .png)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output file already exists",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep original .npy files (default deletes after conversion)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count files, do not write outputs",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="Multiply depth by scale before saving (default: 1000 for m->mm)",
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Auto-scale to full 16-bit range when needed (default: off)",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Clip minimum depth before scaling/auto-scaling",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Clip maximum depth before scaling/auto-scaling",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    if not root.exists():
        print(f"Root not found: {root}")
        return 1

    if imwrite is None and Image is None:
        print("Missing image writer. Install one of: imageio or pillow.")
        return 1

    files = find_npy_files(root)
    total = len(files)
    if total == 0:
        print("No .npy files found under depth/ folders.")
        return 0

    processed = 0
    skipped = 0
    failed = 0
    errors: list[str] = []

    start = time.time()
    for idx, npy_path in enumerate(files, start=1):
        ok, err = convert_file(
            npy_path=npy_path,
            output_suffix=args.output_suffix,
            scale=args.scale,
            auto_scale=args.auto_scale,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
            delete_source=not args.keep_source,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        if ok:
            processed += 1
        else:
            if err is None:
                skipped += 1
            else:
                failed += 1
                errors.append(f"{npy_path}: {err}")

        print_progress(idx, total)

    elapsed = time.time() - start
    print(
        f"Done. total={total}, processed={processed}, "
        f"skipped={skipped}, failed={failed}, "
        f"elapsed={elapsed:.1f}s"
    )
    if errors:
        print("Errors (first 5):")
        for line in errors[:5]:
            print(f"  {line}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
