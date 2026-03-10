"""Command-line interface for fasthcc."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fasthcc.reader import HCCFile


def _format_bytes(n_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{n_bytes / 1024 ** 3:.2f} GB"


def _calibration_mode_str(mode: int) -> str:
    """Convert calibration mode integer to descriptive string."""
    modes = {
        0: "raw",
        1: "NUC",
        2: "RT",
    }
    return modes.get(mode, f"unknown ({mode})")


def cmd_info(args: argparse.Namespace) -> None:
    """Print detailed info about an HCC file."""
    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with HCCFile(path) as hcc:
        file_size = path.stat().st_size

        n_frames = hcc.n_frames
        width = hcc.width
        height = hcc.height
        frame_rate = hcc.frame_rate
        duration = n_frames / frame_rate if frame_rate > 0 else float("inf")

        mem_uint16 = n_frames * height * width * 2
        mem_float64 = n_frames * height * width * 8

        print(f"File:             {path}")
        print(f"File size:        {_format_bytes(file_size)}")
        print(f"Header version:   {hcc.header_version}")
        print(f"Dimensions:       {width} x {height}")
        print(f"Number of frames: {n_frames}")
        print(f"Frame rate:       {frame_rate:.2f} Hz")
        print(f"Calibration mode: {_calibration_mode_str(hcc.calibration_mode)}")
        print(f"Data offset:      {hcc.data_offset}")
        print(f"Data exponent:    {hcc.data_exp}")
        print(f"Duration:         {duration:.3f} s")
        print(f"Memory (uint16):  {_format_bytes(mem_uint16)}")
        print(f"Memory (float64): {_format_bytes(mem_float64)}")


def _discover_hcc_files(path: Path, recursive: bool) -> list[Path]:
    """Find all .hcc files at the given path."""
    if path.is_file():
        if path.suffix.lower() == ".hcc":
            return [path]
        else:
            print(f"Warning: {path} is not an .hcc file, skipping.", file=sys.stderr)
            return []
    elif path.is_dir():
        pattern = "**/*.hcc" if recursive else "*.hcc"
        files = sorted(path.glob(pattern))
        if not files:
            print(f"No .hcc files found in {path}", file=sys.stderr)
        return files
    else:
        print(f"Error: path not found: {path}", file=sys.stderr)
        sys.exit(1)


def cmd_convert(args: argparse.Namespace) -> None:
    """Convert HCC file(s) to NPY format."""
    path = Path(args.path)
    files = _discover_hcc_files(path, args.recursive)

    if not files:
        return

    # Determine output dtype
    if args.calibrated:
        if args.dtype == "uint16":
            print(
                "Warning: --calibrated requires a float dtype, "
                "using float32 instead of uint16.",
                file=sys.stderr,
            )
            dtype = np.float32
        else:
            dtype = np.dtype(args.dtype).type
    else:
        dtype = np.dtype(args.dtype).type

    total_bytes_written = 0
    total_start = time.perf_counter()
    converted = 0

    for hcc_path in files:
        # Determine output path
        if args.output and len(files) == 1:
            out_path = Path(args.output)
        elif args.output and path.is_dir():
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / hcc_path.with_suffix(".npy").name
        else:
            out_path = hcc_path.with_suffix(".npy")

        # Skip existing
        if args.skip_existing and out_path.exists():
            if not args.quiet:
                print(f"  SKIP {hcc_path.name} (output exists)")
            continue

        t0 = time.perf_counter()
        with HCCFile(hcc_path) as hcc:
            if not args.quiet:
                print(
                    f"  {hcc_path.name}: {hcc.width}x{hcc.height}, "
                    f"{hcc.n_frames} frames, {hcc.frame_rate:.1f} Hz"
                )

            # Read data
            if args.calibrated:
                data = hcc.to_calibrated(dtype=dtype)
            else:
                data = hcc.read_frames()
                if data.dtype != dtype:
                    data = data.astype(dtype)

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        np.save(out_path, data)
        elapsed = time.perf_counter() - t0

        file_bytes = data.nbytes
        total_bytes_written += file_bytes
        speed = file_bytes / elapsed / 1024 ** 2 if elapsed > 0 else float("inf")
        converted += 1

        if not args.quiet:
            print(
                f"    -> {out_path.name} [{_format_bytes(file_bytes)}, "
                f"{elapsed:.2f}s, {speed:.1f} MB/s]"
            )

    total_elapsed = time.perf_counter() - total_start
    if not args.quiet and converted > 0:
        total_speed = (
            total_bytes_written / total_elapsed / 1024 ** 2
            if total_elapsed > 0
            else float("inf")
        )
        print(
            f"\nConverted {converted} file(s) in {total_elapsed:.2f}s "
            f"({_format_bytes(total_bytes_written)}, {total_speed:.1f} MB/s)"
        )


def main() -> None:
    """Entry point for the fasthcc CLI."""
    parser = argparse.ArgumentParser(
        prog="fasthcc",
        description="Fast reader for Telops HCC infrared camera files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- info subcommand ---
    info_parser = subparsers.add_parser(
        "info",
        help="Print detailed info about an HCC file.",
    )
    info_parser.add_argument(
        "file",
        type=str,
        help="Path to the .hcc file.",
    )
    info_parser.set_defaults(func=cmd_info)

    # --- convert subcommand ---
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert HCC file(s) to NPY format.",
    )
    convert_parser.add_argument(
        "path",
        type=str,
        help="Path to an .hcc file or a directory containing .hcc files.",
    )
    convert_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path. For single file: output .npy path. "
             "For directory: output directory. Default: same name with .npy extension.",
    )
    convert_parser.add_argument(
        "--dtype",
        type=str,
        choices=["uint16", "float32", "float64"],
        default="uint16",
        help="Output array dtype (default: uint16).",
    )
    convert_parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Apply denormalization (data * 2^exp + offset). Implies float dtype.",
    )
    convert_parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search directory recursively for .hcc files.",
    )
    convert_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion if output .npy already exists.",
    )
    convert_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output.",
    )
    convert_parser.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
