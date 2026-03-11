"""
Benchmark: fasthcc vs TelopsToolbox for reading HCC files.

Compares read speed and correctness of fasthcc against the reference
TelopsToolbox implementation for a real Telops FAST M3k HCC recording.

Correctness checks:
  1. fasthcc raw uint16 vs reference NPY (which stores raw counts as float64)
  2. fasthcc calibrated float32 vs TelopsToolbox calibrated float32
"""

import os
import sys
import time
import statistics
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration — set via environment variables or edit these defaults
# ---------------------------------------------------------------------------
N_RUNS = 3

HCC_PATH = Path(os.environ.get("BENCH_HCC_PATH", "recording.hcc"))
REF_NPY_PATH = Path(os.environ.get("BENCH_REF_NPY_PATH", "recording.npy"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_size(nbytes):
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def fmt_time(seconds):
    """Human-readable time."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def timed(fn, n=N_RUNS):
    """Run fn() n times, return (median_seconds, last_result)."""
    times = []
    result = None
    for i in range(n):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"    run {i+1}/{n}: {fmt_time(elapsed)}")
    med = statistics.median(times)
    return med, result


# ===================================================================
# BENCHMARK
# ===================================================================

print("=" * 70)
print("BENCHMARK: fasthcc vs TelopsToolbox")
print("=" * 70)
print()

# File info
file_size = HCC_PATH.stat().st_size
print(f"HCC file : {HCC_PATH.name}")
print(f"File size: {fmt_size(file_size)}")
print()

# --- fasthcc: raw uint16 ---
print("-" * 50)
print("[1/3] fasthcc -- raw uint16 read")
print("-" * 50)

from fasthcc import read_hcc, HCCReader

# Quick peek for file info
with HCCReader(HCC_PATH) as hcc:
    n_frames = hcc.n_frames
    width = hcc.width
    height = hcc.height
    cal_mode = hcc.calibration_mode
    frame_rate = hcc.frame_rate
    hdr_ver = hcc.header_version
    data_offset = hcc.data_offset
    data_exp = hcc.data_exp

print(f"  Frames: {n_frames}, Resolution: {width}x{height}")
print(f"  Header version: v{hdr_ver[0]}.{hdr_ver[1]}, CalMode: {cal_mode}")
print(f"  Frame rate: {frame_rate:.1f} Hz")
print(f"  Calibration: DataOffset={data_offset}, DataExp={data_exp}")
print()

t_fasthcc_raw, data_raw = timed(lambda: read_hcc(HCC_PATH))
print(f"  -> Median: {fmt_time(t_fasthcc_raw)}")
print(f"  -> Shape: {data_raw.shape}, dtype: {data_raw.dtype}")
print(f"  -> Value range: [{data_raw.min()}, {data_raw.max()}]")
print()

# --- fasthcc: calibrated float32 ---
print("-" * 50)
print("[2/3] fasthcc -- calibrated float32 read")
print("-" * 50)

t_fasthcc_cal, data_cal = timed(lambda: read_hcc(HCC_PATH, calibrated=True))
print(f"  -> Median: {fmt_time(t_fasthcc_cal)}")
print(f"  -> Shape: {data_cal.shape}, dtype: {data_cal.dtype}")
print(f"  -> Value range: [{data_cal.min():.4f}, {data_cal.max():.4f}] K")
print()

# --- TelopsToolbox ---
print("-" * 50)
print("[3/3] TelopsToolbox -- calibrated float32 read")
print("-" * 50)

# Add TelopsToolbox to path (set TELOPS_TOOLBOX_PATH if not on sys.path)
_telops_path = os.environ.get("TELOPS_TOOLBOX_PATH")
if _telops_path:
    sys.path.insert(0, _telops_path)

import math

# Monkey-patch np.memmap for numpy 2.x
_OrigMemmap = np.memmap

class _SafeMemmap(_OrigMemmap):
    def __new__(cls, filename, dtype='uint8', mode='r+', offset=0,
                shape=None, order='C'):
        return _OrigMemmap.__new__(
            cls, filename, dtype=dtype, mode=mode,
            offset=int(offset), shape=shape, order=order
        )

np.memmap = _SafeMemmap

# Monkey-patch SequenceReaderP.open_file
import TelopsToolbox.hcc.SequenceReaderP as _srp
from natsort import natsorted
import glob as _glob

def _patched_open_file(self, file_names):
    if type(file_names) != str:
        file_path = file_names[0]
    else:
        file_path = file_names
    files_just_names = []
    files = []
    if os.path.isfile(file_path):
        d = {"name": os.path.basename(file_path),
             "folder": os.path.dirname(file_path) or "."}
        files.append(d)
        files_just_names.append(d["name"])
        file_names = [file_path]
    else:
        for f in _glob.glob(os.path.join(file_path, "*.hcc")):
            d = {"name": os.path.basename(f), "folder": os.path.dirname(f)}
            files.append(d)
            files_just_names.append(d["name"])
        files_sorted = [None] * len(files)
        files_just_names_sorted = natsorted(files_just_names)
        for dic_index in range(len(files)):
            current_name = files[dic_index]["name"]
            for name_index in range(len(files_just_names_sorted)):
                if files_just_names_sorted[name_index] == current_name:
                    files_sorted[name_index] = files[dic_index]
        files = files_sorted
        file_names_list = []
        for f in files:
            file_names_list.append(os.path.join(f["folder"], f["name"]))
        file_names = file_names_list

    self.private_length, self.header = self.peek(files)
    for i in range(len(self.private_length)):
        self.private_length[i] = math.floor(self.private_length[i])
    h = self.header
    self.width = int(h["Width"][0])
    self.height = int(h["Height"][0])
    self.length = self.get_length()
    self.size_on_disk = 2 * self.length * self.width * (self.height + 2)
    self.eachfile_size_on_disk = []
    for i in range(len(self.private_length)):
        self.eachfile_size_on_disk.append(
            2 * int(self.private_length[i]) * self.width * (self.height + 2))
    self.size_in_ram = (
        (4 * self.length * self.width * self.height) +
        (2 * self.length * self.width * 2)
    )
    self.format = []
    self.memmap = []
    if isinstance(file_names, str):
        file_names = [file_names]
    for i in range(len(files)):
        offset_header = 0
        offset_data = 2 * self.width * 2
        frame_size = int(self.eachfile_size_on_disk[i] / self.private_length[i])
        self.memmap.append([])
        self.memmap[i].append({"Header": []})
        self.memmap[i].append({"Data": []})
        for j in range(self.private_length[i]):
            if (j + 1) % 1000 == 0:
                print(f"  memmap: frame {j+1}/{self.private_length[i]}")
            self.memmap[i][0]["Header"].append(
                np.memmap(file_names[i], dtype=np.uint8, mode='r',
                          offset=offset_header,
                          shape=(2 * self.width * 2, 1)))
            self.memmap[i][1]["Data"].append(
                np.memmap(file_names[i], dtype=np.uint16, mode='r',
                          offset=offset_data,
                          shape=(self.width * self.height, 1)))
            offset_header += frame_size
            offset_data += frame_size

_srp.SequenceReaderP.open_file = _patched_open_file

from TelopsToolbox.hcc.readIRCam import read_ircam
import TelopsToolbox.utils.image_processing as ip


def telops_read():
    ir_data, header, _, _ = read_ircam(str(HCC_PATH))
    images = np.asarray(ip.form_image(header, ir_data), dtype=np.float32)
    return images


t_telops, data_telops = timed(telops_read)
print(f"  -> Median: {fmt_time(t_telops)}")
print(f"  -> Shape: {data_telops.shape}, dtype: {data_telops.dtype}")
print(f"  -> Value range: [{data_telops.min():.4f}, {data_telops.max():.4f}] K")
print()

# ===================================================================
# CORRECTNESS CHECK
# ===================================================================
print("=" * 70)
print("CORRECTNESS CHECK")
print("=" * 70)
print()

# --- Check 1: fasthcc raw vs reference NPY ---
# The reference NPY stores raw uint16 pixel counts cast to float64
# (no calibration applied). This was verified by inspecting values.
print("Loading reference NPY...")
ref = np.load(str(REF_NPY_PATH))
print(f"  Reference: shape={ref.shape}, dtype={ref.dtype}, "
      f"range=[{ref.min():.1f}, {ref.max():.1f}]")
print()

print("Check 1: fasthcc raw uint16 vs reference NPY (raw counts as float64)")
print(f"  fasthcc raw: shape={data_raw.shape}, dtype={data_raw.dtype}, "
      f"range=[{data_raw.min()}, {data_raw.max()}]")

shapes_match_raw = data_raw.shape == ref.shape
exact_match = np.array_equal(data_raw.astype(np.float64), ref)
allclose_raw = np.allclose(data_raw.astype(np.float64), ref, rtol=0, atol=0)

print(f"  Shapes match:  {shapes_match_raw}")
print(f"  Exact match:   {exact_match}")
print(f"  allclose:      {allclose_raw}")
if not exact_match and shapes_match_raw:
    diff = np.abs(data_raw.astype(np.float64) - ref)
    print(f"  max |diff|:    {diff.max():.6f}")
    n_diff = np.count_nonzero(diff)
    print(f"  # differing:   {n_diff} / {diff.size} pixels")
print()

# --- Check 2: fasthcc calibrated vs TelopsToolbox ---
print("Check 2: fasthcc calibrated float32 vs TelopsToolbox float32")
print(f"  fasthcc cal: shape={data_cal.shape}, dtype={data_cal.dtype}, "
      f"range=[{data_cal.min():.4f}, {data_cal.max():.4f}]")
print(f"  TelopsToolbox: shape={data_telops.shape}, dtype={data_telops.dtype}, "
      f"range=[{data_telops.min():.4f}, {data_telops.max():.4f}]")

shapes_match_cal = data_cal.shape == data_telops.shape
if shapes_match_cal:
    exact_match_cal = np.array_equal(data_cal, data_telops)
    allclose_cal = np.allclose(data_cal, data_telops, rtol=1e-5, atol=1e-4)
    diff_cal = np.abs(data_cal.astype(np.float64) - data_telops.astype(np.float64))
    max_diff_cal = diff_cal.max()
    mean_diff_cal = diff_cal.mean()
    print(f"  Shapes match:  {shapes_match_cal}")
    print(f"  Exact match:   {exact_match_cal}")
    print(f"  allclose:      {allclose_cal}  (rtol=1e-5, atol=1e-4)")
    print(f"  max |diff|:    {max_diff_cal:.8f}")
    print(f"  mean |diff|:   {mean_diff_cal:.8f}")
else:
    print(f"  Shape mismatch! fasthcc={data_cal.shape}, Telops={data_telops.shape}")
    allclose_cal = False
    exact_match_cal = False
print()

# ===================================================================
# RESULTS SUMMARY
# ===================================================================
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print()

speedup_raw = t_telops / t_fasthcc_raw if t_fasthcc_raw > 0 else float('inf')
speedup_cal = t_telops / t_fasthcc_cal if t_fasthcc_cal > 0 else float('inf')

print(f"  File info")
print(f"  {'HCC file:':<20} {HCC_PATH.name}")
print(f"  {'File size:':<20} {fmt_size(file_size)}")
print(f"  {'Frames:':<20} {n_frames}")
print(f"  {'Resolution:':<20} {width} x {height}")
print(f"  {'Frame rate:':<20} {frame_rate:.1f} Hz")
print(f"  {'Calibration mode:':<20} {cal_mode} (0=raw, 1=NUC, 2=RT)")
print(f"  {'Header version:':<20} v{hdr_ver[0]}.{hdr_ver[1]}")
print()

col_w = [37, 14, 10]
print(f"  {'Method':<{col_w[0]}} {'Median time':>{col_w[1]}} {'Speedup':>{col_w[2]}}")
print(f"  {'-'*col_w[0]} {'-'*col_w[1]} {'-'*col_w[2]}")
print(f"  {'fasthcc (raw uint16)':<{col_w[0]}} "
      f"{fmt_time(t_fasthcc_raw):>{col_w[1]}} "
      f"{speedup_raw:>{col_w[2]-1}.1f}x")
print(f"  {'fasthcc (calibrated float32)':<{col_w[0]}} "
      f"{fmt_time(t_fasthcc_cal):>{col_w[1]}} "
      f"{speedup_cal:>{col_w[2]-1}.1f}x")
print(f"  {'TelopsToolbox (calibrated float32)':<{col_w[0]}} "
      f"{fmt_time(t_telops):>{col_w[1]}} "
      f"{'1.0x':>{col_w[2]}}")
print()

raw_status = "PASS (exact)" if exact_match else ("PASS (allclose)" if allclose_raw else "FAIL")
cal_status = "PASS (exact)" if exact_match_cal else ("PASS (allclose)" if allclose_cal else "FAIL")

print(f"  Correctness")
print(f"    fasthcc raw  vs ref NPY:      {raw_status}")
print(f"    fasthcc cal  vs TelopsToolbox: {cal_status}")
print()
print("=" * 70)
