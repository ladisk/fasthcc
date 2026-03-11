"""
Comprehensive test suite for the fasthcc package.

Tests cover header parsing, HCC file reading, calibration,
the convenience read_hcc function, CLI commands, and edge cases.

All tests use synthetic HCC files built from scratch -- no real
camera files are needed.
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import pytest

from fasthcc.header import (
    detect_version,
    frame_stride,
    header_raw_size,
    parse_header,
)
from fasthcc.reader import HCCReader, read_hcc
from fasthcc.cli import cmd_info, cmd_convert


# ======================================================================
# Helper: synthetic HCC file builder
# ======================================================================

def make_hcc(
    path,
    width=16,
    height=8,
    n_frames=3,
    pixel_fill=1000,
    version=(12, 7),
    cal_mode=2,
    data_offset=273.15,
    data_exp=-8,
    exposure_time=11016,
    frame_rate_raw=2000000,
    pixel_data=None,
):
    """Create a synthetic HCC file with known contents.

    Parameters
    ----------
    path : Path
        Output file path.
    width, height : int
        Image dimensions.
    n_frames : int
        Number of frames to write.
    pixel_fill : int
        Constant uint16 value for all pixels (ignored if pixel_data given).
    version : tuple[int, int]
        (major, minor) header version.
    cal_mode : int
        CalibrationMode field.
    data_offset : float
        DataOffset field (float32).
    data_exp : int
        DataExp field (int8).
    exposure_time : int
        Raw ExposureTime (uint32, real = raw/100).
    frame_rate_raw : int
        Raw AcquisitionFrameRate (uint32, real = raw/1000).
    pixel_data : list[np.ndarray] or None
        If given, must be a list of n_frames arrays each (height, width) uint16.
        Overrides pixel_fill.

    Returns
    -------
    Path
        The path written to (same as input).
    """
    major, minor = version
    hdr_size = 2 * width * 2  # "2 header lines" convention

    with open(path, "wb") as f:
        for frame_idx in range(n_frames):
            # Build header buffer (zero-filled, then stamp known fields)
            hdr = bytearray(hdr_size)

            # Signature
            hdr[0:2] = b"TC"
            # Version
            hdr[2] = minor
            hdr[3] = major
            # ImageHeaderLength (uint16 LE)
            struct.pack_into("<H", hdr, 4, hdr_size)
            # FrameID (uint32 LE)
            struct.pack_into("<I", hdr, 8, frame_idx)
            # DataOffset (float32 LE)
            struct.pack_into("<f", hdr, 12, data_offset)
            # DataExp (int8)
            struct.pack_into("<b", hdr, 16, data_exp)
            # ExposureTime (uint32 LE)
            struct.pack_into("<I", hdr, 24, exposure_time)
            # CalibrationMode (uint8)
            hdr[28] = cal_mode
            # Width (uint16 LE)
            struct.pack_into("<H", hdr, 32, width)
            # Height (uint16 LE)
            struct.pack_into("<H", hdr, 34, height)
            # AcquisitionFrameRate (uint32 LE)
            struct.pack_into("<I", hdr, 44, frame_rate_raw)

            f.write(hdr)

            # Pixel data: (height, width) uint16 LE
            if pixel_data is not None:
                pixels = pixel_data[frame_idx].astype("<u2")
            else:
                pixels = np.full((height, width), pixel_fill, dtype="<u2")
            f.write(pixels.tobytes())

    return path


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def hcc_path(tmp_path):
    """A 3-frame synthetic HCC file with constant pixel value 1000."""
    return make_hcc(tmp_path / "test.hcc")


@pytest.fixture
def hcc_path_varied(tmp_path):
    """A 4-frame HCC file where each frame has different pixel values."""
    w, h = 16, 8
    pixel_data = [
        np.full((h, w), fill, dtype=np.uint16)
        for fill in [100, 200, 300, 400]
    ]
    return make_hcc(
        tmp_path / "varied.hcc",
        width=w, height=h, n_frames=4,
        pixel_data=pixel_data,
    )


@pytest.fixture
def single_frame_hcc(tmp_path):
    """A 1-frame HCC file."""
    return make_hcc(tmp_path / "single.hcc", n_frames=1, pixel_fill=42)


# ======================================================================
# header.py tests
# ======================================================================

class TestDetectVersion:
    """Tests for detect_version()."""

    def test_basic(self):
        hdr = bytearray(48)
        hdr[0:2] = b"TC"
        hdr[2] = 7   # minor
        hdr[3] = 12  # major
        assert detect_version(hdr) == (12, 7)

    def test_offset(self):
        buf = bytearray(10) + bytearray(48)
        buf[10:12] = b"TC"
        buf[12] = 3
        buf[13] = 10
        assert detect_version(buf, offset=10) == (10, 3)

    def test_invalid_signature(self):
        buf = bytearray(48)
        buf[0:2] = b"XX"
        with pytest.raises(ValueError, match="Invalid HCC signature"):
            detect_version(buf)

    def test_truncated_buffer(self):
        buf = b"TC\x07"  # only 3 bytes, need 6
        with pytest.raises(ValueError, match="Buffer too short"):
            detect_version(buf)


class TestParseHeader:
    """Tests for parse_header()."""

    def _make_header_buf(self, width=16, height=8, version=(12, 7),
                         data_offset=273.15, data_exp=-8, cal_mode=2,
                         exposure_time=11016, frame_rate_raw=2000000,
                         frame_id=0):
        """Build a minimal header buffer for testing."""
        hdr_size = 2 * width * 2
        hdr = bytearray(hdr_size)
        hdr[0:2] = b"TC"
        hdr[2] = version[1]
        hdr[3] = version[0]
        struct.pack_into("<H", hdr, 4, hdr_size)
        struct.pack_into("<I", hdr, 8, frame_id)
        struct.pack_into("<f", hdr, 12, data_offset)
        struct.pack_into("<b", hdr, 16, data_exp)
        struct.pack_into("<I", hdr, 24, exposure_time)
        hdr[28] = cal_mode
        struct.pack_into("<H", hdr, 32, width)
        struct.pack_into("<H", hdr, 34, height)
        struct.pack_into("<I", hdr, 44, frame_rate_raw)
        return bytes(hdr)

    def test_basic_fields(self):
        buf = self._make_header_buf()
        h = parse_header(buf)

        assert h["Signature"] == "TC"
        assert h["DeviceXMLMajorVersion"] == 12
        assert h["DeviceXMLMinorVersion"] == 7
        assert h["Width"] == 16
        assert h["Height"] == 8
        assert h["CalibrationMode"] == 2
        assert h["FrameID"] == 0
        assert h["DataExp"] == -8
        # DataOffset is float32; check approximate equality
        assert abs(h["DataOffset"] - 273.15) < 0.01
        # ExposureTime is raw/100
        assert abs(h["ExposureTime"] - 110.16) < 0.01
        # AcquisitionFrameRate is raw/1000
        assert abs(h["AcquisitionFrameRate"] - 2000.0) < 0.01

    def test_version_v10(self):
        buf = self._make_header_buf(version=(10, 0))
        h = parse_header(buf)
        assert h["DeviceXMLMajorVersion"] == 10
        assert h["DeviceXMLMinorVersion"] == 0

    def test_legacy_version(self):
        """Versions < 10 return prefix-only fields."""
        buf = self._make_header_buf(version=(5, 2))
        h = parse_header(buf)
        assert h["DeviceXMLMajorVersion"] == 5
        # Should NOT have V10+ tail fields like POSIXTime
        assert "POSIXTime" not in h

    def test_invalid_signature_raises(self):
        buf = bytearray(64)
        buf[0:2] = b"AB"
        with pytest.raises(ValueError, match="Invalid HCC signature"):
            parse_header(buf)

    def test_truncated_buffer_raises(self):
        """A buffer shorter than 48 bytes should raise."""
        buf = b"TC\x07\x0c" + b"\x00" * 10  # only 14 bytes
        with pytest.raises(ValueError, match="Buffer too short"):
            parse_header(buf)

    def test_image_header_length_field(self):
        buf = self._make_header_buf(width=32)
        h = parse_header(buf)
        assert h["ImageHeaderLength"] == 2 * 32 * 2  # 128

    def test_frame_id_nonzero(self):
        buf = self._make_header_buf(frame_id=42)
        h = parse_header(buf)
        assert h["FrameID"] == 42


class TestHeaderHelpers:
    """Tests for header_raw_size and frame_stride."""

    def test_header_raw_size(self):
        assert header_raw_size(16) == 64
        assert header_raw_size(320) == 1280

    def test_frame_stride(self):
        # stride = header_raw_size + width * height * 2
        assert frame_stride(16, 8) == 64 + 16 * 8 * 2
        assert frame_stride(320, 256) == 1280 + 320 * 256 * 2


# ======================================================================
# reader.py — HCCReader tests
# ======================================================================

class TestHCCReader:
    """Tests for the HCCReader class."""

    def test_dimensions(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            assert hcc.width == 16
            assert hcc.height == 8
            assert hcc.n_frames == 3

    def test_version(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            assert hcc.header_version == (12, 7)

    def test_calibration_mode(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            assert hcc.calibration_mode == 2

    def test_frame_rate(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            assert abs(hcc.frame_rate - 2000.0) < 0.01

    def test_data_offset_and_exp(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            assert abs(hcc.data_offset - 273.15) < 0.01
            assert hcc.data_exp == -8

    def test_read_frames_shape_dtype(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            frames = hcc.read_frames()
            assert frames.shape == (3, 8, 16)
            assert frames.dtype == np.uint16

    def test_read_frames_values(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            frames = hcc.read_frames()
            np.testing.assert_array_equal(frames, 1000)

    def test_read_frames_varied(self, hcc_path_varied):
        with HCCReader(hcc_path_varied) as hcc:
            frames = hcc.read_frames()
            assert frames.shape == (4, 8, 16)
            np.testing.assert_array_equal(frames[0], 100)
            np.testing.assert_array_equal(frames[1], 200)
            np.testing.assert_array_equal(frames[2], 300)
            np.testing.assert_array_equal(frames[3], 400)

    def test_read_frames_slicing(self, hcc_path_varied):
        with HCCReader(hcc_path_varied) as hcc:
            subset = hcc.read_frames(start=1, stop=3)
            assert subset.shape == (2, 8, 16)
            np.testing.assert_array_equal(subset[0], 200)
            np.testing.assert_array_equal(subset[1], 300)

    def test_read_frames_start_only(self, hcc_path_varied):
        with HCCReader(hcc_path_varied) as hcc:
            subset = hcc.read_frames(start=2)
            assert subset.shape == (2, 8, 16)
            np.testing.assert_array_equal(subset[0], 300)
            np.testing.assert_array_equal(subset[1], 400)

    def test_read_metadata(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            meta = hcc.read_metadata()
            assert len(meta) == 3
            assert meta[0]["FrameID"] == 0
            assert meta[1]["FrameID"] == 1
            assert meta[2]["FrameID"] == 2
            assert meta[0]["Width"] == 16

    def test_read_metadata_sliced(self, hcc_path_varied):
        with HCCReader(hcc_path_varied) as hcc:
            meta = hcc.read_metadata(start=1, stop=3)
            assert len(meta) == 2
            assert meta[0]["FrameID"] == 1
            assert meta[1]["FrameID"] == 2

    def test_to_calibrated(self, hcc_path):
        """Calibrated = pixel * 2^exp + offset."""
        with HCCReader(hcc_path) as hcc:
            cal = hcc.to_calibrated()
            assert cal.dtype == np.float32
            assert cal.shape == (3, 8, 16)
            # pixel=1000, exp=-8 => scale=1/256, offset~273.15
            expected = np.float32(1000 * (2.0 ** -8) + 273.15)
            np.testing.assert_allclose(cal, expected, rtol=1e-5)

    def test_to_calibrated_dtype(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            cal = hcc.to_calibrated(dtype=np.float64)
            assert cal.dtype == np.float64

    def test_to_npy_uint16(self, hcc_path, tmp_path):
        out = tmp_path / "out.npy"
        with HCCReader(hcc_path) as hcc:
            hcc.to_npy(out)
        data = np.load(out)
        assert data.shape == (3, 8, 16)
        assert data.dtype == np.uint16
        np.testing.assert_array_equal(data, 1000)

    def test_to_npy_calibrated(self, hcc_path, tmp_path):
        out = tmp_path / "cal.npy"
        with HCCReader(hcc_path) as hcc:
            hcc.to_npy(out, dtype=np.float32)
        data = np.load(out)
        assert data.dtype == np.float32
        expected = np.float32(1000 * (2.0 ** -8) + 273.15)
        np.testing.assert_allclose(data, expected, rtol=1e-5)

    def test_context_manager_closes(self, hcc_path):
        hcc = HCCReader(hcc_path)
        hcc.close()
        assert hcc._blocks is None

    def test_read_after_close_raises(self, hcc_path):
        hcc = HCCReader(hcc_path)
        hcc.close()
        with pytest.raises(RuntimeError, match="closed"):
            hcc.read_frames()

    def test_context_manager_exit(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            _ = hcc.read_frames()
        # After exiting, internal buffers should be None
        assert hcc._blocks is None

    def test_too_small_file_raises(self, tmp_path):
        tiny = tmp_path / "tiny.hcc"
        tiny.write_bytes(b"\x00" * 20)
        with pytest.raises(ValueError, match="too small"):
            HCCReader(tiny)

    def test_no_complete_frames_raises(self, tmp_path):
        """File has a valid header but not enough bytes for one full frame."""
        path = tmp_path / "incomplete.hcc"
        w, h = 16, 8
        hdr_size = 2 * w * 2  # 64
        hdr = bytearray(hdr_size)
        hdr[0:2] = b"TC"
        hdr[2] = 7
        hdr[3] = 12
        struct.pack_into("<H", hdr, 4, hdr_size)
        struct.pack_into("<H", hdr, 32, w)
        struct.pack_into("<H", hdr, 34, h)
        struct.pack_into("<I", hdr, 44, 2000000)
        # Write header only -- not enough for a full frame
        path.write_bytes(hdr)
        with pytest.raises(ValueError, match="no complete frames"):
            HCCReader(path)

    def test_repr(self, hcc_path):
        with HCCReader(hcc_path) as hcc:
            r = repr(hcc)
            assert "test.hcc" in r
            assert "3 frames" in r
            assert "16x8" in r
            assert "open" in r
        # After close
        r2 = repr(hcc)
        assert "closed" in r2


# ======================================================================
# reader.py — read_hcc convenience function tests
# ======================================================================

class TestReadHcc:
    """Tests for the read_hcc() convenience function."""

    def test_basic(self, hcc_path):
        data = read_hcc(hcc_path)
        assert data.shape == (3, 8, 16)
        assert data.dtype == np.uint16
        np.testing.assert_array_equal(data, 1000)

    def test_frames_slice(self, hcc_path_varied):
        data = read_hcc(hcc_path_varied, frames=slice(1, 3))
        assert data.shape == (2, 8, 16)
        np.testing.assert_array_equal(data[0], 200)
        np.testing.assert_array_equal(data[1], 300)

    def test_frames_slice_with_step(self, hcc_path_varied):
        data = read_hcc(hcc_path_varied, frames=slice(0, 4, 2))
        assert data.shape == (2, 8, 16)
        np.testing.assert_array_equal(data[0], 100)
        np.testing.assert_array_equal(data[1], 300)

    def test_metadata_true(self, hcc_path):
        result = read_hcc(hcc_path, metadata=True)
        assert isinstance(result, tuple)
        data, meta = result
        assert data.shape == (3, 8, 16)
        assert len(meta) == 3
        assert meta[0]["FrameID"] == 0

    def test_metadata_with_slice(self, hcc_path_varied):
        data, meta = read_hcc(hcc_path_varied, frames=slice(1, 3),
                              metadata=True)
        assert data.shape == (2, 8, 16)
        assert len(meta) == 2
        assert meta[0]["FrameID"] == 1

    def test_calibrated(self, hcc_path):
        data = read_hcc(hcc_path, calibrated=True)
        assert data.dtype == np.float32
        expected = np.float32(1000 * (2.0 ** -8) + 273.15)
        np.testing.assert_allclose(data, expected, rtol=1e-5)

    def test_calibrated_with_dtype(self, hcc_path):
        data = read_hcc(hcc_path, calibrated=True, dtype=np.float64)
        assert data.dtype == np.float64

    def test_dtype_cast(self, hcc_path):
        data = read_hcc(hcc_path, dtype=np.float32)
        assert data.dtype == np.float32
        np.testing.assert_allclose(data, 1000.0, rtol=1e-6)

    def test_frames_range(self, hcc_path_varied):
        data = read_hcc(hcc_path_varied, frames=range(0, 4, 2))
        assert data.shape == (2, 8, 16)
        np.testing.assert_array_equal(data[0], 100)
        np.testing.assert_array_equal(data[1], 300)

    def test_invalid_frames_type(self, hcc_path):
        with pytest.raises(TypeError, match="frames must be"):
            read_hcc(hcc_path, frames=[0, 1, 2])

    def test_calibrated_with_metadata(self, hcc_path):
        data, meta = read_hcc(hcc_path, calibrated=True, metadata=True)
        assert data.dtype == np.float32
        assert len(meta) == 3


# ======================================================================
# cli.py tests
# ======================================================================

class TestCLI:
    """Tests for CLI subcommands."""

    def test_cmd_info(self, hcc_path, capsys):
        args = argparse.Namespace(file=str(hcc_path))
        cmd_info(args)
        captured = capsys.readouterr()
        assert "16 x 8" in captured.out
        assert "3" in captured.out  # n_frames
        assert "2000" in captured.out  # frame rate
        assert "RT" in captured.out  # calibration mode

    def test_cmd_convert_basic(self, hcc_path, tmp_path):
        out_npy = tmp_path / "output.npy"
        args = argparse.Namespace(
            path=str(hcc_path),
            output=str(out_npy),
            dtype="uint16",
            calibrated=False,
            recursive=False,
            skip_existing=False,
            quiet=True,
        )
        cmd_convert(args)
        assert out_npy.exists()
        data = np.load(out_npy)
        assert data.shape == (3, 8, 16)
        assert data.dtype == np.uint16
        np.testing.assert_array_equal(data, 1000)

    def test_cmd_convert_calibrated(self, hcc_path, tmp_path):
        out_npy = tmp_path / "output_cal.npy"
        args = argparse.Namespace(
            path=str(hcc_path),
            output=str(out_npy),
            dtype="float32",
            calibrated=True,
            recursive=False,
            skip_existing=False,
            quiet=True,
        )
        cmd_convert(args)
        assert out_npy.exists()
        data = np.load(out_npy)
        assert np.issubdtype(data.dtype, np.floating)
        # Should be calibrated values, not raw
        assert data.mean() > 200  # offset ~273 dominates

    def test_cmd_convert_skip_existing(self, hcc_path, tmp_path, capsys):
        out_npy = hcc_path.with_suffix(".npy")
        # First conversion
        args = argparse.Namespace(
            path=str(hcc_path),
            output=str(out_npy),
            dtype="uint16",
            calibrated=False,
            recursive=False,
            skip_existing=False,
            quiet=True,
        )
        cmd_convert(args)
        assert out_npy.exists()
        mtime1 = out_npy.stat().st_mtime

        # Second conversion with skip_existing
        args2 = argparse.Namespace(
            path=str(hcc_path),
            output=str(out_npy),
            dtype="uint16",
            calibrated=False,
            recursive=False,
            skip_existing=True,
            quiet=False,
        )
        cmd_convert(args2)
        captured = capsys.readouterr()
        assert "SKIP" in captured.out
        # File should not have been rewritten
        assert out_npy.stat().st_mtime == mtime1

    def test_cmd_convert_default_output_path(self, hcc_path):
        """When no -o given, output goes next to input with .npy extension."""
        args = argparse.Namespace(
            path=str(hcc_path),
            output=None,
            dtype="uint16",
            calibrated=False,
            recursive=False,
            skip_existing=False,
            quiet=True,
        )
        cmd_convert(args)
        expected_out = hcc_path.with_suffix(".npy")
        assert expected_out.exists()
        data = np.load(expected_out)
        assert data.shape == (3, 8, 16)


# ======================================================================
# Edge cases
# ======================================================================

class TestEdgeCases:
    """Edge-case and boundary-condition tests."""

    def test_single_frame(self, single_frame_hcc):
        with HCCReader(single_frame_hcc) as hcc:
            assert hcc.n_frames == 1
            frames = hcc.read_frames()
            assert frames.shape == (1, 8, 16)
            np.testing.assert_array_equal(frames, 42)

    def test_truncated_last_frame_ignored(self, tmp_path):
        """File with 2 full frames + partial 3rd frame -> n_frames=2."""
        path = tmp_path / "truncated.hcc"
        w, h = 16, 8
        full = make_hcc(path, width=w, height=h, n_frames=2, pixel_fill=500)
        # Append a partial frame (just header, no pixel data)
        partial_hdr = bytearray(2 * w * 2)
        partial_hdr[0:2] = b"TC"
        partial_hdr[2] = 7
        partial_hdr[3] = 12
        struct.pack_into("<H", partial_hdr, 4, 2 * w * 2)
        struct.pack_into("<H", partial_hdr, 32, w)
        struct.pack_into("<H", partial_hdr, 34, h)
        with open(path, "ab") as f:
            f.write(partial_hdr)
            # Write only half the pixel data
            f.write(b"\x00" * (w * h))  # half of w*h*2

        with HCCReader(path) as hcc:
            assert hcc.n_frames == 2  # truncated frame ignored
            frames = hcc.read_frames()
            assert frames.shape == (2, 8, 16)
            np.testing.assert_array_equal(frames, 500)

    def test_multiple_frames_different_patterns(self, tmp_path):
        """Each frame gets a unique ramp pattern; verify exact values."""
        # width must be >= 12 so header (2*w*2) >= 48 bytes (prefix struct size)
        w, h, n = 16, 4, 5
        pixel_data = []
        for i in range(n):
            base = i * 1000
            arr = np.arange(base, base + w * h, dtype=np.uint16).reshape(h, w)
            pixel_data.append(arr)

        path = make_hcc(
            tmp_path / "ramp.hcc",
            width=w, height=h, n_frames=n,
            pixel_data=pixel_data,
        )

        with HCCReader(path) as hcc:
            frames = hcc.read_frames()
            assert frames.shape == (n, h, w)
            for i in range(n):
                np.testing.assert_array_equal(frames[i], pixel_data[i])

    def test_large_dimensions(self, tmp_path):
        """320x256, 2 frames -- make sure larger header/stride is correct."""
        w, h = 320, 256
        path = make_hcc(
            tmp_path / "large.hcc",
            width=w, height=h, n_frames=2,
            pixel_fill=55555,
        )
        with HCCReader(path) as hcc:
            assert hcc.width == 320
            assert hcc.height == 256
            assert hcc.n_frames == 2
            frames = hcc.read_frames()
            assert frames.shape == (2, 256, 320)
            np.testing.assert_array_equal(frames, 55555)

    def test_zero_pixel_value(self, tmp_path):
        """Ensure zero-valued pixels round-trip correctly."""
        path = make_hcc(tmp_path / "zeros.hcc", pixel_fill=0, n_frames=1)
        data = read_hcc(path)
        np.testing.assert_array_equal(data, 0)

    def test_max_pixel_value(self, tmp_path):
        """Ensure max uint16 (65535) round-trips correctly."""
        path = make_hcc(tmp_path / "maxval.hcc", pixel_fill=65535, n_frames=1)
        data = read_hcc(path)
        np.testing.assert_array_equal(data, 65535)

    def test_calibration_low_version(self, tmp_path):
        """Version < 10 with cal_mode < 1 should just cast to float, no cal."""
        path = make_hcc(
            tmp_path / "legacy.hcc",
            version=(5, 0),
            cal_mode=0,
            pixel_fill=500,
            n_frames=2,
        )
        with HCCReader(path) as hcc:
            cal = hcc.to_calibrated()
            # Should be plain float cast, no offset/exp applied
            np.testing.assert_allclose(cal, 500.0)

    def test_string_path(self, hcc_path):
        """HCCReader should accept string paths, not just Path objects."""
        with HCCReader(str(hcc_path)) as hcc:
            assert hcc.n_frames == 3

    def test_read_hcc_string_path(self, hcc_path):
        """read_hcc should accept string paths."""
        data = read_hcc(str(hcc_path))
        assert data.shape == (3, 8, 16)
