"""
Comprehensive test suite for the fasthcc writer module.

Tests cover header building (roundtrip with parse_header), write_hcc
convenience function, HCCWriter streaming interface, and the internal
_float_to_raw calibration inversion helper.

All tests use synthetic data -- no real camera files are needed.
"""

import struct

import numpy as np
import pytest

from fasthcc.header import build_header, parse_header, header_raw_size
from fasthcc.reader import read_hcc, HCCReader
from fasthcc.writer import write_hcc, HCCWriter, _float_to_raw


# ======================================================================
# build_header() + parse_header() roundtrip tests
# ======================================================================

class TestBuildHeader:
    def test_roundtrip_default(self):
        """Build header with defaults, parse it back, verify key fields."""
        hdr = build_header(width=64, height=48)
        parsed = parse_header(hdr, offset=0)
        assert parsed['Signature'] == 'TC'
        assert parsed['Width'] == 64
        assert parsed['Height'] == 48
        assert parsed['DeviceXMLMajorVersion'] == 12
        assert parsed['DeviceXMLMinorVersion'] == 7
        assert parsed['CalibrationMode'] == 2
        assert abs(parsed['DataOffset'] - 273.15) < 0.01
        assert parsed['DataExp'] == -8
        assert abs(parsed['AcquisitionFrameRate'] - 2000.0) < 0.1
        assert abs(parsed['ExposureTime'] - 110.16) < 0.1

    def test_roundtrip_custom_fields(self):
        """Custom frame_id, calibration params."""
        hdr = build_header(width=32, height=16, frame_id=42,
                          data_offset=0.0, data_exp=0,
                          calibration_mode=0, frame_rate=500.0)
        parsed = parse_header(hdr, offset=0)
        assert parsed['FrameID'] == 42
        assert abs(parsed['DataOffset']) < 0.01
        assert parsed['DataExp'] == 0
        assert parsed['CalibrationMode'] == 0
        assert abs(parsed['AcquisitionFrameRate'] - 500.0) < 0.1

    def test_header_size(self):
        """Header size matches header_raw_size(width)."""
        for w in [16, 32, 64, 128, 320]:
            hdr = build_header(width=w, height=10)
            assert len(hdr) == header_raw_size(w)

    def test_version_v10(self):
        """V10 headers parse correctly."""
        hdr = build_header(width=64, height=48, version=(10, 0))
        parsed = parse_header(hdr, offset=0)
        assert parsed['DeviceXMLMajorVersion'] == 10
        assert parsed['DeviceXMLMinorVersion'] == 0

    def test_version_v5_prefix_only(self):
        """V5 headers only have prefix fields."""
        hdr = build_header(width=64, height=48, version=(5, 2))
        parsed = parse_header(hdr, offset=0)
        assert parsed['DeviceXMLMajorVersion'] == 5
        assert 'POSIXTime' not in parsed

    def test_extra_fields_tail(self):
        """Extra fields like POSIXTime are written to tail."""
        hdr = build_header(width=64, height=48, POSIXTime=1700000000,
                          SubSecondTime=500000)
        parsed = parse_header(hdr, offset=0)
        assert parsed['POSIXTime'] == 1700000000
        assert parsed['SubSecondTime'] == 500000

    def test_small_width_raises(self):
        """Width < 12 raises ValueError."""
        with pytest.raises(ValueError, match="width must be >= 12"):
            build_header(width=8, height=8)

    def test_small_width_no_tail(self):
        """Width=12 -> header=48 bytes, no room for tail."""
        hdr = build_header(width=12, height=8, version=(12, 7))
        assert len(hdr) == 48
        # Should still parse prefix correctly
        parsed = parse_header(hdr, offset=0)
        assert parsed['Width'] == 12
        # No tail fields since header too small
        assert 'POSIXTime' not in parsed


# ======================================================================
# _float_to_raw() calibration inversion tests
# ======================================================================

class TestFloatToRaw:
    def test_basic_inversion(self):
        """Forward then inverse calibration recovers original uint16."""
        raw_orig = np.array([[[1000, 2000], [3000, 4000]]], dtype=np.uint16)
        data_exp = -8
        data_offset = 273.15
        # Forward: calibrated = raw * 2^exp + offset
        scale = 2.0 ** data_exp
        calibrated = raw_orig.astype(np.float64) * scale + data_offset
        # Inverse
        recovered = _float_to_raw(calibrated, data_offset, data_exp)
        np.testing.assert_array_equal(recovered, raw_orig)

    def test_clip_warning(self):
        """Out-of-range values trigger warning and get clipped."""
        huge = np.array([[[1e10]]], dtype=np.float64)
        with pytest.warns(UserWarning, match="out of uint16 range"):
            raw = _float_to_raw(huge, 0.0, 0)
        assert raw[0, 0, 0] == 65535

    def test_negative_clip(self):
        """Negative results clip to 0."""
        neg = np.array([[[-100.0]]], dtype=np.float64)
        with pytest.warns(UserWarning, match="out of uint16 range"):
            raw = _float_to_raw(neg, 0.0, 0)
        assert raw[0, 0, 0] == 0


# ======================================================================
# write_hcc() convenience function tests
# ======================================================================

class TestWriteHcc:
    def test_roundtrip_uint16(self, tmp_path):
        """Write uint16 frames, read back, verify exact match."""
        path = tmp_path / "test.hcc"
        frames = np.random.randint(0, 65535, (5, 16, 32), dtype=np.uint16)
        write_hcc(path, frames)
        result = read_hcc(path)
        np.testing.assert_array_equal(result, frames)

    def test_roundtrip_calibrated(self, tmp_path):
        """Write float frames, read back calibrated, verify close match."""
        path = tmp_path / "test.hcc"
        data_offset = 273.15
        data_exp = -8
        # Create realistic temperature data
        temps = np.random.uniform(280.0, 350.0, (3, 16, 32)).astype(np.float32)
        write_hcc(path, temps, data_offset=data_offset, data_exp=data_exp)
        result = read_hcc(path, calibrated=True, dtype=np.float64)
        # Tolerance: 1 uint16 step = 2^(-8) ~ 0.004 K
        np.testing.assert_allclose(result, temps, atol=0.005)

    def test_single_frame(self, tmp_path):
        """Single frame write/read."""
        path = tmp_path / "single.hcc"
        frame = np.full((1, 8, 16), 42, dtype=np.uint16)
        write_hcc(path, frame)
        result = read_hcc(path)
        np.testing.assert_array_equal(result, frame)

    def test_metadata_roundtrip(self, tmp_path):
        """Write with metadata, read back metadata, verify fields preserved."""
        path = tmp_path / "meta.hcc"
        frames = np.full((2, 16, 32), 1000, dtype=np.uint16)
        meta = [
            {'FrameID': 10, 'POSIXTime': 1700000000, 'SubSecondTime': 100},
            {'FrameID': 11, 'POSIXTime': 1700000001, 'SubSecondTime': 200},
        ]
        write_hcc(path, frames, metadata=meta)
        result, result_meta = read_hcc(path, metadata=True)
        np.testing.assert_array_equal(result, frames)
        assert result_meta[0]['FrameID'] == 10
        assert result_meta[0]['POSIXTime'] == 1700000000
        assert result_meta[1]['FrameID'] == 11
        assert result_meta[1]['POSIXTime'] == 1700000001

    def test_frame_rate_preserved(self, tmp_path):
        """Frame rate is correctly written."""
        path = tmp_path / "rate.hcc"
        frames = np.full((1, 8, 16), 0, dtype=np.uint16)
        write_hcc(path, frames, frame_rate=500.0)
        with HCCReader(path) as hcc:
            assert abs(hcc.frame_rate - 500.0) < 0.1

    def test_calibration_mode_preserved(self, tmp_path):
        """Calibration mode is correctly written."""
        path = tmp_path / "cal.hcc"
        frames = np.full((1, 16, 32), 0, dtype=np.uint16)
        write_hcc(path, frames, calibration_mode=1)
        with HCCReader(path) as hcc:
            assert hcc.calibration_mode == 1

    def test_invalid_ndim_raises(self, tmp_path):
        """2D input raises ValueError."""
        path = tmp_path / "bad.hcc"
        with pytest.raises(ValueError, match="3-D"):
            write_hcc(path, np.zeros((8, 16), dtype=np.uint16))

    def test_invalid_dtype_raises(self, tmp_path):
        """Non-numeric dtype raises TypeError."""
        path = tmp_path / "bad.hcc"
        with pytest.raises(TypeError, match="uint16 or float"):
            write_hcc(path, np.zeros((1, 8, 16), dtype=np.int32))

    def test_metadata_length_mismatch(self, tmp_path):
        """Metadata list length != frame count raises ValueError."""
        path = tmp_path / "bad.hcc"
        frames = np.zeros((3, 8, 16), dtype=np.uint16)
        with pytest.raises(ValueError, match="metadata"):
            write_hcc(path, frames, metadata=[{}, {}])

    def test_string_path(self, tmp_path):
        """Accept string paths."""
        path = str(tmp_path / "str.hcc")
        write_hcc(path, np.zeros((1, 8, 16), dtype=np.uint16))
        result = read_hcc(path)
        assert result.shape == (1, 8, 16)

    def test_large_dimensions(self, tmp_path):
        """Write 320x256 frames."""
        path = tmp_path / "large.hcc"
        frames = np.random.randint(0, 65535, (2, 256, 320), dtype=np.uint16)
        write_hcc(path, frames)
        result = read_hcc(path)
        np.testing.assert_array_equal(result, frames)

    def test_zero_frames_raises(self, tmp_path):
        """Empty frame array raises ValueError."""
        path = tmp_path / "empty.hcc"
        with pytest.raises(ValueError, match="at least one frame"):
            write_hcc(path, np.zeros((0, 8, 16), dtype=np.uint16))

    def test_cross_version_metadata_roundtrip(self, tmp_path):
        """Metadata from V10 file written with default V12.7 should not crash."""
        src = tmp_path / "v10.hcc"
        write_hcc(src, np.full((1, 16, 32), 1000, dtype=np.uint16), version=(10, 0))
        data, meta = read_hcc(src, metadata=True)
        dst = tmp_path / "roundtrip.hcc"
        write_hcc(dst, data, metadata=meta)
        result = read_hcc(dst)
        np.testing.assert_array_equal(result, data)

    def test_version_v10(self, tmp_path):
        """V10 version roundtrips."""
        path = tmp_path / "v10.hcc"
        frames = np.full((2, 16, 32), 500, dtype=np.uint16)
        write_hcc(path, frames, version=(10, 0))
        with HCCReader(path) as hcc:
            assert hcc.header_version == (10, 0)
            result = hcc.read_frames()
        np.testing.assert_array_equal(result, frames)


# ======================================================================
# HCCWriter streaming writer tests
# ======================================================================

class TestHCCWriter:
    def test_streaming_write(self, tmp_path):
        """Write frames one at a time, read all back."""
        path = tmp_path / "stream.hcc"
        frames = [np.full((8, 16), i * 100, dtype=np.uint16) for i in range(5)]
        with HCCWriter(path, width=16, height=8) as w:
            for f in frames:
                w.write_frame(f)
            assert w.n_frames_written == 5
        result = read_hcc(path)
        assert result.shape == (5, 8, 16)
        for i in range(5):
            np.testing.assert_array_equal(result[i], frames[i])

    def test_context_manager_closes(self, tmp_path):
        """File is closed after context manager exit."""
        path = tmp_path / "ctx.hcc"
        with HCCWriter(path, width=16, height=8) as w:
            w.write_frame(np.zeros((8, 16), dtype=np.uint16))
        assert w._file is None

    def test_write_after_close_raises(self, tmp_path):
        """Writing after close raises RuntimeError."""
        path = tmp_path / "closed.hcc"
        w = HCCWriter(path, width=16, height=8)
        w.close()
        with pytest.raises(RuntimeError, match="closed"):
            w.write_frame(np.zeros((8, 16), dtype=np.uint16))

    def test_wrong_shape_raises(self, tmp_path):
        """Wrong frame shape raises ValueError."""
        path = tmp_path / "shape.hcc"
        with HCCWriter(path, width=16, height=8) as w:
            with pytest.raises(ValueError, match="shape"):
                w.write_frame(np.zeros((10, 20), dtype=np.uint16))

    def test_float_frames(self, tmp_path):
        """Float frames are inverse-calibrated."""
        path = tmp_path / "float.hcc"
        data_offset = 273.15
        data_exp = -8
        scale = 2.0 ** data_exp
        raw_val = 1000
        temp = raw_val * scale + data_offset
        with HCCWriter(path, width=16, height=8,
                       data_offset=data_offset, data_exp=data_exp) as w:
            w.write_frame(np.full((8, 16), temp, dtype=np.float64))
        result = read_hcc(path)
        assert result[0, 0, 0] == raw_val

    def test_repr(self, tmp_path):
        """Repr shows useful info."""
        path = tmp_path / "repr.hcc"
        with HCCWriter(path, width=16, height=8) as w:
            assert 'open' in repr(w)
            assert '16x8' in repr(w)
        assert 'closed' in repr(w)


# ======================================================================
# End-to-end roundtrip tests
# ======================================================================

class TestEndToEnd:
    def test_full_roundtrip_with_metadata(self, tmp_path):
        """Read -> modify -> write -> read, preserving metadata."""
        # Create source file
        src = tmp_path / "source.hcc"
        orig_frames = np.random.randint(100, 60000, (4, 16, 32), dtype=np.uint16)
        write_hcc(src, orig_frames, frame_rate=1000.0)

        # Read with metadata
        frames, meta = read_hcc(src, metadata=True)

        # Modify one frame
        frames[1] = frames[1] + 1

        # Write back
        dst = tmp_path / "modified.hcc"
        write_hcc(dst, frames, metadata=meta)

        # Read again and verify
        result, result_meta = read_hcc(dst, metadata=True)
        np.testing.assert_array_equal(result[0], orig_frames[0])
        np.testing.assert_array_equal(result[1], orig_frames[1] + 1)
        np.testing.assert_array_equal(result[2], orig_frames[2])
        assert abs(result_meta[0]['AcquisitionFrameRate'] - 1000.0) < 0.1

    def test_calibrated_roundtrip(self, tmp_path):
        """Write calibrated -> read calibrated -> verify."""
        path = tmp_path / "cal.hcc"
        data_offset = 273.15
        data_exp = -8
        # Generate temperature field
        temps = np.linspace(280.0, 340.0, 16 * 32).reshape(1, 16, 32).astype(np.float32)
        write_hcc(path, temps, data_offset=data_offset, data_exp=data_exp,
                  calibration_mode=2)
        result = read_hcc(path, calibrated=True, dtype=np.float64)
        np.testing.assert_allclose(result, temps, atol=0.005)
