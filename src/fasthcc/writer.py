"""HCC file writer for Telops IR camera data.

Companion to reader.py — creates valid HCC files from numpy arrays.

Usage::

    from fasthcc import write_hcc

    # Write raw uint16 frames
    write_hcc("output.hcc", frames)

    # Write calibrated float data (inverse-calibrates to uint16)
    write_hcc("output.hcc", temps, calibration_mode=2,
              data_offset=273.15, data_exp=-8)

    # Round-trip: read, modify, write back
    frames, meta = read_hcc("input.hcc", metadata=True)
    frames[0] = modify(frames[0])
    write_hcc("output.hcc", frames, metadata=meta)

    # Streaming writer
    with HCCWriter("output.hcc", width=320, height=256) as w:
        for frame in source:
            w.write_frame(frame)
"""

import warnings
from pathlib import Path

import numpy as np

from .header import build_header, header_raw_size


# ======================================================================
# Internal helpers
# ======================================================================

def _float_to_raw(frames, data_offset, data_exp):
    """Convert calibrated float frames to raw uint16.

    Inverse of: calibrated = raw * 2^data_exp + data_offset

    Parameters
    ----------
    frames : np.ndarray
        Float array, shape (N, H, W).
    data_offset : float
        Calibration offset.
    data_exp : int
        Calibration exponent.

    Returns
    -------
    np.ndarray
        uint16 array, shape (N, H, W).
    """
    inv_scale = 2.0 ** (-data_exp)
    raw_float = (frames.astype(np.float64) - np.float64(data_offset)) * np.float64(inv_scale)
    raw_rounded = np.round(raw_float)

    lo, hi = raw_rounded.min(), raw_rounded.max()
    if lo < 0 or hi > 65535:
        n_bad = int(np.sum((raw_rounded < 0) | (raw_rounded > 65535)))
        warnings.warn(
            f"{n_bad} pixel(s) out of uint16 range [0, 65535] — "
            f"values will be clipped",
            stacklevel=3,
        )
        raw_rounded = np.clip(raw_rounded, 0, 65535)

    return raw_rounded.astype(np.uint16)


# ======================================================================
# Convenience function
# ======================================================================

def write_hcc(
    path,
    frames,
    *,
    calibration_mode=2,
    frame_rate=2000.0,
    data_offset=273.15,
    data_exp=-8,
    exposure_time=110.16,
    metadata=None,
    version=(12, 7),
):
    """Write frames to an HCC file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    frames : np.ndarray
        Frame data, shape (N, H, W).
        - uint16: written directly as raw pixel data.
        - floating: inverse-calibrated to uint16.
    calibration_mode : int
        0 = raw, 1 = NUC, 2 = RT. Default 2.
    frame_rate : float
        Acquisition frame rate in Hz. Default 2000.0.
    data_offset : float
        Calibration offset. Default 273.15.
    data_exp : int
        Calibration exponent. Default -8.
    exposure_time : float
        Exposure time in microseconds. Default 110.16.
    metadata : list[dict] or None
        Per-frame header overrides. If provided, len must equal N.
        Each dict can contain any header field name. File-level defaults
        (calibration_mode, frame_rate, etc.) are overridden by metadata
        values when present.
    version : tuple[int, int]
        Header version (major, minor). Default (12, 7).
    """
    if frames.ndim != 3:
        raise ValueError(
            f"frames must be 3-D (N, H, W) — got ndim={frames.ndim}"
        )

    if frames.shape[0] == 0:
        raise ValueError("frames must contain at least one frame")

    n_frames, height, width = frames.shape

    if metadata is not None and len(metadata) != n_frames:
        raise ValueError(
            f"metadata length ({len(metadata)}) must equal number of "
            f"frames ({n_frames})"
        )

    # Convert to raw uint16
    if np.issubdtype(frames.dtype, np.floating):
        if metadata is not None:
            # Check if calibration varies per frame
            offsets = [m.get('DataOffset', data_offset) for m in metadata]
            exps = [m.get('DataExp', data_exp) for m in metadata]
            if len(set(zip(offsets, exps))) == 1:
                raw = _float_to_raw(frames, offsets[0], exps[0])
            else:
                # Per-frame conversion
                raw = np.empty_like(frames, dtype=np.uint16)
                for i in range(n_frames):
                    raw[i] = _float_to_raw(
                        frames[i:i+1], offsets[i], exps[i]
                    )[0]
        else:
            raw = _float_to_raw(frames, data_offset, data_exp)
    elif frames.dtype == np.uint16:
        raw = frames
    else:
        raise TypeError(
            f"frames must be uint16 or float — got {frames.dtype}"
        )

    # Write file
    path = Path(path)
    with open(path, 'wb') as f:
        for i in range(n_frames):
            extra = dict(metadata[i]) if metadata is not None else {}

            # Determine per-frame calibration params
            cal_mode = extra.pop('CalibrationMode', calibration_mode)
            fr = extra.pop('AcquisitionFrameRate', frame_rate)
            exp_t = extra.pop('ExposureTime', exposure_time)
            d_off = extra.pop('DataOffset', data_offset)
            d_exp = extra.pop('DataExp', data_exp)
            fid = extra.pop('FrameID', i)

            # Strip fields that must be computed, not copied from metadata
            for key in ('Width', 'Height', 'ImageHeaderLength',
                        'DeviceXMLMajorVersion', 'DeviceXMLMinorVersion',
                        'Signature'):
                extra.pop(key, None)

            hdr = build_header(
                width, height,
                frame_id=fid,
                data_offset=d_off,
                data_exp=d_exp,
                calibration_mode=cal_mode,
                exposure_time=exp_t,
                frame_rate=fr,
                version=version,
                **extra,
            )
            f.write(hdr)
            f.write(raw[i].astype('<u2').tobytes())


# ======================================================================
# Streaming writer
# ======================================================================

class HCCWriter:
    """Incremental HCC file writer.

    Parameters
    ----------
    path : str or Path
        Output file path.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    calibration_mode : int
        Default 2.
    frame_rate : float
        Default 2000.0.
    data_offset : float
        Default 273.15.
    data_exp : int
        Default -8.
    exposure_time : float
        Exposure time in microseconds. Default 110.16.
    version : tuple[int, int]
        Default (12, 7).
    """

    __slots__ = (
        'path', 'width', 'height', 'calibration_mode', 'frame_rate',
        'data_offset', 'data_exp', 'exposure_time', 'version',
        '_file', '_frame_count',
    )

    def __init__(self, path, width, height, *, calibration_mode=2,
                 frame_rate=2000.0, data_offset=273.15, data_exp=-8,
                 exposure_time=110.16, version=(12, 7)):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.calibration_mode = calibration_mode
        self.frame_rate = frame_rate
        self.data_offset = data_offset
        self.data_exp = data_exp
        self.exposure_time = exposure_time
        self.version = version
        self._file = open(self.path, 'wb')
        self._frame_count = 0

    def write_frame(self, pixels, metadata=None):
        """Append a single frame.

        Parameters
        ----------
        pixels : np.ndarray
            Shape (H, W), dtype uint16 or float.
        metadata : dict or None
            Header field overrides for this frame.
        """
        if self._file is None:
            raise RuntimeError("HCCWriter is closed")

        if pixels.shape != (self.height, self.width):
            raise ValueError(
                f"Expected shape ({self.height}, {self.width}), "
                f"got {pixels.shape}"
            )

        # Convert to uint16 if float
        if np.issubdtype(pixels.dtype, np.floating):
            d_off = metadata.get('DataOffset', self.data_offset) if metadata else self.data_offset
            d_exp = metadata.get('DataExp', self.data_exp) if metadata else self.data_exp
            raw = _float_to_raw(pixels[np.newaxis], d_off, d_exp)[0]
        elif pixels.dtype == np.uint16:
            raw = pixels
        else:
            raise TypeError(
                f"pixels must be uint16 or float — got {pixels.dtype}"
            )

        extra = dict(metadata) if metadata else {}
        # Remove fields that are explicit parameters to avoid double-setting
        cal_mode = extra.pop('CalibrationMode', self.calibration_mode)
        fr = extra.pop('AcquisitionFrameRate', self.frame_rate)
        exp_t = extra.pop('ExposureTime', self.exposure_time)
        d_off = extra.pop('DataOffset', self.data_offset)
        d_exp = extra.pop('DataExp', self.data_exp)
        fid = extra.pop('FrameID', self._frame_count)

        # Strip fields that must be computed, not copied from metadata
        for key in ('Width', 'Height', 'ImageHeaderLength',
                    'DeviceXMLMajorVersion', 'DeviceXMLMinorVersion',
                    'Signature'):
            extra.pop(key, None)

        hdr = build_header(
            self.width, self.height,
            frame_id=fid,
            data_offset=d_off,
            data_exp=d_exp,
            calibration_mode=cal_mode,
            exposure_time=exp_t,
            frame_rate=fr,
            version=self.version,
            **extra,
        )
        self._file.write(hdr)
        self._file.write(raw.astype('<u2').tobytes())
        self._frame_count += 1

    def close(self):
        """Flush and close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    @property
    def n_frames_written(self):
        """Number of frames written so far."""
        return self._frame_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self):
        status = 'closed' if self._file is None else 'open'
        return (
            f"HCCWriter({self.path.name!r}, {self.width}x{self.height}, "
            f"{self._frame_count} frames written, {status})"
        )
