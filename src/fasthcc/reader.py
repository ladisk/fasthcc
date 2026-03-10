"""
Fast HCC file reader for Telops IR camera data.

Performance strategy:
- Memory-mapped file via np.memmap (handles any file size, near-zero RSS)
- numpy structured dtype for zero-copy frame extraction
- struct.unpack_from() for header parsing (no intermediate copies)

Usage::

    # Quick one-liner
    frames = read_hcc("recording.hcc")                     # (N, H, W) uint16
    frames, meta = read_hcc("recording.hcc", metadata=True) # + per-frame headers
    temps = read_hcc("recording.hcc", calibrated=True)      # float32 calibrated

    # Class-based for repeated access
    with HCCFile("recording.hcc") as hcc:
        print(hcc.n_frames, hcc.width, hcc.height)
        subset = hcc.read_frames(start=10, stop=20)
        meta = hcc.read_metadata(start=0, stop=5)
        hcc.to_npy("output.npy", dtype=np.float32)
"""

from pathlib import Path

import numpy as np

from .header import (
    detect_version,
    frame_stride,
    header_raw_size,
    parse_header,
)


class HCCFile:
    """Fast reader for Telops HCC files.

    Uses memory-mapped I/O with numpy structured dtypes for zero-copy
    frame extraction.  Handles files of any size without loading them
    entirely into RAM.

    Parameters
    ----------
    path : str or Path
        Path to the .hcc file.

    Attributes
    ----------
    path : Path
        Resolved file path.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    n_frames : int
        Number of frames in the file.
    header_version : tuple[int, int]
        (major, minor) header version.
    calibration_mode : int
        0 = raw, 1 = NUC, 2 = RT (radiometric temperature).
    frame_rate : float
        Acquisition frame rate in Hz.
    data_offset : float
        Per-pixel additive offset for calibration denormalization.
    data_exp : int
        Per-pixel exponent for calibration (multiply by 2**data_exp).
    """

    __slots__ = (
        'path', 'width', 'height', 'n_frames', 'header_version',
        'calibration_mode', 'frame_rate', 'data_offset', 'data_exp',
        '_raw', '_header_raw_size', '_frame_stride', '_first_header',
        '_block_dtype', '_blocks',
    )

    def __init__(self, path):
        self.path = Path(path)
        file_size = self.path.stat().st_size

        if file_size < 48:
            raise ValueError(
                f"File too small to contain an HCC header ({file_size} bytes)"
            )

        # Read just the first 512 bytes to parse the header and get geometry
        with open(self.path, 'rb') as f:
            first_bytes = f.read(512)

        self._first_header = parse_header(first_bytes, offset=0)
        h = self._first_header

        self.width = h['Width']
        self.height = h['Height']

        if self.width == 0 or self.height == 0:
            raise ValueError(
                f"Invalid image dimensions: {self.width}x{self.height}"
            )

        self.header_version = (h['DeviceXMLMajorVersion'],
                               h['DeviceXMLMinorVersion'])
        self.calibration_mode = h['CalibrationMode']
        self.frame_rate = h['AcquisitionFrameRate']
        self.data_offset = h['DataOffset']
        self.data_exp = h['DataExp']

        # Geometry
        self._header_raw_size = header_raw_size(self.width)
        self._frame_stride = frame_stride(self.width, self.height)

        self.n_frames = file_size // self._frame_stride

        if self.n_frames == 0:
            raise ValueError(
                f"File contains no complete frames "
                f"(file={file_size} bytes, stride={self._frame_stride})"
            )

        # Build structured dtype for zero-copy extraction
        self._block_dtype = np.dtype([
            ('header', f'V{self._header_raw_size}'),
            ('pixels', '<u2', (self.height, self.width)),
        ])

        # Memory-map the file with the structured dtype
        self._blocks = np.memmap(
            self.path, dtype=self._block_dtype, mode='r',
            shape=(self.n_frames,),
        )

        # Keep a raw bytes reference for header parsing (memmap as uint8)
        self._raw = np.memmap(self.path, dtype=np.uint8, mode='r')

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Release the memory-mapped buffers."""
        if self._blocks is not None:
            del self._blocks
            self._blocks = None
        if self._raw is not None:
            del self._raw
            self._raw = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frames(self, start=0, stop=None):
        """Return pixel data as a contiguous (N, H, W) uint16 array.

        Parameters
        ----------
        start : int
            First frame index (inclusive). Default 0.
        stop : int or None
            Last frame index (exclusive). Default None = all remaining.

        Returns
        -------
        np.ndarray
            Shape (n, height, width), dtype uint16.
        """
        if self._blocks is None:
            raise RuntimeError("HCCFile is closed")
        sliced = self._blocks[start:stop]['pixels']
        return sliced.copy()  # contiguous copy, releases view on raw buffer

    def read_metadata(self, start=0, stop=None):
        """Parse per-frame headers for the specified range.

        Parameters
        ----------
        start : int
            First frame index (inclusive). Default 0.
        stop : int or None
            Last frame index (exclusive). Default None = all remaining.

        Returns
        -------
        list[dict]
            One dict per frame with all parsed header fields.
        """
        if self._blocks is None:
            raise RuntimeError("HCCFile is closed")

        if stop is None:
            stop = self.n_frames
        stop = min(stop, self.n_frames)

        meta = []
        for i in range(start, stop):
            offset = i * self._frame_stride
            hdr_bytes = bytes(self._raw[offset:offset + self._header_raw_size])
            hdr = parse_header(hdr_bytes, offset=0)
            meta.append(hdr)
        return meta

    def to_calibrated(self, dtype=np.float32):
        """Return denormalized calibrated data for ALL frames.

        Applies per-frame DataOffset and DataExp from each frame's header::

            real_value = pixel * 2**DataExp + DataOffset

        A fast path is used when calibration parameters are uniform across
        all frames (checked by sampling the first, middle, and last frame
        headers).  If a mid-sequence recalibration changes DataOffset/DataExp
        for only a few frames, those frames may receive incorrect calibration.

        Parameters
        ----------
        dtype : numpy dtype
            Output dtype. Default float32.

        Returns
        -------
        np.ndarray
            Shape (n_frames, height, width), dtype as specified.
        """
        if self._blocks is None:
            raise RuntimeError("HCCFile is closed")

        major = self.header_version[0]
        if major < 10 or self.calibration_mode < 1:
            # No calibration info — just cast to float
            return self._blocks['pixels'].astype(dtype).copy()

        # Fast path: if all frames share the same DataOffset/DataExp
        # (very common), do a single vectorized operation.
        # Check a sample of frames to decide.
        n = self.n_frames
        sample_indices = [0]
        if n > 1:
            sample_indices.append(n - 1)
        if n > 10:
            sample_indices.append(n // 2)

        offsets_exps = set()
        for idx in sample_indices:
            off = idx * self._frame_stride
            hdr_bytes = bytes(self._raw[off:off + self._header_raw_size])
            hdr = parse_header(hdr_bytes, offset=0)
            offsets_exps.add((hdr['DataOffset'], hdr['DataExp']))

        if len(offsets_exps) == 1:
            # Uniform calibration — single vectorized pass
            data_offset, data_exp = offsets_exps.pop()
            scale = 2.0 ** data_exp
            pixels = self._blocks['pixels']
            result = pixels.astype(dtype) * dtype(scale) + dtype(data_offset)
            return result

        # Slow path: per-frame calibration
        result = np.empty((n, self.height, self.width), dtype=dtype)
        for i in range(n):
            off = i * self._frame_stride
            hdr_bytes = bytes(self._raw[off:off + self._header_raw_size])
            hdr = parse_header(hdr_bytes, offset=0)
            scale = dtype(2.0 ** hdr['DataExp'])
            data_off = dtype(hdr['DataOffset'])
            result[i] = self._blocks[i]['pixels'].astype(dtype) * scale + data_off

        return result

    def to_npy(self, path, dtype=None):
        """Save frame data to a .npy file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        dtype : numpy dtype or None
            If a float dtype is given and calibration data is available,
            denormalization is applied. If None, raw uint16 is saved.
        """
        if dtype is not None and np.issubdtype(dtype, np.floating):
            data = self.to_calibrated(dtype=dtype)
        else:
            data = self.read_frames()
            if dtype is not None:
                data = data.astype(dtype)
        np.save(path, data)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        status = 'closed' if self._blocks is None else 'open'
        return (
            f"HCCFile({self.path.name!r}, {self.n_frames} frames, "
            f"{self.width}x{self.height}, "
            f"v{self.header_version[0]}.{self.header_version[1]}, "
            f"cal={self.calibration_mode}, {status})"
        )


# ======================================================================
# Convenience function
# ======================================================================

def read_hcc(path, frames=None, metadata=False, calibrated=False, dtype=None):
    """One-shot HCC file reader.

    Parameters
    ----------
    path : str or Path
        Path to the .hcc file.
    frames : slice, range, or None
        Frame selection. None = all frames.
        Examples: ``slice(10, 20)``, ``range(0, 100, 2)``.
    metadata : bool
        If True, also parse per-frame headers and return them.
    calibrated : bool
        If True, apply denormalization (pixel * 2^DataExp + DataOffset).
        Forces float32 output unless *dtype* specifies otherwise.
    dtype : numpy dtype or None
        Output dtype. Default: uint16 (or float32 if calibrated=True).

    Returns
    -------
    np.ndarray or tuple[np.ndarray, list[dict]]
        If metadata=False: array of shape (N, H, W).
        If metadata=True: (array, list_of_header_dicts).
    """
    with HCCFile(path) as hcc:
        # Determine frame range
        start, stop, step = _resolve_frame_selection(frames, hcc.n_frames)

        if calibrated:
            if dtype is None:
                dtype = np.float32
            # Use to_calibrated() which checks per-frame calibration
            all_cal = hcc.to_calibrated(dtype=dtype)
            data = all_cal[start:stop]
            if step is not None and step != 1:
                data = data[::step].copy()
        else:
            if step is not None and step != 1:
                # Non-unit step: read all then slice
                all_frames = hcc.read_frames(start=start, stop=stop)
                # Adjust for the relative indexing after start
                data = all_frames[::step].copy()
            else:
                data = hcc.read_frames(start=start, stop=stop)

            if dtype is not None:
                data = data.astype(dtype)

        if metadata:
            # Parse headers for the selected frames
            if step is not None and step != 1:
                meta = []
                for i in range(start, stop if stop is not None else hcc.n_frames, step):
                    off = i * hcc._frame_stride
                    hdr_bytes = bytes(hcc._raw[off:off + hcc._header_raw_size])
                    hdr = parse_header(hdr_bytes, offset=0)
                    meta.append(hdr)
            else:
                meta = hcc.read_metadata(start=start, stop=stop)
            return data, meta

        return data


def _resolve_frame_selection(frames, n_frames):
    """Convert frames argument to (start, stop, step).

    Returns
    -------
    tuple[int, int | None, int | None]
    """
    if frames is None:
        return 0, None, None

    if isinstance(frames, slice):
        start, stop, step = frames.indices(n_frames)
        if step < 0:
            raise ValueError("Negative step is not supported for frame selection")
        return start, stop, step

    if isinstance(frames, range):
        if frames.step < 0:
            raise ValueError("Negative step is not supported for frame selection")
        start = frames.start if frames.start >= 0 else max(0, n_frames + frames.start)
        stop = frames.stop if frames.stop >= 0 else max(0, n_frames + frames.stop)
        step = frames.step
        return start, stop, step

    raise TypeError(
        f"frames must be None, slice, or range — got {type(frames).__name__}"
    )
