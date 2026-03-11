"""
HCC header version detection and field parsing for Telops IR camera files.

The HCC format stores per-frame headers followed by uint16 pixel data.
Header size per frame = 2 * width * 2 bytes ("2 header lines").
The actual meaningful header content length is given by ImageHeaderLength (offset 4).

Supports:
- V10.x: Original field layout (TriggerDelay as float32)
- V12.0: Extended fields (TriggerDelay as float32, adds ADCReadout + filter wheel)
- V12.1+: Same as V12.0 but TriggerDelay as uint32
- V5-V9: Fallback — parses core fields only (signature, version, dims, calibration)
"""

import struct

# ---------------------------------------------------------------------------
# Pre-compiled struct formats (all little-endian)
# ---------------------------------------------------------------------------

# Common prefix: offsets 0-47 (shared by V10+)
# 0:  2s   Signature ("TC")
# 2:  B    DeviceXMLMinorVersion
# 3:  B    DeviceXMLMajorVersion
# 4:  H    ImageHeaderLength
# 6:  2x   reserved
# 8:  I    FrameID
# 12: f    DataOffset
# 16: b    DataExp
# 17: 7x   reserved
# 24: I    ExposureTime (raw, /100)
# 28: B    CalibrationMode
# 29: B    BPRApplied
# 30: B    FrameBufferMode
# 31: B    CalibrationBlockIndex
# 32: H    Width
# 34: H    Height
# 36: H    OffsetX
# 38: H    OffsetY
# 40: B    ReverseX
# 41: B    ReverseY
# 42: B    TestImageSelector
# 43: B    SensorWellDepth
# 44: I    AcquisitionFrameRate (raw, /1000)

_ST_PREFIX = struct.Struct('<2s BB H 2x I f b 7x I BBBB HH HH BBBB I')
_PREFIX_FIELDS = (
    'Signature',
    'DeviceXMLMinorVersion',
    'DeviceXMLMajorVersion',
    'ImageHeaderLength',
    'FrameID',
    'DataOffset',
    'DataExp',
    'ExposureTime',
    'CalibrationMode',
    'BPRApplied',
    'FrameBufferMode',
    'CalibrationBlockIndex',
    'Width',
    'Height',
    'OffsetX',
    'OffsetY',
    'ReverseX',
    'ReverseY',
    'TestImageSelector',
    'SensorWellDepth',
    'AcquisitionFrameRate',
)

# ---- V10 tail (offset 48 onwards) ----
# 48: f    TriggerDelay (float32)
# 52: B    TriggerMode
# 53: B    TriggerSource
# 54: B    IntegrationMode
# 55: 1x   reserved
# 56: B    AveragingNumber
# 57: 2x   reserved
# 59: B    ExposureAuto
# 60: f    AECResponseTime
# 64: f    AECImageFraction
# 68: f    AECTargetWellFilling
# 72: 28x  reserved
# 100: I   POSIXTime
# 104: I   SubSecondTime

_ST_V10_TAIL = struct.Struct('<f BBB 1x B 2x B f f f 28x I I')
_V10_TAIL_FIELDS = (
    'TriggerDelay',
    'TriggerMode',
    'TriggerSource',
    'IntegrationMode',
    'AveragingNumber',
    'ExposureAuto',
    'AECResponseTime',
    'AECImageFraction',
    'AECTargetWellFilling',
    'POSIXTime',
    'SubSecondTime',
)
_V10_TAIL_OFFSET = 48

# ---- V12.0 tail (offset 48 onwards, TriggerDelay = float32) ----
# 48: f    TriggerDelay (float32)
# 52: B    TriggerMode
# 53: B    TriggerSource
# 54: B    IntegrationMode
# 55: 1x   reserved
# 56: B    AveragingNumber
# 57: h    ADCReadout (int16) — V12 adds this
# 59: B    ExposureAuto
# 60: f    AECResponseTime
# 64: f    AECImageFraction
# 68: f    AECTargetWellFilling
# 72: 3x   reserved
# 75: B    FWMode
# 76: H    FWSpeedSetpoint
# 78: H    FWSpeed
# 80: 20x  reserved
# 100: I   POSIXTime
# 104: I   SubSecondTime

_ST_V12_0_TAIL = struct.Struct('<f BBB 1x B h B f f f 3x B H H 20x I I')
_V12_0_TAIL_FIELDS = (
    'TriggerDelay',
    'TriggerMode',
    'TriggerSource',
    'IntegrationMode',
    'AveragingNumber',
    'ADCReadout',
    'ExposureAuto',
    'AECResponseTime',
    'AECImageFraction',
    'AECTargetWellFilling',
    'FWMode',
    'FWSpeedSetpoint',
    'FWSpeed',
    'POSIXTime',
    'SubSecondTime',
)
_V12_0_TAIL_OFFSET = 48

# ---- V12.1+ tail (offset 48 onwards, TriggerDelay = uint32) ----
_ST_V12_1_TAIL = struct.Struct('<I BBB 1x B h B f f f 3x B H H 20x I I')
_V12_1_TAIL_FIELDS = (
    'TriggerDelay',
    'TriggerMode',
    'TriggerSource',
    'IntegrationMode',
    'AveragingNumber',
    'ADCReadout',
    'ExposureAuto',
    'AECResponseTime',
    'AECImageFraction',
    'AECTargetWellFilling',
    'FWMode',
    'FWSpeedSetpoint',
    'FWSpeed',
    'POSIXTime',
    'SubSecondTime',
)
_V12_1_TAIL_OFFSET = 48

# Minimal struct for legacy versions (V5-V9): just the first 48 bytes
# Same as prefix, works because prefix is 48 bytes.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_version(buf, offset=0):
    """Detect HCC header version from raw bytes.

    Parameters
    ----------
    buf : bytes or bytearray
        Buffer containing at least 6 bytes from header start.
    offset : int
        Byte offset into *buf* where the header begins.

    Returns
    -------
    tuple[int, int]
        (major, minor) version numbers.

    Raises
    ------
    ValueError
        If the signature bytes are not "TC".
    """
    if len(buf) - offset < 6:
        raise ValueError(
            f"Buffer too short to read header signature+version "
            f"(need 6 bytes, got {len(buf) - offset})"
        )
    sig = buf[offset:offset + 2]
    if sig != b'TC':
        raise ValueError(
            f"Invalid HCC signature: expected b'TC', got {sig!r}"
        )
    minor = buf[offset + 2]
    major = buf[offset + 3]
    return (major, minor)


def parse_header(buf, offset=0):
    """Parse a single HCC frame header from raw bytes.

    Parameters
    ----------
    buf : bytes or bytearray
        Buffer containing the full header (at least *ImageHeaderLength* bytes
        from *offset*).
    offset : int
        Byte offset into *buf* where this frame's header begins.

    Returns
    -------
    dict
        Parsed header fields. Always includes at minimum:
        Signature, DeviceXMLMajorVersion, DeviceXMLMinorVersion,
        ImageHeaderLength, Width, Height, CalibrationMode, FrameID,
        DataOffset, DataExp, AcquisitionFrameRate.

        For V10+ headers, also includes version-specific fields like
        POSIXTime, SubSecondTime, TriggerDelay, etc.

    Raises
    ------
    ValueError
        If signature is invalid or buffer is too short.
    """
    major, minor = detect_version(buf, offset)

    available = len(buf) - offset

    # For all versions, try to parse the common 48-byte prefix
    if available < _ST_PREFIX.size:
        raise ValueError(
            f"Buffer too short for common header prefix "
            f"(need {_ST_PREFIX.size} bytes, got {available})"
        )

    prefix_values = _ST_PREFIX.unpack_from(buf, offset)
    result = dict(zip(_PREFIX_FIELDS, prefix_values))

    # Decode signature to string
    result['Signature'] = result['Signature'].decode('ascii', errors='replace')

    # Post-process prefix fields
    result['ExposureTime'] = result['ExposureTime'] / 100.0
    result['AcquisitionFrameRate'] = result['AcquisitionFrameRate'] / 1000.0

    # Legacy versions (< 10): return prefix-only fields
    if major < 10:
        return result

    # V10+ : parse version-dependent tail
    if major in (10, 11):
        _parse_tail(buf, offset, result, _ST_V10_TAIL, _V10_TAIL_FIELDS,
                     _V10_TAIL_OFFSET)
    elif major == 12 and minor == 0:
        _parse_tail(buf, offset, result, _ST_V12_0_TAIL, _V12_0_TAIL_FIELDS,
                     _V12_0_TAIL_OFFSET)
    elif major >= 12:
        # V12.1+ and any future versions — use uint32 TriggerDelay layout
        _parse_tail(buf, offset, result, _ST_V12_1_TAIL, _V12_1_TAIL_FIELDS,
                     _V12_1_TAIL_OFFSET)

    return result


def build_header(
    width,
    height,
    frame_id=0,
    data_offset=273.15,
    data_exp=-8,
    calibration_mode=2,
    exposure_time=110.16,
    frame_rate=2000.0,
    version=(12, 7),
    **extra_fields,
):
    """Construct a binary HCC frame header.

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    frame_id : int
        Frame counter. Default 0.
    data_offset : float
        Calibration additive offset. Default 273.15.
    data_exp : int
        Calibration exponent. Default -8.
    calibration_mode : int
        0 = raw, 1 = NUC, 2 = RT. Default 2.
    exposure_time : float
        Exposure time in microseconds. Default 110.16.
    frame_rate : float
        Acquisition frame rate in Hz. Default 2000.0.
    version : tuple[int, int]
        Header version (major, minor). Default (12, 7).
    **extra_fields
        Additional header fields (e.g., POSIXTime, SubSecondTime,
        TriggerDelay). Unknown field names are silently ignored.

    Returns
    -------
    bytes
        Complete header buffer of size ``header_raw_size(width)``.
    """
    if width < 12:
        raise ValueError(
            f"width must be >= 12 (header needs at least 48 bytes), got {width}"
        )
    major, minor = version
    hdr_size = header_raw_size(width)
    buf = bytearray(hdr_size)

    # -- Build prefix values --------------------------------------------------
    prefix_defaults = {
        'Signature': b'TC',
        'DeviceXMLMinorVersion': minor,
        'DeviceXMLMajorVersion': major,
        'ImageHeaderLength': hdr_size,
        'FrameID': frame_id,
        'DataOffset': float(data_offset),
        'DataExp': int(data_exp),
        'ExposureTime': int(round(exposure_time * 100)),
        'CalibrationMode': int(calibration_mode),
        'BPRApplied': 0,
        'FrameBufferMode': 0,
        'CalibrationBlockIndex': 0,
        'Width': width,
        'Height': height,
        'OffsetX': 0,
        'OffsetY': 0,
        'ReverseX': 0,
        'ReverseY': 0,
        'TestImageSelector': 0,
        'SensorWellDepth': 0,
        'AcquisitionFrameRate': int(round(frame_rate * 1000)),
    }

    # Apply extra_fields overrides for prefix fields
    for key in _PREFIX_FIELDS:
        if key in extra_fields:
            val = extra_fields[key]
            if key == 'ExposureTime':
                val = int(round(val * 100))
            elif key == 'AcquisitionFrameRate':
                val = int(round(val * 1000))
            elif key == 'Signature':
                if isinstance(val, str):
                    val = val.encode('ascii')
            prefix_defaults[key] = val

    prefix_values = tuple(prefix_defaults[k] for k in _PREFIX_FIELDS)
    _ST_PREFIX.pack_into(buf, 0, *prefix_values)

    # -- Build tail values (V10+) ---------------------------------------------
    if major in (10, 11):
        tail_struct, tail_fields = _ST_V10_TAIL, _V10_TAIL_FIELDS
    elif major == 12 and minor == 0:
        tail_struct, tail_fields = _ST_V12_0_TAIL, _V12_0_TAIL_FIELDS
    elif major >= 12:
        tail_struct, tail_fields = _ST_V12_1_TAIL, _V12_1_TAIL_FIELDS
    else:
        tail_struct, tail_fields = None, None

    if (
        tail_struct is not None
        and major >= 10
        and hdr_size >= _ST_PREFIX.size + tail_struct.size
    ):
        tail_defaults = {name: 0 for name in tail_fields}
        for name in tail_fields:
            if name in extra_fields:
                tail_defaults[name] = extra_fields[name]
        # V12.1+ packs TriggerDelay as uint32; coerce float from V10 metadata
        if tail_struct is _ST_V12_1_TAIL and 'TriggerDelay' in tail_defaults:
            td = tail_defaults['TriggerDelay']
            if isinstance(td, float):
                tail_defaults['TriggerDelay'] = int(round(td))
        tail_values = tuple(tail_defaults[k] for k in tail_fields)
        tail_struct.pack_into(buf, _ST_PREFIX.size, *tail_values)

    return bytes(buf)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_tail(buf, base_offset, result, st, fields, tail_offset):
    """Unpack a tail struct into *result*, silently skipping if buffer is too short."""
    start = base_offset + tail_offset
    available = len(buf) - start
    if available < st.size:
        # Header is truncated — parse what we can with the prefix
        return
    values = st.unpack_from(buf, start)
    for name, val in zip(fields, values):
        result[name] = val


def header_raw_size(width):
    """Return the raw header size in bytes for a given image width.

    The HCC convention is "2 header lines": 2 * width * 2 bytes.

    Parameters
    ----------
    width : int
        Image width in pixels.

    Returns
    -------
    int
        Header size in bytes.
    """
    return 2 * width * 2


def frame_stride(width, height):
    """Return the total byte length of one frame (header + pixels).

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.

    Returns
    -------
    int
        Bytes per frame.
    """
    return header_raw_size(width) + width * height * 2
