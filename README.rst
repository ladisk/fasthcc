fasthcc
=======

Fast, pure-Python reader for Telops HCC infrared camera files — reverse-engineered directly from the binary format, no SDK or TelopsToolbox required.

``fasthcc`` reads ``.hcc`` files produced by Telops FAST-series IR cameras and returns NumPy arrays.
The binary header format (V5–V12) was determined by analysis of the file structure and validated against real camera data.
It has a single dependency (numpy) and is designed as a drop-in replacement for TelopsToolbox's ``SequenceReaderP``,
which is slow and has compatibility bugs with numpy 2.x.

Licensed under the MIT License.


Benchmark
---------

This is the primary motivation for the package. Benchmark results from three examples on real recordings from a Telops FAST M3k camera are presented below. The results show a consistent ~100× speedup for raw reads and ~25× for calibrated reads.

**Test #1:** 128×132 resolution, 12,000 frames, 2000 Hz, RT calibration, 392.6 MB HCC file

=================================  ========  =========
Method                             Time      Speedup
=================================  ========  =========
``fasthcc`` (raw uint16)           0.22 s    **90×**
``fasthcc`` (calibrated float32)   0.73 s    **27×**
TelopsToolbox                      19.7 s    1× (baseline)
=================================  ========  =========

**Test #2:** 320×256 resolution, 6,350 frames, 1000 Hz, RT calibration, 999.9 MB HCC file

=================================  ========  =========
Method                             Time      Speedup
=================================  ========  =========
``fasthcc`` (raw uint16)           0.67 s    **105×**
``fasthcc`` (calibrated float32)   2.79 s    **25×**
TelopsToolbox                      70.9 s    1× (baseline)
=================================  ========  =========

**Test #3:** 320×256 resolution, 10,000 frames, 1000 Hz, RT calibration, 1.53 GB HCC file

=================================  ========  =========
Method                             Time      Speedup
=================================  ========  =========
``fasthcc`` (raw uint16)           0.93 s    **100×**
``fasthcc`` (calibrated float32)   3.21 s    **24×**
TelopsToolbox                      93.2 s    1× (baseline)
=================================  ========  =========

Why fasthcc is faster
~~~~~~~~~~~~~~~~~~~~~

**TelopsToolbox** creates two ``np.memmap`` objects per frame (header + pixels) in a Python
for-loop. For a 12 000-frame file that means 24 000 memmap instantiations — the loop alone
dominates the total read time, regardless of disk speed.

**fasthcc** memory-maps the entire file once using a single numpy structured dtype, extracting
all frames in one ``np.memmap()`` call. No Python-level frame loop, and files of any size are
handled without loading them entirely into RAM.


Ready-to-use arrays
~~~~~~~~~~~~~~~~~~~

fasthcc returns properly shaped ``(N, H, W)`` numpy arrays directly. TelopsToolbox returns
flattened pixel data that requires a separate ``form_image(header, ir_data)`` call to reshape.


Correctness
~~~~~~~~~~~

fasthcc produces bit-identical output to TelopsToolbox (raw uint16 exact match, calibrated
float32 exact match). Verified on RT-calibrated recordings.


Installation
------------

::

    pip install fasthcc

For development::

    git clone https://github.com/ladisk/fasthcc.git
    cd fasthcc
    pip install -e ".[dev]"


Usage
-----

Python API
~~~~~~~~~~

.. code-block:: python

    from fasthcc import read_hcc, HCCFile

    # One-shot read
    frames = read_hcc("recording.hcc")  # (n_frames, height, width) uint16

    # Calibrated (temperature in Kelvin)
    frames = read_hcc("recording.hcc", calibrated=True)  # float32

    # Read subset of frames
    frames = read_hcc("recording.hcc", frames=slice(0, 100))

    # With metadata
    frames, meta = read_hcc("recording.hcc", metadata=True)
    print(meta[0]["AcquisitionFrameRate"])  # 2000.0

    # Class-based access
    with HCCFile("recording.hcc") as hcc:
        print(hcc.width, hcc.height, hcc.n_frames)
        print(hcc.frame_rate, hcc.calibration_mode)
        subset = hcc.read_frames(0, 100)
        hcc.to_npy("output.npy")


CLI
~~~

Print file information::

    fasthcc info recording.hcc

Convert to NPY::

    fasthcc convert recording.hcc
    fasthcc convert recording.hcc --calibrated --dtype float32
    fasthcc convert folder/ -r --skip-existing


HCC file format
----------------

HCC is a binary format used by Telops infrared cameras. Each file contains a sequence of frames,
where each frame consists of a fixed-size header followed by raw pixel data.

- **Signature**: The first 2 bytes of each frame header are ``"TC"`` (ASCII).
- **Version**: Bytes 2–3 encode the minor and major header version numbers. Supported versions
  range from 5.x through 12.x.
- **Frame layout**: Each frame occupies ``header_size + width * height * 2`` bytes, where
  ``header_size = 2 * width * 2`` bytes (two "header lines" worth of uint16 values).
- **Pixel data**: ``width * height`` unsigned 16-bit integers, little-endian.
- **Calibration modes**: 0 = raw, 1 = NUC (non-uniformity corrected), 2 = RT (radiometric temperature).
- **RT calibration**: Temperature in Kelvin is recovered from raw pixel values via
  ``pixel * 2^DataExp + DataOffset``, where ``DataExp`` and ``DataOffset`` are stored in the
  per-frame header.


Limitations
-----------

- **Read-only.** fasthcc does not support writing HCC files.
- **Designed for Telops FAST-series cameras.** Other Telops camera models may use header versions
  that require version-specific adjustments.
- **Memory-mapped I/O.** Files are memory-mapped, so they can exceed available RAM. However,
  operations like ``to_calibrated()`` or ``read_frames()`` that return full numpy arrays will
  allocate output buffers proportional to the number of requested frames.
