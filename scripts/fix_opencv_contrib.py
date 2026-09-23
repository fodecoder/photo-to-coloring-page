"""Reinstall opencv-contrib-python to restore cv2.ximgproc after a conflicting co-install.

The `lineart`/`lineart_raster` optional extras pull in `controlnet_aux`,
which depends on `opencv-python-headless`. That package and
`opencv-contrib-python` both install their compiled extension into the
same `cv2/` folder, so whichever gets installed (or reinstalled) *last*
silently overwrites the other's files. When `opencv-python-headless` wins,
`cv2.ximgproc`'s compiled algorithms disappear -- the submodule still
imports fine, but every real function on it raises `AttributeError`.

Symptom this fixes::

    AttributeError: module 'cv2.ximgproc' has no attribute 'thinning'
    AttributeError: module 'cv2.ximgproc' has no attribute 'rollingGuidanceFilter'

which breaks the `chained` (default CLI style), `region`, and
`skeleton_redraw` engines, plus the `paths_from_mask`/`hysteresis_centerline`
helpers every vector-producing engine relies on.

Run this any time you install or upgrade an extra that pulls in
`controlnet_aux` (`lineart`, `lineart_raster`), or whenever you see the
`AttributeError` above.

Usage
-----
::

    python scripts/fix_opencv_contrib.py
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    """Reinstall opencv-contrib-python, without touching its own dependencies.

    ``--no-deps`` is deliberate: reinstalling normally would just let pip
    resolve `opencv-python-headless` back in if something else in the
    environment still requires it, recreating the exact conflict this
    script exists to fix.

    Returns
    -------
    int
        The underlying ``pip install`` command's exit code.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "opencv-contrib-python",
        ],
        check=False,
    )
    if result.returncode == 0:
        print(
            "\nopencv-contrib-python reinstalled. Verify with:\n"
            '  python -c "import cv2; cv2.ximgproc.thinning; '
            'cv2.ximgproc.rollingGuidanceFilter"\n'
            "(should print nothing / not raise AttributeError)."
        )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
