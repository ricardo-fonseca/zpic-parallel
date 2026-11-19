"""
visxd - Quick-look visualization of ZDF simulation output.

The plotting routines live in ``visxd._core``; this file re-exports the public
API so that ``import visxd`` works as before. Geometry specific extensions are
provided as submodules, e.g. ``visxd.cyl`` for quasi-3D (cylindrical modes)
simulations.
"""

from ._core import *            # noqa: F401,F403
from ._core import __all__      # noqa: F401
from ._core import __doc__      # noqa: F401
