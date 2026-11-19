"""
visxd.cyl - visxd extensions for quasi-3D (cylindrical modes) simulations.

Drop-in replacement for ``visxd``: every public ``visxd`` function is
available here, so a script can switch geometry by changing only the import::

    import visxd.cyl as vis

    vis.grid2d("EMF/E0z-000100.zdf")                  # same as visxd.grid2d
    vis.part2D("PARTICLES/electrons/particles-electrons-000100.zdf")   # (z, r)
    vis.part2D(filename, qx="x", qy="y")               # transverse plane

Differences from ``visxd`` (particle data only):

* Particle scatter plots default to ``qx = "z"`` and ``qy = "r"``.
* Cartesian transverse positions are available as derived quantities:
  ``x = r cos θ`` and ``y = r sin θ``, computed from the ``r``, ``cosθ`` and
  ``sinθ`` quantities in the file. They can be used anywhere a particle
  quantity is accepted (``part2D``, ``histogram``, ``plot_data``).

Derived quantities are only added when the file has the required quantities
and does not already contain a quantity with the same name.
"""


from ._core import *            # noqa: F401,F403  (re-export the visxd API)
from ._core import __all__
from ._core import (
    _read as _core_read,
    _part2D, _histogram, _plot_grid, _pop_iter,
)

# -----------------------------------------------------------------------------
# Derived particle quantities
# -----------------------------------------------------------------------------

# name : ( required quantities, function, label, quantity whose units are used )
_DERIVED = {
    "x": (("r", "cosθ"), lambda p: p["r"] * p["cosθ"], "x", "r"),
    "y": (("r", "sinθ"), lambda p: p["r"] * p["sinθ"], "y", "r"),
}


def _add_derived(particles, info):
    """Add the derived quantities to particle data read from a ZDF file.

    `particles` (dict of arrays) and `info.particles` (quants, qlabels,
    qunits) are updated in place, so the derived quantities behave exactly
    like quantities stored in the file.
    """
    pinfo = info.particles
    for q, (requires, f, label, units_from) in _DERIVED.items():
        if q in pinfo.quants:
            continue
        if not all(r in pinfo.quants for r in requires):
            continue
        particles[q] = f(particles)
        pinfo.quants.append(q)
        pinfo.qlabels[q] = label
        pinfo.qunits[q] = pinfo.qunits.get(units_from, "")
    return particles, info


def _read(filename, kind=None, ndims=None):
    """Read a ZDF file, adding derived quantities to particle data."""
    data, info = _core_read(filename, kind, ndims)
    if info.type == "particles":
        _add_derived(data, info)
    return data, info


# -----------------------------------------------------------------------------
# Particle plots with cylindrical defaults
# -----------------------------------------------------------------------------

def part2D(filename, qx='z', qy='r', xlim=None, ylim=None, grid=True,
           marker='.', ms=1, alpha=1, **kwargs):
    """Generates a scatter plot from a quasi-3D ZDF particle file.

    Same as `visxd.part2D()`, but defaults to a (z, r) plot, and also accepts
    the derived quantities "x" (r cos θ) and "y" (r sin θ).

    Args:
        filename (str): Name of ZDF file to open.
        qx (str, optional): x axis quantity, e.g. "z", "r", "x", "y", "ux",
            "uy", "uz". Defaults to "z".
        qy (str, optional): y axis quantity. Defaults to "r".
        xlim, ylim, grid, marker, ms, alpha, **kwargs: see `visxd.part2D()`.

    Returns:
        list of matplotlib.lines.Line2D: The plotted points.
    """
    particles, info = _read(filename, "particles")
    return _part2D(particles, info, filename, qx, qy, xlim=xlim, ylim=ylim,
                   grid=grid, marker=marker, ms=ms, alpha=alpha, **kwargs)


def histogram(filename, q, bins=128, range=None, density=True, log=False,
              color=None, histtype='bar', **kwargs):
    """Generates a histogram plot from a quasi-3D ZDF particle file.

    Same as `visxd.histogram()`, but also accepts the derived quantities "x"
    (r cos θ) and "y" (r sin θ).

    Args:
        filename (str): Name of ZDF file to open.
        q (str): Quantity to use, e.g. "z", "r", "x", "y", "ux", "uy", "uz".
        bins, range, density, log, color, histtype, **kwargs: see
            `visxd.histogram()`.

    Returns:
        tuple: (n, bins, patches) as returned by `Axes.hist()`.
    """
    particles, info = _read(filename, "particles")
    return _histogram(particles, info, filename, q, bins=bins, range=range,
                      density=density, log=log, color=color,
                      histtype=histtype, **kwargs)


def plot_data(data, iteration=None, **kwargs):
    """Plots a quasi-3D ZDF file, choosing the plot type from the file contents.

    Same as `visxd.plot_data()`, but particle files default to a (z, r) plot
    and accept the derived quantities "x" and "y".

    Args:
        data (str): File name prefix (e.g. "EMF/E0z"), or the full file name if
            `iteration` is not given.
        iteration (int, optional): Iteration number; the file name is
            "{data}-{iteration:06d}.zdf".
        **kwargs: Additional arguments passed to the plotting function.

    Returns:
        The artist(s) created by the selected plotting function.
    """
    iteration = _pop_iter(iteration, kwargs)
    filename = data if iteration is None else "{}-{:06d}.zdf".format(data, iteration)

    print("Plotting {}".format(filename))

    obj, info = _read(filename)

    if info.type == "grid":
        return _plot_grid(obj, info, filename, **kwargs)

    if info.type == "particles":
        kwargs.setdefault("qx", "z")
        kwargs.setdefault("qy", "r")
        return _part2D(obj, info, filename, **kwargs)

    raise ValueError("{}: unsupported file type '{}'".format(filename, info.type))
