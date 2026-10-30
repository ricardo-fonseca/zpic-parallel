"""
visxd - Quick-look visualization of ZDF simulation output.

Conventions
-----------
* 2D grid data is stored as ``data[iy, ix]`` (shape ``(ny, nx)``), so the
  first grid axis (``info.grid.axis[0]``) is the horizontal (x) axis.
* Every plotting function draws into a Matplotlib ``Axes`` and returns the
  artist(s) it created (image, lines or histogram), so the result can be
  further customized.
* ``ax`` selects the Axes to draw into. If omitted, the current axes
  (``plt.gca()``) are used.
* ``show`` controls whether ``plt.show()`` is called. The default (``None``)
  shows the figure only when no ``ax`` was given, which matches the previous
  behavior for interactive use while allowing figures with several subplots
  to be composed by passing ``ax=``.
* ``save`` writes the figure to the given file. The format is inferred from
  the file extension (``.pdf``, ``.png``, ``.svg``, ...).
* Default titles and axis labels are built from the file metadata, but can
  be overridden by passing ``title``, ``xtitle``, ``ytitle`` (and ``vtitle``
  for colorbars). Passing ``None`` explicitly removes them.
* Errors (missing files, wrong file type, invalid options, inconsistent
  data) raise ``FileNotFoundError`` / ``ValueError``.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import zdf

__all__ = [
    "plot1d", "plot2d",
    "grid", "grid1d", "grid2d", "grid2d_fft", "vfield2d",
    "part2D", "histogram",
    "plot_data", "plot_vfield2d",
]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _esc(s):
    """Prepare a metadata string for use inside mathtext.

    Mathtext ignores whitespace, so spaces are converted to explicit
    mathtext spaces.
    """
    return str(s).replace(" ", "\\;")


def _mlabel(q, units=None):
    """Label from an already mathtext-ready quantity `q` and optional units.

    The quantity is typeset in math italic and the units in sans-serif.
    """
    if not q:
        return None
    s = "$" + q
    if units:
        s += "\\,[{\\sf " + _esc(units) + "}]"
    return s + "$"


def _qlabel(label, units=None):
    """Axis/colorbar label from a metadata label and units."""
    if not label:
        return None
    return _mlabel(_esc(label), units)


def _sf(q):
    """Sans-serif mathtext string (used for titles)."""
    return "$\\sf " + q + "$"


def _title(main, info):
    """Plot title: `main` followed by a line with the simulation time.

    The time line is only added if the file has iteration information.
    """
    it = getattr(info, "iteration", None)
    if not it:
        return main
    t = "t = ${:g}$".format(it.t)
    if getattr(it, "tunits", None):
        t += " [${\\sf " + _esc(it.tunits) + "}$]"
    return t if main is None else main + "\n" + t


def _grid_quantity(info):
    """Mathtext-ready name of a grid quantity (without the $ delimiters)."""
    if getattr(info.grid, "label", None):
        return _esc(info.grid.label)
    name = getattr(info.grid, "name", None) or "data"
    return _esc(name.replace("_", "\\_"))


def _has_axis(info):
    return bool(getattr(info.grid, "axis", None))


def _axis_range(info, i):
    """[min, max] of grid axis `i`, or cell units if the file has no axis info."""
    if _has_axis(info):
        a = info.grid.axis[i]
        return [a.min, a.max]
    return [0, info.grid.nx[i]]


def _axis_label(info, i):
    if not _has_axis(info):
        return None
    a = info.grid.axis[i]
    return _qlabel(a.label, getattr(a, "units", None))


def _read(filename, kind=None, ndims=None):
    """Read a ZDF file, checking file type and grid dimensions."""
    if not os.path.exists(filename):
        raise FileNotFoundError("ZDF file not found: {}".format(filename))

    data, info = zdf.read(filename)

    if kind is not None and info.type != kind:
        raise ValueError("{} is not a {} file (type is '{}')".format(
            filename, kind, info.type))

    if ndims is not None and info.grid.ndims != ndims:
        raise ValueError("{} is not a {}D grid file ({}D)".format(
            filename, ndims, info.grid.ndims))

    return data, info


# Complex number parts: (function, mathtext template for the quantity name)
_PARTS = {
    "real":  (np.real,  "\\Re\\;{}"),
    "imag":  (np.imag,  "\\Im\\;{}"),
    "abs":   (np.abs,   "\\left|{}\\right|"),
    "angle": (np.angle, "Arg\\;\\,{}"),
}
_PARTS["mag"] = _PARTS["abs"]


def _select_part(data, part):
    """Select part of complex data.

    Returns the (real) data and a template for formatting the quantity name.
    Real data is returned unchanged.
    """
    if part not in _PARTS:
        raise ValueError(
            "Invalid part '{}', must be one of 'real', 'imag', 'abs' "
            "(or 'mag') or 'angle'".format(part))
    if not np.iscomplexobj(data):
        return data, "{}"
    f, fmt = _PARTS[part]
    return f(data), fmt


def _apply_scale(data, scale):
    """Return data * scale[0] + scale[1]. A scalar `scale` is a pure factor."""
    s = np.atleast_1d(scale)
    return data * s[0] + (s[1] if s.size > 1 else 0)


def _shift(data, range, shift):
    """Circularly shift 2D data by `shift` cells (int or (sx, sy)).

    The plot range is moved accordingly so that features stay at their
    correct physical position (the shifted-in data is shown at its
    periodic image coordinates).
    """
    s = np.broadcast_to(np.asarray(shift, dtype=int), (2,))
    sx, sy = int(s[0]), int(s[1])
    data = np.roll(data, (sx, sy), axis=(1, 0))

    ny, nx = data.shape
    dx = (range[0][1] - range[0][0]) / nx
    dy = (range[1][1] - range[1][0]) / ny

    range = [
        [range[0][0] - sx * dx, range[0][1] - sx * dx],
        [range[1][0] - sy * dy, range[1][1] - sy * dy],
    ]
    return data, range


def _get_ax(ax):
    """Returns the axes to use, and whether the caller supplied them."""
    if ax is None:
        return plt.gca(), False
    return ax, True


def _decorate(ax, title, xtitle, ytitle, xlim, ylim, grid):
    if title is not None:
        ax.set_title(title)
    if xtitle is not None:
        ax.set_xlabel(xtitle)
    if ytitle is not None:
        ax.set_ylabel(ytitle)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(grid)


def _finish(ax, user_ax, show, save):
    if save is not None:
        ax.figure.savefig(save, bbox_inches="tight")
    if show is None:
        show = not user_ax
    if show:
        plt.show()


def _pop_iter(iteration, kwargs):
    """Support the old `iter=` keyword as an alias for `iteration=`."""
    if "iter" in kwargs:
        if iteration is not None:
            raise TypeError("Specify only one of 'iteration' and 'iter'")
        return kwargs.pop("iter")
    return iteration


def _check_quant(info, q, filename):
    if q not in info.particles.quants:
        raise ValueError(
            "Quantity '{}' is not present in {}. Available quantities: {}".format(
                q, filename, ", ".join(info.particles.quants)))


# -----------------------------------------------------------------------------
# Generic plots
# -----------------------------------------------------------------------------

def plot1d(x, y, marker=None, ms=None, alpha=None, c=None,
           xlim=None, ylim=None, grid=True,
           yscale=None, xscale=None,
           title=None, xtitle=None, ytitle=None,
           show=None, save=None, ax=None, **kwargs):
    """Generate an x-y line / scatter plot

    Args:
        x (array): x values to plot.
        y (array): y values to plot. Must be real.
        marker (str, optional): Matplotlib format string (e.g. '.', 'o', '--').
            Defaults to None, which draws a solid line.
        ms (float, optional): Marker size. Defaults to None (Matplotlib default).
        alpha (float, optional): Marker/line opacity, 0 to 1. Defaults to None
            (opaque).
        c (str, optional): Line/marker color. Defaults to None, which uses
            the Matplotlib color cycle.
        xlim (list, optional): Range of values for the x-axis. Defaults to None,
            which sets the axis range automatically.
        ylim (list, optional): Range of values for the y-axis. Defaults to None,
            which sets the axis range automatically.
        grid (bool, optional): Display gridlines on the plot. Defaults to True.
        yscale (str, optional): y-axis scale ('linear', 'log', 'symlog', ...).
            Defaults to None (unchanged).
        xscale (str, optional): x-axis scale ('linear', 'log', 'symlog', ...).
            Defaults to None (unchanged).
        title (str, optional): Plot title. Defaults to None.
        xtitle (str, optional): x-axis label. Defaults to None.
        ytitle (str, optional): y-axis label. Defaults to None.
        show (bool, optional): Call `plt.show()` at the end. Defaults to None,
            which shows the plot only if `ax` was not given. Use False to
            overplot additional quantities.
        save (str, optional): Save the figure to this file; the format is
            inferred from the extension. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes to draw into. Defaults to
            the current axes.
        **kwargs: Additional arguments passed to `Axes.plot()` (e.g. `label`,
            `lw`, `ls`).

    Returns:
        list of matplotlib.lines.Line2D: The plotted lines.
    """
    if np.iscomplexobj(y):
        raise ValueError("y data is complex; plot np.real(y), np.imag(y), "
                         "np.abs(y) or np.angle(y) instead")

    ax, user_ax = _get_ax(ax)

    args = (x, y) if marker is None else (x, y, marker)
    lines = ax.plot(*args, ms=ms, alpha=alpha, c=c, **kwargs)

    _decorate(ax, title, xtitle, ytitle, xlim, ylim, grid)

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    _finish(ax, user_ax, show, save)
    return lines


def plot2d(data, range=None, xlim=None, ylim=None, grid=False, cmap=None, norm=None,
           vsym=False, vmin=None, vmax=None, scale=None, shift=None,
           title=None, xtitle=None, ytitle=None, vtitle=None,
           show=None, save=None, ax=None, colorbar=True, vsim=None):
    """Generate a 2D image map plot

    Args:
        data (array): 2D grid data to plot, indexed as data[iy, ix]. Must be real.
        range (list, optional): Physical range covered by the grid (position of
            the cell edges), as [[xmin, xmax], [ymin, ymax]]. Defaults to cell
            units, [[0, nx], [0, ny]].
        xlim (list, optional): Range of values for the x-axis. Defaults to None,
            which shows the full grid.
        ylim (list, optional): Range of values for the y-axis. Defaults to None,
            which shows the full grid.
        grid (bool, optional): Display gridlines on the plot. Defaults to False.
        cmap (str or matplotlib.colors.Colormap, optional): Colormap to use.
            Defaults to None (Matplotlib default colormap).
        norm (matplotlib.colors.Normalize, optional): Color scale normalization,
            e.g. `colors.LogNorm()` or `colors.CenteredNorm()`. Defaults to None
            (linear scale). If used, set the value limits on the norm object
            itself, not through `vmin` / `vmax`.
        vsym (bool, optional): Use a color scale symmetric around 0, i.e.
            [-max|data|, max|data|], or [-vmax, vmax] if `vmax` is given.
            Cannot be combined with `norm`. Defaults to False.
        vmin (float, optional): Minimum value for the color scale. Defaults to
            None, which uses the data minimum.
        vmax (float, optional): Maximum value for the color scale. Defaults to
            None, which uses the data maximum.
        scale (float or list, optional): Linearly scale the data before
            plotting: plots data * scale[0] + scale[1]. A single value is
            used as a pure multiplicative factor. Defaults to None.
        shift (int or list, optional): Circularly shift the data by this number
            of cells, (sx, sy) or a single value for both directions. The axes
            are shifted accordingly, so features keep their physical
            coordinates. Defaults to None.
        title (str, optional): Plot title. Defaults to None.
        xtitle (str, optional): x-axis label. Defaults to None.
        ytitle (str, optional): y-axis label. Defaults to None.
        vtitle (str, optional): Colorbar label. Defaults to None.
        show (bool, optional): Call `plt.show()` at the end. Defaults to None,
            which shows the plot only if `ax` was not given.
        save (str, optional): Save the figure to this file; the format is
            inferred from the extension. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes to draw into. Defaults to
            the current axes.
        colorbar (bool, optional): Add a colorbar. Defaults to True.
        vsim (bool, optional): Deprecated alias for `vsym`.

    Returns:
        matplotlib.image.AxesImage: The image.
    """
    if vsim is not None:
        vsym = vsim

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("plot2d requires 2D data, got shape {}".format(data.shape))
    if np.iscomplexobj(data):
        raise ValueError("Data is complex; select a part first (np.real, "
                         "np.imag, np.abs or np.angle)")

    if scale is not None:
        data = _apply_scale(data, scale)

    if range is None:
        range = [[0, data.shape[1]], [0, data.shape[0]]]

    if shift is not None:
        data, range = _shift(data, range, shift)

    if vsym:
        if norm is not None:
            raise ValueError("vsym cannot be combined with norm; use a centered "
                             "norm directly, e.g. colors.CenteredNorm() or "
                             "colors.SymLogNorm()")
        norm = colors.CenteredNorm(halfrange=vmax)
        vmin = vmax = None

    ax, user_ax = _get_ax(ax)

    im = ax.imshow(data, interpolation="nearest", origin="lower",
                   vmin=vmin, vmax=vmax, norm=norm,
                   extent=(range[0][0], range[0][1], range[1][0], range[1][1]),
                   aspect="auto", cmap=cmap)

    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax)
        if vtitle is not None:
            cb.set_label(vtitle)

    _decorate(ax, title, xtitle, ytitle, xlim, ylim, grid)

    _finish(ax, user_ax, show, save)
    return im


# -----------------------------------------------------------------------------
# Grid files
# -----------------------------------------------------------------------------

def _grid1d(data, info, part="real", scale=None, **kwargs):
    data, fmt = _select_part(data, part)

    if scale is not None:
        data = _apply_scale(data, scale)

    # Cell centered x positions
    nx = info.grid.nx[0]
    xmin, xmax = _axis_range(info, 0)
    dx = (xmax - xmin) / nx
    x = xmin + (np.arange(nx) + 0.5) * dx

    kwargs.setdefault("title", _title(_sf(fmt.format(_grid_quantity(info))), info))
    kwargs.setdefault("xtitle", _axis_label(info, 0))
    kwargs.setdefault("ytitle", _qlabel(info.grid.label, getattr(info.grid, "units", None)))

    return plot1d(x, data, **kwargs)


def _grid2d(data, info, part="real", **kwargs):
    data, fmt = _select_part(data, part)

    kwargs.setdefault("range", [_axis_range(info, 0), _axis_range(info, 1)])
    kwargs.setdefault("title", _title(_sf(fmt.format(_grid_quantity(info))), info))
    kwargs.setdefault("xtitle", _axis_label(info, 0))
    kwargs.setdefault("ytitle", _axis_label(info, 1))
    kwargs.setdefault("vtitle", _qlabel(info.grid.label, getattr(info.grid, "units", None)))

    return plot2d(data, **kwargs)


def _plot_grid(data, info, filename, **kwargs):
    ndims = info.grid.ndims
    if ndims == 1:
        return _grid1d(data, info, **kwargs)
    if ndims == 2:
        return _grid2d(data, info, **kwargs)
    raise ValueError("{}: unsupported grid dimensions ({})".format(filename, ndims))


def grid1d(filename: str, xlim=None, grid: bool = True, scale=None,
           part: str = "real", **kwargs):
    """Generates a line plot from a 1D grid file

    Args:
        filename (str): Name of ZDF file to open.
        xlim (list, optional): Range of values for the x-axis. Defaults to the
            grid range.
        grid (bool, optional): Display gridlines on the plot. Defaults to True.
        scale (float or list, optional): Linearly scale the data before
            plotting: plots data * scale[0] + scale[1]. A single value is used
            as a pure multiplicative factor. Defaults to None.
        part (str, optional): Part of complex data to plot: 'real', 'imag',
            'abs' (or 'mag') or 'angle'. Ignored for real data. Defaults to 'real'.
        **kwargs: Additional arguments passed to `plot1d()` (e.g. `ylim`,
            `yscale`, `marker`, `title`, `show`, `save`, `ax`).

    Returns:
        list of matplotlib.lines.Line2D: The plotted lines.
    """
    data, info = _read(filename, "grid", 1)
    return _grid1d(data, info, part=part, scale=scale, xlim=xlim, grid=grid, **kwargs)


def grid2d(filename: str, part: str = "real", **kwargs):
    """Generates a colormap plot from a 2D grid file

    Args:
        filename (str): Name of ZDF file to open.
        part (str, optional): Part of complex data to plot: 'real' (real part),
            'imag' (imaginary part), 'abs' or 'mag' (magnitude), or 'angle'
            (complex argument). Ignored for real data. Defaults to 'real'.
        **kwargs: Additional arguments passed to `plot2d()` (e.g. `xlim`,
            `cmap`, `norm`, `vsym`, `shift`, `title`, `show`, `save`, `ax`).

    Returns:
        matplotlib.image.AxesImage: The image.
    """
    data, info = _read(filename, "grid", 2)
    return _grid2d(data, info, part=part, **kwargs)


def grid(filename: str, **kwargs):
    """Generates a plot from a 1D or 2D grid file.

    Works as a driver for `grid1d()` and `grid2d()`.

    Args:
        filename (str): Name of ZDF file to open.
        **kwargs: Additional arguments passed to `grid1d()` / `grid2d()`.

    Returns:
        The artist(s) created by `grid1d()` or `grid2d()`.
    """
    data, info = _read(filename, "grid")
    return _plot_grid(data, info, filename, **kwargs)


def _vector_lines(ax, fx, fy, range, kind, opts, xlim, ylim):
    """Draw field lines (streamplot) or arrows (quiver) of the (fx, fy) field.

    fx, fy are indexed [iy, ix] with cell edges at `range`. Only the visible
    region (xlim / ylim) is used, so that the line density refers to what is
    actually shown.
    """
    if kind is True:
        kind = "stream"
    if kind not in ("stream", "quiver"):
        raise ValueError("Invalid lines option '{}', must be 'stream' (or True) "
                         "or 'quiver'".format(kind))

    opts = dict(opts) if opts else {}

    # Cell centered positions
    ny, nx = fx.shape
    dx = (range[0][1] - range[0][0]) / nx
    dy = (range[1][1] - range[1][0]) / ny
    x = range[0][0] + (np.arange(nx) + 0.5) * dx
    y = range[1][0] + (np.arange(ny) + 0.5) * dy

    # Restrict to visible region
    xl = range[0] if xlim is None else xlim
    yl = range[1] if ylim is None else ylim
    ix = np.nonzero((x >= min(xl)) & (x <= max(xl)))[0]
    iy = np.nonzero((y >= min(yl)) & (y <= max(yl)))[0]
    if ix.size < 2 or iy.size < 2:
        return None

    x, y = x[ix], y[iy]
    fx = fx[np.ix_(iy, ix)]
    fy = fy[np.ix_(iy, ix)]

    if kind == "stream":
        opts.setdefault("color", "k")
        opts.setdefault("linewidth", 0.6)
        opts.setdefault("arrowsize", 0.8)
        opts.setdefault("density", 1)
        artist = ax.streamplot(x, y, fx, fy, **opts)
    else:
        stride = opts.pop("stride", None)
        if stride is None:
            # About 24 arrows along each direction
            stride = (max(1, round(x.size / 24)), max(1, round(y.size / 24)))
        s = np.broadcast_to(np.asarray(stride, dtype=int), (2,))
        sx, sy = int(s[0]), int(s[1])
        opts.setdefault("color", "k")
        opts.setdefault("pivot", "mid")
        opts.setdefault("angles", "xy")
        artist = ax.quiver(x[::sx], y[::sy], fx[::sy, ::sx], fy[::sy, ::sx], **opts)

    # streamplot / quiver may change the axis limits
    ax.set_xlim(xl)
    ax.set_ylim(yl)

    return artist


def vfield2d(filex, filey, xlim=None, ylim=None, grid=False, cmap=None,
             vmin=None, vmax=None, title=None, shift=None,
             lines=None, lines_kwargs=None, **kwargs):
    """Generates a colormap plot of the magnitude of a 2D vector field

    The plotted value is sqrt(|fx|^2 + |fy|^2). Optionally, field lines or
    field arrows can be drawn on top of the magnitude.

    Args:
        filex (str): Name of ZDF file to open for the x field component.
        filey (str): Name of ZDF file to open for the y field component.
        xlim (list, optional): Range of values for the x-axis. Defaults to the
            grid range.
        ylim (list, optional): Range of values for the y-axis. Defaults to the
            grid range.
        grid (bool, optional): Display gridlines on the plot. Defaults to False.
        cmap (str, optional): Colormap to use. Defaults to the Matplotlib
            default colormap.
        vmin (float, optional): Minimum value for the color scale. Defaults to
            0 (or to the norm settings if `norm` is given).
        vmax (float, optional): Maximum value for the color scale. Defaults to
            the data maximum.
        title (str, optional): Main plot title (Matplotlib text, may include
            $...$ math); the simulation time is added on a second line.
            Defaults to the field magnitude expression.
        shift (int or list, optional): Circularly shift the data, see `plot2d()`.
        lines (str or bool, optional): Draw the field direction on top of the
            magnitude: 'stream' (or True) draws field lines using
            `Axes.streamplot()`, 'quiver' draws arrows using `Axes.quiver()`.
            Only the real part of complex fields is used. Defaults to None
            (no lines).
        lines_kwargs (dict, optional): Additional arguments passed to
            `Axes.streamplot()` (e.g. `density`, `color`, `linewidth`,
            `arrowsize`) or `Axes.quiver()` (e.g. `color`, `scale`, `width`).
            For 'quiver', the extra key `stride` (int or (sx, sy)) sets the
            spacing between arrows in cells; by default about 24 arrows are
            drawn along each direction. Defaults to None.
        **kwargs: Additional arguments passed to `plot2d()` (e.g. `norm`,
            `show`, `save`, `ax`).

    Returns:
        matplotlib.image.AxesImage: The image.
    """
    if os.path.abspath(filex) == os.path.abspath(filey):
        raise ValueError("The 2 files are the same: {}".format(filex))

    datax, infox = _read(filex, "grid", 2)
    datay, infoy = _read(filey, "grid", 2)

    # Check that both components are compatible
    if np.shape(datax) != np.shape(datay):
        raise ValueError("{} / {} don't have the same grid dimensions".format(filex, filey))

    rx = [_axis_range(infox, 0), _axis_range(infox, 1)]
    ry = [_axis_range(infoy, 0), _axis_range(infoy, 1)]
    if not np.allclose(rx, ry):
        raise ValueError("{} / {} don't have the same grid range".format(filex, filey))

    tx = getattr(getattr(infox, "iteration", None), "t", None)
    ty = getattr(getattr(infoy, "iteration", None), "t", None)
    if tx is not None and ty is not None and not np.isclose(tx, ty):
        raise ValueError("{} / {} are from different times ({:g} / {:g})".format(
            filex, filey, tx, ty))

    # Shift here (rather than in plot2d) so that the field lines match
    range = rx
    if shift is not None:
        datax, range = _shift(datax, rx, shift)
        datay, _ = _shift(datay, rx, shift)

    data = np.hypot(np.abs(datax), np.abs(datay))

    q = "\\sqrt{{" + _grid_quantity(infox) + "}^2 + {" + _grid_quantity(infoy) + "}^2}"

    if vmin is None and kwargs.get("norm") is None:
        vmin = 0

    kwargs.setdefault("title", _title(title if title is not None else _sf(q), infox))
    kwargs.setdefault("xtitle", _axis_label(infox, 0))
    kwargs.setdefault("ytitle", _axis_label(infox, 1))
    kwargs.setdefault("vtitle", _mlabel(q, getattr(infox.grid, "units", None)))

    # Lines must be drawn before the figure is saved / shown
    ax, user_ax = _get_ax(kwargs.pop("ax", None))
    show = kwargs.pop("show", None)
    save = kwargs.pop("save", None)

    im = plot2d(data, range=range, xlim=xlim, ylim=ylim, grid=grid, cmap=cmap,
                vmin=vmin, vmax=vmax, ax=ax, show=False, **kwargs)

    if lines:
        _vector_lines(ax, np.real(datax), np.real(datay), range,
                      lines, lines_kwargs, xlim, ylim)

    _finish(ax, user_ax, show, save)
    return im


def grid2d_fft(filename: str, xlim=None, ylim=None, grid=False, cmap=None, norm=None,
               vmin=None, vmax=None, plot="abs", part=None, **kwargs):
    """Generates a colormap plot from the FFT of a 2D grid file

    The spectrum is shown with k = 0 at the center, and the k axes go from
    -k_Nyquist to +k_Nyquist (k = 2 pi / lambda).

    Args:
        filename (str): Name of ZDF file to open.
        xlim (list, optional): Range of values for the kx axis. Defaults to the
            full kx range.
        ylim (list, optional): Range of values for the ky axis. Defaults to the
            full ky range.
        grid (bool, optional): Display gridlines on the plot. Defaults to False.
        cmap (str, optional): Colormap to use. Defaults to the Matplotlib
            default colormap.
        norm (matplotlib.colors.Normalize, optional): Color scale normalization,
            e.g. `colors.LogNorm()`. Defaults to None (linear).
        vmin (float, optional): Minimum value for the color scale. Defaults to
            the smallest value in the plotted data.
        vmax (float, optional): Maximum value for the color scale. Defaults to
            the largest value in the plotted data.
        plot (str, optional): Part of the FFT to plot: 'abs' (or 'mag'),
            'real', 'imag' or 'angle'. Defaults to 'abs'.
        part (str, optional): Alias for `plot`, consistent with `grid2d()`.
        **kwargs: Additional arguments passed to `plot2d()` (e.g. `vsym`,
            `title`, `show`, `save`, `ax`).

    Returns:
        matplotlib.image.AxesImage: The image.
    """
    data, info = _read(filename, "grid", 2)

    if part is None:
        part = plot

    ny, nx = np.shape(data)
    xmin, xmax = _axis_range(info, 0)
    ymin, ymax = _axis_range(info, 1)
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    # Get FFT of data, k = 0 at the center
    fdata = np.fft.fftshift(np.fft.fft2(data))
    fdata, fmt = _select_part(fdata, part)

    # k values at the cell centers, valid for both even and odd grid sizes
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    dkx = 2 * np.pi / (nx * dx)
    dky = 2 * np.pi / (ny * dy)

    range = [
        [kx[0] - 0.5 * dkx, kx[-1] + 0.5 * dkx],
        [ky[0] - 0.5 * dky, ky[-1] + 0.5 * dky],
    ]

    q = fmt.format("\\mathcal{F}(" + _grid_quantity(info) + ")")

    if _has_axis(info):
        xlabel = "$k_{" + _esc(info.grid.axis[0].label) + "}$"
        ylabel = "$k_{" + _esc(info.grid.axis[1].label) + "}$"
    else:
        xlabel, ylabel = "$k_x$", "$k_y$"

    kwargs.setdefault("title", _title(_sf(q), info))
    kwargs.setdefault("xtitle", xlabel)
    kwargs.setdefault("ytitle", ylabel)
    kwargs.setdefault("vtitle", "$" + q + "$")

    return plot2d(fdata, range=range, xlim=xlim, ylim=ylim, grid=grid, cmap=cmap,
                  norm=norm, vmin=vmin, vmax=vmax, **kwargs)


# -----------------------------------------------------------------------------
# Particle files
# -----------------------------------------------------------------------------

def _species_prefix(info):
    label = getattr(info.particles, "label", None)
    return _esc(label) + "\\;-\\;" if label else ""


def _part2D(particles, info, filename, qx, qy, **kwargs):
    _check_quant(info, qx, filename)
    _check_quant(info, qy, filename)

    qlabels = info.particles.qlabels
    qunits = info.particles.qunits

    main = _sf(_species_prefix(info) + _esc(qlabels[qy]) + "\\,/\\," + _esc(qlabels[qx]))

    kwargs.setdefault("title", _title(main, info))
    kwargs.setdefault("xtitle", _qlabel(qlabels[qx], qunits[qx]))
    kwargs.setdefault("ytitle", _qlabel(qlabels[qy], qunits[qy]))

    return plot1d(particles[qx], particles[qy], **kwargs)


def part2D(filename, qx, qy, xlim=None, ylim=None, grid=True,
           marker='.', ms=1, alpha=1, **kwargs):
    """Generates an (x,y) scatter plot from a ZDF particle file.

    Args:
        filename (str): Name of ZDF file to open.
        qx (str): x axis quantity, usually one of "x", "y", "ux", "uy", "uz", etc.
        qy (str): y axis quantity, usually one of "x", "y", "ux", "uy", "uz", etc.
        xlim (list, optional): Range of values for the x-axis. Defaults to the
            limits of the `qx` particle data.
        ylim (list, optional): Range of values for the y-axis. Defaults to the
            limits of the `qy` particle data.
        grid (bool, optional): Display gridlines on the plot. Defaults to True.
        marker (str, optional): Marker to use for the scatter plot. Defaults to '.'.
        ms (float, optional): Marker size. Defaults to 1.
        alpha (float, optional): Marker opacity, 0 to 1. Defaults to 1.
        **kwargs: Additional arguments passed to `plot1d()` (e.g. `c`, `title`,
            `show`, `save`, `ax`).

    Returns:
        list of matplotlib.lines.Line2D: The plotted points.
    """
    particles, info = _read(filename, "particles")
    return _part2D(particles, info, filename, qx, qy, xlim=xlim, ylim=ylim,
                   grid=grid, marker=marker, ms=ms, alpha=alpha, **kwargs)


def histogram(filename, q, bins=128, range=None, density=True, log=False,
              color=None, histtype='bar',
              xlim=None, ylim=None, grid=False,
              title=None, xtitle=None, ytitle=None,
              show=None, save=None, ax=None, **kwargs):
    """Generates a histogram (frequency) plot from a ZDF particle file.

    Args:
        filename (str): Name of ZDF file to open.
        q (str): Quantity to use, usually one of "x", "y", "ux", "uy", "uz", etc.
        bins (int, optional): Number of bins. Defaults to 128.
        range (tuple, optional): Lower and upper limits of the histogram.
            Defaults to the minimum and maximum values of the selected quantity.
        density (bool, optional): Plot a probability density (bin count
            divided by the total number of counts and the bin width) instead of
            the bin count. Defaults to True.
        log (bool, optional): Use log scale for the histogram axis. Defaults
            to False.
        color (str, optional): Color for plot. Defaults to the Matplotlib
            color cycle.
        histtype (str, optional): Type of histogram to draw ('bar', 'step',
            'stepfilled', ...), see Matplotlib `hist()`. Defaults to 'bar'.
        xlim (list, optional): Range of values for the x-axis. Defaults to None.
        ylim (list, optional): Range of values for the y-axis. Defaults to None.
        grid (bool, optional): Display gridlines on the plot. Defaults to False.
        title (str, optional): Plot title. Defaults to species and quantity
            name plus simulation time.
        xtitle (str, optional): x-axis label. Defaults to the quantity name
            and units.
        ytitle (str, optional): y-axis label. Defaults to "Probability density"
            or "Counts", depending on `density`.
        show (bool, optional): Call `plt.show()` at the end. Defaults to None,
            which shows the plot only if `ax` was not given.
        save (str, optional): Save the figure to this file; the format is
            inferred from the extension. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes to draw into. Defaults to
            the current axes.
        **kwargs: Additional arguments passed to `Axes.hist()` (e.g.
            `weights`, `label`).

    Returns:
        tuple: (n, bins, patches) as returned by `Axes.hist()`.
    """
    particles, info = _read(filename, "particles")
    _check_quant(info, q, filename)

    ax, user_ax = _get_ax(ax)

    result = ax.hist(particles[q], bins=bins, range=range, density=density,
                     log=log, color=color, histtype=histtype, **kwargs)

    qlabel = info.particles.qlabels[q]
    if title is None:
        title = _title(_sf(_species_prefix(info) + _esc(qlabel)), info)
    if xtitle is None:
        xtitle = _qlabel(qlabel, info.particles.qunits[q])
    if ytitle is None:
        ytitle = "Probability density" if density else "Counts"

    _decorate(ax, title, xtitle, ytitle, xlim, ylim, grid)

    _finish(ax, user_ax, show, save)
    return result


# -----------------------------------------------------------------------------
# Convenience drivers
# -----------------------------------------------------------------------------

def plot_data(data, iteration=None, **kwargs):
    """Plots a ZDF file, choosing the plot type from the file contents.

    Args:
        data (str): File name prefix (e.g. "EMF/e1"), or the full file name if
            `iteration` is not given.
        iteration (int, optional): Iteration number; the file name is
            "{data}-{iteration:06d}.zdf". The old `iter=` keyword is also
            accepted.
        **kwargs: Additional arguments passed to `grid1d()`, `grid2d()` or
            `part2D()`. Particle files require the `qx` and `qy` arguments.

    Returns:
        The artist(s) created by the selected plotting function.
    """
    iteration = _pop_iter(iteration, kwargs)

    if iteration is None:
        filename = data
    else:
        filename = "{}-{:06d}.zdf".format(data, iteration)

    print("Plotting {}".format(filename))

    obj, info = _read(filename)

    if info.type == "grid":
        return _plot_grid(obj, info, filename, **kwargs)

    if info.type == "particles":
        qx = kwargs.pop("qx", None)
        qy = kwargs.pop("qy", None)
        if qx is None or qy is None:
            raise ValueError("Particle file {}: the 'qx' and 'qy' quantities "
                             "must be specified".format(filename))
        return _part2D(obj, info, filename, qx, qy, **kwargs)

    raise ValueError("{}: unsupported file type '{}'".format(filename, info.type))


def plot_vfield2d(fld, iteration=None, xlim=None, ylim=None, grid=False,
                  norm=None, cmap=None, shift=None, zcmap='BrBG',
                  lines=None, lines_kwargs=None, figsize=None,
                  show=None, save=None, **kwargs):
    """Plots the in-plane magnitude and out-of-plane component of a 2D vector field.

    Reads "{fld}x-{iteration:06d}.zdf" and "{fld}y-{iteration:06d}.zdf" for the
    in-plane magnitude, and "{fld}z-{iteration:06d}.zdf" (if present) for the
    out-of-plane component. The two plots are shown side by side in a new
    figure.

    Args:
        fld (str): Field file name prefix (e.g. "EMF/e" or "EMF/b").
        iteration (int): Iteration number. The old `iter=` keyword is also
            accepted.
        xlim (list, optional): Range of values for the x-axis. Defaults to the
            grid range.
        ylim (list, optional): Range of values for the y-axis. Defaults to the
            grid range.
        grid (bool, optional): Display gridlines on the plots. Defaults to False.
        norm (matplotlib.colors.Normalize, optional): Color normalization for
            the out-of-plane component. Defaults to `colors.CenteredNorm()`.
        cmap (str, optional): Colormap for the in-plane magnitude. Defaults to
            'YlOrBr'.
        shift (int or list, optional): Circularly shift the data, see `plot2d()`.
        zcmap (str, optional): Colormap for the out-of-plane component.
            Defaults to 'BrBG'.
        lines (str or bool, optional): Draw field lines ('stream' or True) or
            arrows ('quiver') on top of the in-plane magnitude, see
            `vfield2d()`. Defaults to None (no lines).
        lines_kwargs (dict, optional): Options for the field lines / arrows,
            see `vfield2d()`. Defaults to None.
        figsize (tuple, optional): Figure size in inches. Defaults to
            (6.4, 4.8) per plot.
        show (bool, optional): Call `plt.show()` at the end. Defaults to True.
        save (str, optional): Save the figure to this file; the format is
            inferred from the extension. Defaults to None.

    Returns:
        tuple: (in-plane image, out-of-plane image or None). The figure is
            available as `im.figure`.
    """
    iteration = _pop_iter(iteration, kwargs)
    if kwargs:
        raise TypeError("Unexpected arguments: {}".format(", ".join(kwargs)))
    if iteration is None:
        raise TypeError("plot_vfield2d() requires the iteration number")

    filex = "{}x-{:06d}.zdf".format(fld, iteration)
    filey = "{}y-{:06d}.zdf".format(fld, iteration)
    filez = "{}z-{:06d}.zdf".format(fld, iteration)

    has_z = os.path.exists(filez)
    ncols = 2 if has_z else 1

    if cmap is None:
        cmap = 'YlOrBr'
    if figsize is None:
        figsize = (6.4 * ncols, 4.8)

    fig, axs = plt.subplots(1, ncols, figsize=figsize, squeeze=False,
                            layout="constrained")
    axs = axs[0]

    im_xy = vfield2d(filex, filey, xlim=xlim, ylim=ylim, grid=grid, cmap=cmap,
                     title="In-plane {} field".format(fld), shift=shift,
                     lines=lines, lines_kwargs=lines_kwargs,
                     ax=axs[0], show=False)

    im_z = None
    if has_z:
        if norm is None:
            norm = colors.CenteredNorm()
        im_z = grid2d(filez, xlim=xlim, ylim=ylim, grid=grid, cmap=zcmap,
                      norm=norm, shift=shift, ax=axs[1], show=False)

    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    if show is None or show:
        plt.show()

    return im_xy, im_z
