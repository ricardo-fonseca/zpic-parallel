# Writing a nanobind wrapper

## Installing nanobind

PiP installation worked fine:

```shell
$ python3 -m pip install nanobind
```

## CMAKE

You __must__ use `CMake`, there is no way arround it. Installed with `brew`:

```shell
$ brew install cmake
```

## Compilation

Once you have created the appropriate `pyproject.toml` file, you can compile the file using:

```shell
pip install --no-build-isolation -Ceditable.rebuild=true -ve .
```

This will trigger a recompilation of the necessary source files whenever you import the module from Python/iPython (not sure about Jupyter notebooks)


## Other notes

### Not including `visxd`(or `zdf`) modules

You can make `visxd` an optional dependency, it is only used for `plot()` methods. This could be achieved doing something like:

```python
# ---------------------------------------------------------------------------
# Optional visxd dependency
# ---------------------------------------------------------------------------
try:
    import visxd as _visxd
    _HAS_VISXD = True
except ImportError:
    _visxd = None
    _HAS_VISXD = False


def _require_visxd() -> None:
    if not _HAS_VISXD:
        raise ImportError(
            "Plotting requires the 'visxd' package. "
            "Install it to enable Simulation.plot() and related helpers."
        )
```

And then, in the plotting routines:

```python
def _current_plot(self, fc, *, box=None, **kwargs):
    """Plot a current-density component using ``visxd.plot2d``.

    Parameters
    ----------
    fc : str or fcomp.cart
        Component to plot: ``'x'``, ``'y'`` or ``'z'``.
    box : (float, float), optional
        Physical size ``(Lx, Ly)`` of the simulation box. If omitted,
        the axes are drawn in cell units.
    **kwargs
        Extra keyword arguments forwarded to ``visxd.plot2d``.
    """
    
    # Break if visxd is not present
    _require_visxd()
    
    fc_enum = _resolve_fc(fc)
    fc_str  = _fc_name(fc)

    data   = self.gather(fc_enum)
    flabel = f"J_{fc_str}"
    time   = self.iter * self.dt

    box = self.box
    frange = [[0.0, float(box[0])], [0.0, float(box[1])]]

    # Notice that we are using _visxd here
    _visxd.plot2d(
        data,
        range  = frange,
        title  = r"$\sf {} $""\n"r"$t = {:g} \;[\sf {}]$".format(
                     flabel, time, r"1 / \omega_n"),
        xtitle = r"$\sf x \;[c / \omega_n]$",
        ytitle = r"$\sf y \;[c / \omega_n]$",
        vtitle = r"$\sf {} \;[e \omega_n^2 / c]$".format(flabel),
        **kwargs,
    )
```