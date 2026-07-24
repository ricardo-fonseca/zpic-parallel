"""ZPIC EM2D — 2D electromagnetic PIC plasma simulation."""

from __future__ import annotations

from . import _core
from . import visxd

from . import zdf
import numpy as np

# Re-export the C++ classes and submodules at the package level so users
# can write `from em2d import Simulation, Species, udist, density`.
from ._core import (
    Simulation,
    EMF,
    Current,
    Species,
    cart,
    fcomp,
    part,
    phasespace,
    udist,
    density,
    sys_info,
    build_info,
)

__all__ = [
    "Simulation",
    "EMF",
    "Current",
    "Species",
    "cart",
    "fcomp",
    "part",
    "phasespace",
    "udist",
    "density",
    "visxd",
    "zdf",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_fc(fc):
    """Accept 'x'/'y'/'z' or an fcomp.cart enum; return the enum."""
    if isinstance(fc, str):
        try:
            return getattr(_core.fcomp.cart, fc)
        except AttributeError as e:
            raise ValueError(
                f"Invalid component {fc!r}; expected 'x', 'y' or 'z'."
            ) from e
    return fc

def _fc_name(fc):
    """Human-readable name for a component (works for str or enum)."""
    return fc if isinstance(fc, str) else fc.name

# ---------------------------------------------------------------------------
# EMF.plot
# ---------------------------------------------------------------------------

def _emf_plot(self, fld, fc, **kwargs):
    """Plot a selected EMF component using ``visxd.plot2d``.

    Parameters
    ----------
    fld : str
        Field type: ``'E'`` (electric) or ``'B'`` (magnetic).
    fc : str or fcomp.cart
        Field component: ``'x'``, ``'y'`` or ``'z'``.
    **kwargs
        Extra keyword arguments forwarded to ``visxd.plot2d``.
    """
    fc_enum = _resolve_fc(fc)
    fc_str  = _fc_name(fc)

    if fld == "E":
        field_enum = _core.emf.field.e
    elif fld == "B":
        field_enum = _core.emf.field.b
    else:
        raise ValueError(f"Invalid field {fld!r}; expected 'E' or 'B'.")

    data   = self.gather(field_enum, fc_enum)
    flabel = f"{fld}_{fc_str}"
    time   = self.iter * self.dt

    box = self.box
    frange = [[0.0, float(box[0])], [0.0, float(box[1])]]

    visxd.plot2d(
        data,
        range  = frange,
        title  = r"$\sf {} $""\n"r"$t = {:g} \;[\sf {}]$".format(
                     flabel, time, r"1 / \omega_n"),
        xtitle = r"$\sf x \;[c / \omega_n]$",
        ytitle = r"$\sf y \;[c / \omega_n]$",
        vtitle = r"$\sf {} \;[m_e c \omega_n e^{{-1}}]$".format(flabel),
        **kwargs,
    )

EMF.plot = _emf_plot

def _emf_vplot(self, fld, **kwargs):
    """Plot in-plane EMF field magnitude using ``visxd.plot2d``.

    Parameters
    ----------
    fld : str
        Field type: ``'E'`` (electric) or ``'B'`` (magnetic).
    **kwargs
        Extra keyword arguments forwarded to ``visxd.plot2d``.
    """

    if fld == "E":
        field_enum = _core.emf.field.e
    elif fld == "B":
        field_enum = _core.emf.field.b
    else:
        raise ValueError(f"Invalid field {fld!r}; expected 'E' or 'B'.")

    xdata   = self.gather(field_enum, _resolve_fc('x'))
    ydata   = self.gather(field_enum, _resolve_fc('y'))
    data    = np.sqrt( np.square( xdata ) + np.square( ydata ) )

    flabel = f"{fld}"
    time   = self.iter * self.dt

    box = self.box
    frange = [[0.0, float(box[0])], [0.0, float(box[1])]]

    visxd.plot2d(
        data,
        range  = frange,
        title  = r"$\sf {} $""\n"r"$t = {:g} \;[\sf {}]$".format(
                     flabel, time, r"1 / \omega_n"),
        xtitle = r"$\sf x \;[c / \omega_n]$",
        ytitle = r"$\sf y \;[c / \omega_n]$",
        vtitle = r"$\sf {} \;[m_e c \omega_n e^{{-1}}]$".format(flabel),
        **kwargs,
    )

EMF.vplot = _emf_vplot

# ---------------------------------------------------------------------------
# Current.plot
# ---------------------------------------------------------------------------

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
    fc_enum = _resolve_fc(fc)
    fc_str  = _fc_name(fc)

    data   = self.gather(fc_enum)
    flabel = f"J_{fc_str}"
    time   = self.iter * self.dt

    box = self.box
    frange = [[0.0, float(box[0])], [0.0, float(box[1])]]

    visxd.plot2d(
        data,
        range  = frange,
        title  = r"$\sf {} $""\n"r"$t = {:g} \;[\sf {}]$".format(
                     flabel, time, r"1 / \omega_n"),
        xtitle = r"$\sf x \;[c / \omega_n]$",
        ytitle = r"$\sf y \;[c / \omega_n]$",
        vtitle = r"$\sf {} \;[e \omega_n^2 / c]$".format(flabel),
        **kwargs,
    )

Current.plot = _current_plot

# ---------------------------------------------------------------------------
# Species.plot
# ---------------------------------------------------------------------------

def _species_plot( self, qx, qy, marker = '.', ms = 0.1, alpha = 0.5, **kwargs ):
    """Do an x,y scatter plot of every particle using the selected quantities.
    Plot is done using visxd.plot1d()

    Parameters
    ----------
    qx : str
        Quantity to use for x axis. Must must be one of 'x', 'y', 'ux', 'uy' or 'uz'
    qy : str
        Quantity to use for y axis. Must must be one of 'x', 'y', 'ux', 'uy' or 'uz'
    marker : string, default = '.'
        Marker to use for plotting data, defaults to dot ('.')
    ms : float, default = 0.1
        Marker size, defaults to 0.1
    alpha : float
        Marker transparency, defaults to 0.5
    **kwargs
        Additional keyword arguments to be passed on to visxd.plot1d()
    """

    qlabels = { 'x':'x', 'y':'y', 'ux':'u_x', 'uy':'u_y', 'uz':'u_z' }
    qunits  = { 'x':r'c/\omega_n', 'y':r'c/\omega_n', 'ux':'c', 'uy':'c', 'uz':'c' }

    time = self.iter * self.dt

    qx_enum = getattr( _core.part.quant, qx )
    qy_enum = getattr( _core.part.quant, qy )

    visxd.plot1d( self.gather(qx_enum), self.gather(qy_enum), marker,
        ms = ms, alpha = alpha,
        xtitle = r"$\sf {} \;[{}]$".format( qlabels[qx], qunits[qx] ),
        ytitle = r"$\sf {} \;[{}]$".format( qlabels[qy], qunits[qy] ),
        title  = r"$\sf {} - {}/{} $""\n"r"$t = {:g} \;[\sf {}]$".format( 
                self.name, qlabels[qy], qlabels[qx], time, r"1 / \omega_n"
            ),
        **kwargs )

Species.plot = _species_plot

# ---------------------------------------------------------------------------
# Species.plot_charge
# ---------------------------------------------------------------------------
def _species_plot_charge(self, **kwargs):
    """Plot charge density using ``visxd.plot2d``.

    Parameters
    ----------
    **kwargs
        Extra keyword arguments forwarded to ``visxd.plot2d``.
    """
    box = self.box
    frange = [[0, box[0]], [0, box[1]]]
    time = self.iter * self.dt

    visxd.plot2d(
        self.get_charge(),
        range=frange,
        title=(
            r"$\sf {} \;charge \;density$""\n"
            r"$t = {:g} \;[\sf {}]$"
        ).format(self.name, time, r"1 / \omega_n"),
        xtitle = r"$\sf {} \;[{}]$".format("x", r"c / \omega_n"),
        ytitle = r"$\sf {} \;[{}]$".format("y", r"c / \omega_n"),
        vtitle = r"$\sf {} - {} \;[{}]$".format(self.name, r"\rho", "n_e"),
        **kwargs,
    )

Species.plot_charge = _species_plot_charge


# ---------------------------------------------------------------------------
# Species.plot_phasespace
# ---------------------------------------------------------------------------
def _resolve_pha_quant(q):
    """Accept 'x'/'y'/'ux'/'uy'/'uz' or an phasespace.quant enum; return the enum."""
    if isinstance(q, str):
        try:
            return getattr(_core.phasespace.quant, q)
        except AttributeError as e:
            raise ValueError(
                f"Invalid phasespace quantity {q!r}; expected 'x', 'y', 'ux', 'uy' or 'uz'."
            ) from e
    return q

def _species_plot_phasespace(self, quant0, range0, size0, 
            quant1 = None, range1 = None, size1 = None,
            marker = '-',
            **kwargs):
    """Plot selected phasespace density

    Arguments
    ----------
    quant0 : str
        Quantitity for x axis, must be one of 'x', 'y', 'ux', 'uy' or 'uz'
    range0 : list
        Limits [min,max] for x axis
    size0 : int
        Size (number of cells) for x axis
    quant1 : str
        Quantitity for y axis, must be one of 'x', 'y', 'ux', 'uy' or 'uz'.
        Defaults to None, which will generate a 1d phasespace. Must be different from quant0
    range1 : list
        Limits [min,max] for y axis
    size1 : int
        Size (number of cells) for y axis        
    marker (str, optional): 
        Marker to use for 1D phasespace plots, defaults to '-'.
    **kwargs
        Additional keyword arguments to be passed on to visxd.plot*()
    """

    qlabels = { 'x':'x', 'y':'y', 'ux':'u_x', 'uy':'u_y', 'uz':'u_z' }
    qunits  = { 'x':r'c/\omega_n', 'y':r'c/\omega_n', 'ux':'c', 'uy':'c', 'uz':'c' }

    time = self.iter * self.dt

    if ( quant1 is None ):
        # 1D phasespace
        x = np.linspace( range0[0], range0[1], num = size0 )
        y = self.get_phasespace( _resolve_pha_quant(quant0), range0, size0 )

        visxd.plot1d( x, y, marker = marker,
            xtitle = r"$\sf {} \;[{}]$".format( qlabels[quant0], qunits[quant0] ),
            ytitle = r"$\sf {} \;[{}]$".format( 'density', 'n_0' ),
            title  = r"$\sf {} - {} $""\n"r"$t = {:g} \;[\sf {}]$".format( 
                self.name, qlabels[quant0], time, r"1 / \omega_n"
                ),
            **kwargs
        )

    else:
        # 2D phasespace
        if ( quant1 == quant0 ):
            raise Exception( "Cannot plot a 2d phasespace with the same quantity on both axis")

        data = self.get_phasespace( 
            _resolve_pha_quant(quant0), range0, size0, 
            _resolve_pha_quant(quant1), range1, size1
        )
        
        visxd.plot2d( data, [ range0, range1 ],
            xtitle = r"$\sf {} \;[{}]$".format( qlabels[quant0], qunits[quant0] ),
            ytitle = r"$\sf {} \;[{}]$".format( qlabels[quant1], qunits[quant1] ),
            title  = r"$\sf {} - {}/{} $""\n"r"$t = {:g} \;[\sf {}]$".format( 
                self.name, qlabels[quant1], qlabels[quant0], time, r"1 / \omega_n"
                ),
            vtitle = r"$\sf {} \;[{}]$".format( 'density', 'n_0' ),
            **kwargs
        )

Species.plot_phasespace = _species_plot_phasespace

del ( 
    _emf_plot, _emf_vplot, _current_plot, 
    _species_plot, _species_plot_charge, _species_plot_phasespace
)