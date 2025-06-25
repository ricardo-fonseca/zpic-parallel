import zdf

import sys
import os.path

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

def grid1d( filename : str, xlim = None, grid : bool = None, scale = None ):
    """Generates a line plot from a 1D grid file

    Args:
        filename (str): _description_
        xlim ( float(2), optional): Force x-axis range. Defaults to None.
        grid (bool, optional): Use a grid on the plot. Defaults to True.
        scale ( float(2), optional): Tuple describing parameter for linearly
            scaling the data before plotting. Defaults to None.
    """
    
    if ( not os.path.exists(filename) ):
        # raise FileNotFoundError( filename ) 
        print("(*error*) file {} not found.".format(filename), file = sys.stderr )
        return

    (ydata, info) = zdf.read(filename)

    if ( info.type != "grid" ):
        print("(*error*) file {} is not a grid file".format(filename))
        return
    
    if ( info.grid.ndims != 1 ):
        print("(*error*) file {} is not a 1D grid file".format(filename))
        return

    # Linearly scale data if requested
    if ( scale ):
        ydata = ydata * scale[0] + scale[1]


    xlabel = info.grid.axis[0].label + ' [' + info.grid.axis[0].units + ']'
    ylabel = info.grid.label + ' [' + info.grid.units + ']'
    title = info.grid.label
    timeLabel = "t = {:g}\\,[{:s}]".format(info.iteration.t, info.iteration.tunits)

    delta = ( info.grid.axis[0].max - info.grid.axis[0].min ) / info.grid.nx[0]
    start = info.grid.axis[0].min + 0.5*delta
    stop  = info.grid.axis[0].max - 0.5*delta

    xdata = np.linspace( start, stop, num = info.grid.nx[0] )

    plt.plot(xdata, ydata)
    plt.title(r'$\sf{' + title + r'}$' + '\n' + r'$\sf{'+ timeLabel+ r'}$')

    plt.xlabel(r'$\sf{' +xlabel+ r'}$')
    plt.ylabel(r'$\sf{' +ylabel+ r'}$')

    plt.grid(True)
    plt.show()

def grid2d( filename : str, xlim = None, ylim = None, grid = False, cmap = None, norm = None,
    vsim = False, vmin = None, vmax = None, scale = None, shift = None ):
    """Generates a colormap plot from a 2D grid zdf file

    Args:
        filename (str):
            Name of ZDF file to open
        xlim (tuple, optional):
            Lower and upper limits of x axis. Defaults to the x limits of the
            grid data.
        ylim (tuple, optional):
            Lower and upper limits of y axis. Defaults to the y limits of the
            grid data.
        grid (bool, optional):
            Display a grid on top of colormap. Defaults to False.
        cmap (str, optional):
            Name of the colormap to use. Defaults to the matplotlib imshow() 
            colormap.
        vsim:
            Setup a symmetric value scale [ -max(|val|), max(|val|) ]. Defaults to setting
            the value scale to [ min, max ]
    """

    if ( not os.path.exists(filename) ):
        # raise FileNotFoundError( filename ) 
        print("(*error*) file {} not found.".format(filename), file = sys.stderr )
        return

    (data, info) = zdf.read(filename)

    if ( info.type != "grid" ):
        print("(*error*) file {} is not a grid file".format(filename))
        return
    
    if ( info.grid.ndims != 2 ):
        print("(*error*) file {} is not a 2D grid file".format(filename))
        return

    range = [
        [info.grid.axis[0].min, info.grid.axis[0].max],
        [info.grid.axis[1].min, info.grid.axis[1].max]
    ]

    # Linearly scale data if requested
    if ( scale ):
        data = data * scale[0] + scale[1]
    
    if ( shift ):
        data = np.roll( data, shift, axis=(1,0) )

    if ( vsim ):
        amax = np.amax( np.abs(data) )
        plt.imshow( data, interpolation = 'nearest', origin = 'lower',
            vmin = -amax, vmax = +amax, norm = norm,
            extent = ( range[0][0], range[0][1], range[1][0], range[1][1] ),
            aspect = 'auto', cmap=cmap )
    else:
        plt.imshow( data, interpolation = 'nearest', origin = 'lower',
            vmin = vmin, vmax = vmax, norm = norm,
            extent = ( range[0][0], range[0][1], range[1][0], range[1][1] ),
            aspect = 'auto', cmap=cmap )

    zlabel = "{}\\,[{:s}]".format( info.grid.label, info.grid.units )

    plt.colorbar().set_label(r'$\sf{' + zlabel + r'}$')

    xlabel = "{}\\,[{:s}]".format( info.grid.axis[0].label, info.grid.axis[0].units )
    ylabel = "{}\\,[{:s}]".format( info.grid.axis[1].label, info.grid.axis[1].units )

    plt.xlabel(r'$\sf{' + xlabel + r'}$')
    plt.ylabel(r'$\sf{' + ylabel + r'}$')

    plt.title("$\\sf {} $\nt = ${:g}$ [$\\sf {}$]".format(
        info.grid.label.replace(" ","\\;"),
        info.iteration.t,
        info.iteration.tunits))

    if ( xlim ):
        plt.xlim(xlim)
    if ( ylim ):
        plt.ylim(ylim)

    plt.grid(grid)

    plt.show()

def grid( filename : str, xlim = None, ylim = None, grid : bool = False, cmap = None, norm = None,
    vsim = False, vmin = None, vmax = None, scale = None, shift = None ):
    """Generates a plot from 1D or 2D grids.

    This works as driver for grid1d and grid2d routines.

    Args:
        filename (str): _description_
        xlim (_type_, optional): _description_. Defaults to None.
        ylim (_type_, optional): _description_. Defaults to None.
        grid (bool, optional): _description_. Defaults to False.
        cmap (_type_, optional): _description_. Defaults to None.
        vsim (bool, optional): _description_. Defaults to False.
        vmin (_type_, optional): _description_. Defaults to None.
        vmax (_type_, optional): _description_. Defaults to None.
        scale (_type_, optional): _description_. Defaults to None.
    """

    info = zdf.info(filename)

    if ( info.type != "grid" ):
        print("(*error*) file {} is not a grid file".format(filename))
    elif ( info.grid.ndims == 1 ):
        grid1d( filename, xlim = xlim, grid = grid, scale = scale )
    elif ( info.grid.ndims == 2 ):
        grid2d( filename, xlim = xlim, ylim = ylim, grid = grid, cmap = cmap, norm = norm,
            vsim = vsim, vmin = vmin, vmax = vmax, scale = scale, shift = shift )
    else:
        print("(*error*) file {} - unsupported grid dimensions ({}).".format(filename, info.grid.ndims))


def vfield2d( filex, filey, xlim = None, ylim = None, grid = False, cmap = None,
              vmin = 0, vmax = None, title = None, shift = None ):
    """Generates a colormap plot 

    Args:
        filex (str):
            Name of ZDF file to open for the x field component
        filey (str):
            Name of ZDF file to open for the y field component
        xlim (tuple, optional):
            Lower and upper limits of x axis. Defaults to the x limits of the
            grid data.
        ylim (tuple, optional):
            Lower and upper limits of y axis. Defaults to the y limits of the
            grid data.
        grid (bool, optional):
            Display a grid on top of colormap. Defaults to False.
        cmap (str, optional):
            Name of the colormap to use. Defaults to the matplotlib imshow() 
            colormap.
        vsim:
            Setup a symmetric value scale [ -max(|val|), max(|val|) ]. Defaults to setting
            the value scale to [ min, max ]

    """

    if (( not os.path.exists(filex) ) or ( not os.path.exists(filey) )):
        print("(*error*) files missing:")
        if ( not os.path.exists(filex) ):
            print("(*error*) file {} not found.".format(filex), file = sys.stderr )
        if ( not os.path.exists(filey) ):
            print("(*error*) file {} not found.".format(filey), file = sys.stderr )
        return

    if ( filex == filey ):
        print("(*error*) the 2 files are the same: {}".format(filex), file = sys.stderr )
        return

    # Check filex
    infox = zdf.info(filex)

    if ( infox.type != "grid" ):
        print("(*error*) file {} is not a grid file".format(filex))
        return
    
    if ( infox.grid.ndims != 2 ):
        print("(*error*) file {} is not a 2D grid file".format(filex))
        return

    range = [
        [infox.grid.axis[0].min, infox.grid.axis[0].max],
        [infox.grid.axis[1].min, infox.grid.axis[1].max]
    ]

    # Check filey
    infoy = zdf.info(filey)

    if ( infoy.type != "grid" ):
        print("(*error*) file {} is not a grid file".format(filey))
        return
    
    if ( infoy.grid.ndims != 2 ):
        print("(*error*) file {} is not a 2D grid file".format(filey))
        return

    if (( infox.grid.nx[0] != infoy.grid.nx[0] ) or
        ( infox.grid.nx[1] != infoy.grid.nx[1] )):
        print("(*error*) files {} / {} don't have the same grid dimensions".format(filex, filey))
        return

    # Everything seems ok proceed

    (datax, infox) = zdf.read( filex )
    (datay, infoy) = zdf.read( filey )

    range = [
        [infox.grid.axis[0].min, infox.grid.axis[0].max],
        [infox.grid.axis[1].min, infox.grid.axis[1].max]
    ]

    data = np.sqrt( np.square( datax ) + np.square( datay ) )

    if ( shift ):
        data = np.roll( data, shift, axis=(1,0) )

    plt.imshow( data, interpolation = 'nearest', origin = 'lower',
            extent = ( range[0][0], range[0][1], range[1][0], range[1][1] ),
            aspect = 'auto', cmap=cmap, vmin = vmin, vmax = vmax )

    if ( not title ):
        title = '\\sqrt{' + infox.grid.label + '^2 + ' + infoy.grid.label + '^2 }'

    zlabel = "{}\\;[{:s}]".format( title, infox.grid.units )

    plt.colorbar().set_label(r'$\sf{' + zlabel + r'}$')

    xlabel = "{}\\,[{:s}]".format( infox.grid.axis[0].label, infox.grid.axis[0].units )
    ylabel = "{}\\,[{:s}]".format( infox.grid.axis[1].label, infox.grid.axis[1].units )

    plt.xlabel(r'$\sf{' + xlabel + r'}$')
    plt.ylabel(r'$\sf{' + ylabel + r'}$')

    plt.title("$\\sf {} $\nt = ${:g}$ [$\\sf {}$]".format(
        title.replace(" ","\\;"),
        infox.iteration.t,
        infox.iteration.tunits))

    if ( xlim ):
        plt.xlim(xlim)
    if ( ylim ):
        plt.ylim(ylim)

    plt.grid(grid)

    plt.show()

def part2D( filename, qx, qy, xlim = None, ylim = None, grid = True, 
    marker = '.', ms = 1, alpha = 1 ):
    """Generates an (x,y) scatter plot from a ZDF particle file.

    Args:
        filename (str):
            Name of ZDF file to open
        qx (str):
            X axis quantity, usually one of "x", "y", "ux", "uy", "uz", etc.
        qy (str): _description_
            Y axis quantity, usually one of "x", "y", "ux", "uy", "uz", etc.
        xlim (tuple, optional):
            Lower and upper limits of x axis. Defaults to the limits of the "qx" particle data.
        ylim (tuple, optional):
            Lower and upper limits of y axis. Defaults to the limits of the "qy" particle data.
        grid (bool, optional):
            Display a grid on top of scatter plot. Defaults to True.
        marker (str, optional)
            Plot marker to use for the scatter plot. Defaults to '.'.
        ms (int, optional):
            Marker size to use for the scatter plot. Defaults to 1.
        alpha (int, optional):
            Marker opacity to use for the scatter plot. Defaults to 1.
    """

    if ( not os.path.exists(filename) ):
        # raise FileNotFoundError( filename ) 
        print("(*error*) file {} not found.".format(filename), file = sys.stderr )
        return

    (particles, info) = zdf.read(filename)

    if ( info.type != "particles" ):
        print("(*error*) file {} is not a particles file".format(filename))
        return
    
    if ( not qx in info.particles.quants ):
        print("(*error*) '{}' quantity (q1) is not present in file".format(qx) )
        return

    if ( not qy in info.particles.quants ):
        print("(*error*) '{}' quantity (q2) is not present in file".format(qy) )
        return

    x = particles[qx]
    y = particles[qy]

    plt.plot(x, y, marker, ms=ms, alpha = alpha)

    title = "{}/{}".format( info.particles.qlabels[qy], info.particles.qlabels[qx])
    timeLabel = "t = {:g}\\,[{:s}]".format(info.iteration.t, info.iteration.tunits)

    plt.title(r'$\sf{' + title + r'}$' + '\n' + r'$\sf{' + timeLabel + r'}$')

    xlabel = "{}\\,[{:s}]".format( info.particles.qlabels[qx], info.particles.qunits[qx] )
    ylabel = "{}\\,[{:s}]".format( info.particles.qlabels[qy], info.particles.qunits[qy] )

    plt.xlabel(r'$\sf{' + xlabel + r'}$')
    plt.ylabel(r'$\sf{' + ylabel + r'}$')

    if ( xlim ):
        plt.xlim(xlim)
    if ( ylim ):
        plt.ylim(ylim)

    plt.grid(grid)

    plt.show()

def histogram( filename, q, bins = 128, range = None, density = True, log = False, color = None, histtype = 'bar' ):
    """Generates a histogram (frequency) plot from a ZDF particle file.

    Args:
        filename (str):
            Name of ZDF file to open
        q (str):
            Quantity to use, usually one of "x", "y", "ux", "uy", "uz", etc.
        bins (int, optional):
            Number of bins to use for the histogram. Defaults to 128.
        range (tuple, optional):
            Lower and upper limits of the histogram. Defaults to minimum and maximum values of the selected quantity.
        density (bool, optional):
            Plot a probability density (bin count divided by the total number of counts and the bin width) instead of 
            bin count. Defaults to True.
        log (bool, optional):
            Use log scale for histogram axis. Defaults to False.
        color (str, optional):
            Color for plot. Defaults to the matplotlib plot color.
        histtype (str, optional):
            Type of histogram to draw, check matplotlib histogram documentation for details. Defaults to 'bar'.
    """

    if ( not os.path.exists(filename) ):
        # raise FileNotFoundError( filename ) 
        print("(*error*) file {} not found.".format(filename), file = sys.stderr )
        return

    (particles, info) = zdf.read(filename)

    if ( info.type != "particles" ):
        print("(*error*) file {} is not a particles file".format(filename))
        return
    
    if ( not q in info.particles.quants ):
        print("(*error*) '{}' quantity (q1) is not present in file".format(q) )
        return
    
    data = particles[q]

    plt.hist( data, bins = bins, range = range, density = density, log = log, color = color, histtype = histtype )
    title = "{} - {}".format( info.particles.label, info.particles.qlabels[q])
    timeLabel = "t = {:g}\\,[{:s}]".format(info.iteration.t, info.iteration.tunits)

    plt.title(r'$\sf{' + title + r'}$' + '\n' + r'$\sf{' + timeLabel + r'}$')

    xlabel = "{}\\,[{:s}]".format( info.particles.qlabels[q], info.particles.qunits[q] )

    plt.xlabel(r'$\sf{' + xlabel + r'}$')
    plt.ylabel(r'$\sf{' + "n" + r'}$')

    plt.show()

def grid2d_fft( filename : str, xlim = None, ylim = None, grid = False, cmap = None, norm = None,
    vmin = None, vmax = None, plot = "abs" ):
    """Generates a colormap plot from the FFT of a 2D grid zdf file

    Args:
        filename (str): Name of ZDF file to open
        xlim (tuple, optional): Lower and upper limits of x axis. Defaults to the kx limits of the
            grid data FFT.
        ylim (tuple, optional): Lower and upper limits of y axis. Defaults to the ky limits of the
            grid data FFT.
        grid (bool, optional): Display a grid on top of colormap. Defaults to False.
        cmap (str, optional): Name of the colormap to use. Defaults to the matplotlib imshow() 
            colormap
        norm (matplotlib.colors, optional): Colormap normalization to use. Defaults to None (linear).
        vmin (float, optional): Max. value to plot. Defaults to the smallest value in the data FFT.
        vmax (float, optional): Min. value to plot. Defaults to the largest value in the data FFT.
        plot (str, optional): Type of plot to produce: absolute value of FFT ("abs"), real part of
            the FFT ("real") or imaginary part of the FFT ("imag"). Defaults to "abs".
    """

    if ( not os.path.exists(filename) ):
        # raise FileNotFoundError( filename ) 
        print("(*error*) file {} not found.".format(filename), file = sys.stderr )
        return

    (data, info) = zdf.read(filename)

    if ( info.type != "grid" ):
        print("(*error*) file {} is not a grid file".format(filename))
        return
    
    if ( info.grid.ndims != 2 ):
        print("(*error*) file {} is not a 2D grid file".format(filename))
        return
    
    if ( not plot in ("abs","real","imag") ):
        print("(*error*) Invalid plot parameter ({}), must be one of 'abs', 'real' or 'imag'.".
            format(plot), file = sys.stderr )
        return

    # Get fft of data
    data = np.fft.fft2( data )
    data = np.fft.fftshift( data )

    dx = (info.grid.axis[0].max - info.grid.axis[0].min)/info.grid.nx[0]
    dy = (info.grid.axis[1].max - info.grid.axis[1].min)/info.grid.nx[1]

    nfx = np.pi / dx
    nfy = np.pi / dy

    range = [
        [-nfx, nfx],
        [-nfy, nfy]
    ]

    if ( plot == "real"):
        data = np.real(data)
        zlabel = "Re\\left[\\mathcal{F}(" + info.grid.label + ")\\right]"
    elif ( plot == "imag"):
        data = np.imag(data)
        zlabel = "Im\\left[\\mathcal{F}(" + info.grid.label + ")\\right]"
    else:
        data = np.abs(data)
        zlabel = "\\left|\\mathcal{F}(" + info.grid.label + ")\\right|"

    plt.imshow( data, interpolation = 'nearest', origin = 'lower',
        vmin = vmin, vmax = vmax, norm = norm,
        extent = ( range[0][0], range[0][1], range[1][0], range[1][1] ),
        aspect = 'auto', cmap=cmap )

    plt.colorbar().set_label(r'$\sf{' + zlabel + r'}$')

    xlabel = "k_{" + info.grid.axis[0].label + "}"
    ylabel = "k_{" + info.grid.axis[1].label + "}"

    plt.xlabel(r'$\sf{' + xlabel + r'}$')
    plt.ylabel(r'$\sf{' + ylabel + r'}$')

    plt.title("$\\sf {} $\nt = ${:g}$ [$\\sf {}$]".format(
        zlabel.replace(" ","\\;"),
        info.iteration.t,
        info.iteration.tunits))

    if ( xlim ):
        plt.xlim(xlim)
    if ( ylim ):
        plt.ylim(ylim)

    plt.grid(grid)

    plt.show()


def plot_part( part, iter = None, qx = "x", qy = "y", xlim = None, ylim = None, grid = True, 
    marker = '.', ms = 1, alpha = 1 ):
    
    file = "{}-{:06d}.zdf".format(part, iter)

    if ( os.path.exists(file) ):
        print("Plotting {}".format(file))
        part2D( file, qx, qy, xlim = xlim, ylim = ylim, grid = grid,
               marker = marker, ms = ms, alpha = alpha )
    else:
        print("(*error*) file {} not found.".format(file), file = sys.stderr )


def plot_data( fld, iter = None, xlim = None, ylim = None, cmap = None, norm = None,
    vsim = None, vmin = None, vmax = None, scale = None, shift = None ):
    
    file = "{}-{:06d}.zdf".format(fld, iter)

    if ( os.path.exists(file) ):
        print("Plotting {}".format(file))
        grid( file, xlim = xlim, ylim = ylim, grid = False, cmap = cmap, norm = norm,
            vsim = vsim, vmin = vmin, vmax = vmax, scale = scale, shift = shift )
    else:
        print("(*error*) file {} not found.".format(file), file = sys.stderr )

def plot_vfield2d( fld, iter, xlim = None, ylim = None, grid = False, norm = None, cmap = None, shift = None ):
    print("Plotting {} in plane field for iteration {}.".format(fld,iter))
    
    filex = "{}x-{:06d}.zdf".format(fld, iter)
    filey = "{}y-{:06d}.zdf".format(fld, iter)

    if ( not cmap ):
        cmap = 'YlOrBr'
    vfield2d( filex, filey, xlim = xlim, ylim = ylim, grid = grid, cmap = cmap,
        title = "In-plane {} field".format(fld), shift = shift )
    
    filez = "{}z-{:06d}.zdf".format(fld, iter)

    if ( os.path.exists(filez) ):
        if ( not norm ):
            norm = colors.CenteredNorm()
        print("Plotting {} out of plane field for iteration {}.".format(fld,iter))
        grid2d(filez, xlim = xlim, ylim = ylim, grid = grid, cmap = 'BrBG', norm = norm, shift = shift )
