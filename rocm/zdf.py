"""
Python module for reading ZDF data files

Copyright (C) 2017 Instituto Superior Tecnico

This file is part of the ZPIC Educational code suite

The ZPIC Educational code suite is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

The ZPIC Educational code suite is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with the ZPIC Educational code suite. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import numpy as np

class ZDF_Record:
    """ZDF_Record()

    ZDF Datafile record information

    Attributes
    ----------
    pos : int64
        Position in ZDF file
    id : int
        Type id
    name : str
        Record name
    len : uint64
        Additional record length in bytes
    """
    def __init__( self ):
        self.pos   = -1
        self.id    = -1
        self.name  = ''
        self.len   = -1
    
    def version( self ):
        """version()

        Gets record version number

        Returns
        -------
        version : int
            Record version number
        """
        return self.id & 0x0000FFFF

    def type( self ):
        """type( typeTag )

        Gets record type from tag

        Returns
        -------
        type : {'int', 'double', 'string', 'dataset', 'iteration', 'grid_info', 'part_info','unknown'}
            Type of record
        """

        typeID = self.id & 0xFFFF0000

        types = {0x00010000: "int",
                 0x00020000: "double",
                 0x00030000: "string",
                 0x00100000: "dataset",
                 0x00200000: "iteration",
                 0x00210000: "grid_info",
                 0x00220000: "part_info", }

        if (typeID in types):
            return types[typeID]
        else:
            return "unknown"


class ZDF_Iteration:
    """ZDF_Iteration()
    
    Class describing iteration information.

    Attributes
    ----------
    name : str
        Name for iteration metadata (usually set to 'ITERATION')
    n : int
        Iteration value
    t : float
        Time value
    tunits : str
        Units used for time value
    """
    def __init__( self ):
        self.name = ""
        self.n = 0
        self.t = 0.0
        self.tunits = ""

class ZDF_Grid_Axis:
    """ZDF_Grid_Axis()

    Class describing grid axis

    Attributes
    ----------
    name : str
        Axis name
    type : {0,1,2}
        Axis type, must be one of 0 (linear), 1 (log10) or 2 (log2)
    min : float
        Minimum axis value
    max : float
        Maximum axis value
    label : str
        Axis label
    units : str
        Axis values data units
    """
    def __init__( self ):
        self.name = ''
        self.type = 0
        self.min = 0.
        self.max = 0.
        self.label = ''
        self.units = ''

class ZDF_Grid_Info:
    """ZDF_Grid_Info()

    Grid dataset information

    Attributes
    ----------
    ndims : int
        Dimensionality of dataset
    nx : list of int (ndims)
        Number of grid points in each direction
    label : str
        Dataset label
    units : str
        Dataset units
    has_axis : bool
        True if the dataset includes axis information
    axis : list of ZDF_Grid_Axis (ndims)
        Information on each axis, when available
    """
    def __init__( self ):
        self.name = ''
        self.ndims = 0
        self.nx = []
        self.label = ''
        self.units = ''
        self.has_axis = 0
        self.axis = []

class ZDF_Part_Info:
    """ZDF_Part_Info()

    Particle dataset information

    Attributes
    ----------
    name : str
        Particle dataset name
    label : str
        Particle dataset label
    nquants : int
        Number of quantities per particle
    quants : list of str (nquants)
        Name of individual quantities
    qlabels : dictionary
        Labels for each quantity
    qunits : dictionary
        Units for each quantity
    nparts: int
        Number of particles in dataset
    """
    def __init__( self ):
        self.name = ''
        self.label = ''
        self.nquants = 0
        self.quants = []
        self.qlabels = dict()
        self.qunits = dict()
        self.nparts = 0

class ZDFfile:
    """ZDFfile( file_name )
    
    ZDF data file class

    Parameters
    ----------
    file_name : str
        File name of ZDF data file, should include path
    """
    def __init__(self, file_name):
        self.__file = open(file_name, "rb")

        # Check magic number
        magic = self.__file.read(4)
        if (magic != b'ZDF1'):
            print('File is not a proper ZDF file, aborting', file=sys.stderr)
            self.__file.close

    def close(self):
        """close()

        Closes ZDF file
        """
        self.__file.close

    def __read_uint32(self):
        data = np.fromfile(self.__file,dtype='<u4',count=1)
        if ( data.size == 0 ):
            return False
        else:
            return data[0]

    def __read_int32(self):
        return np.fromfile(self.__file,dtype='<i4',count=1)[0]

    def __read_uint64(self):
        return np.fromfile(self.__file,dtype='<u8',count=1)[0]

    def __read_int64(self):
        return np.fromfile(self.__file,dtype='<i8',count=1)[0]

    def __read_float32(self):
        return np.fromfile(self.__file,dtype='<f4',count=1)[0]

    def __read_float64(self):
        return np.fromfile(self.__file,dtype='<f8',count=1)[0]

    def __read_string(self):

        length = self.__read_uint32()

        if (length > 0):
            data = self.__file.read(length)
            fstring = data.decode()

            # Data is always written in blocks of 4 byt
            pad = ((length - 1) // 4 + 1) * 4 - length
            if ( pad > 0 ):
                self.__file.seek(pad,1)
        else:
            fstring = ""

        return fstring

    def record_type(self, typeTag):
        version = typeTag & 0x0000FFFF
        typeID = typeTag & 0xFFFF0000

        types = {0x00010000: "int",
                 0x00020000: "double",
                 0x00030000: "string",
                 0x00100000: "dataset",
                 0x00110000: "cdset_start",
                 0x00120000: "cdset_chunk",
                 0x00130000: "cdset_end",
                 0x00200000: "iteration",
                 0x00210000: "grid_info",
                 0x00220000: "part_info",
                 0x00230000: "track_info", }

        if (typeID in types):
            return types[typeID]
        else:
            return "unknown"

# -----------------------------------------------------------------------------
# Array datatypes
# -----------------------------------------------------------------------------
    
    def __read_int32_arr(self, nx):
        size = np.prod(nx)
        data = np.fromfile(self.__file,dtype='<i4',count=size)
        data.shape = np.flip(nx)
        return data

    def __read_uint32_arr(self, nx):
        size = np.prod(nx)
        data = np.fromfile(self.__file,dtype='<u4',count=size)
        data.shape = np.flip(nx)
        return data

    def __read_int64_arr(self, nx):
        size = np.prod(nx)
        data = np.fromfile(self.__file,dtype='<i8',count=size)
        data.shape = np.flip(nx)
        return data

    def __read_uint64_arr(self, nx):
        size = np.prod(nx)
        data = np.fromfile(self.__file,dtype='<u8',count=size)
        data.shape = np.flip(nx)
        return data

    def __read_float32_arr(self, nx):
        size = np.prod(nx)
        data = np.fromfile(self.__file,dtype='<f4',count=size)
        data.shape = np.flip(nx)
        return data

    def __read_float64_arr(self, nx):
        size = np.prod(nx)
        data = np.fromfile(self.__file,dtype='<f8',count=size)
        data.shape = np.flip(nx)
        return data
    
    def __read_arr( self, dtype, nx ):
        
        if ( dtype == 5 ):
            data = self.__read_int32_arr(nx)           
        elif ( dtype == 3 ):
            data = self.__read_uint32_arr(nx)           
        elif ( dtype == 7 ):
            data = self.__read_int64_arr(nx)           
        elif ( dtype == 8 ):
            data = self.__read_uint64_arr(nx)           
        elif ( dtype == 9 ):
            data = self.__read_float32_arr(nx)           
        elif ( dtype == 10 ):
            data = self.__read_float64_arr(nx)           
        else:
            print( '(*error*) ZDF: Data type not yet supported.' , file=sys.stderr)
            data = False
        
        return data
        
# -----------------------------------------------------------------------------
# Low level interfaces
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Read record header
# -----------------------------------------------------------------------------

    def read_record(self, skip=False):
        """read_record(skip=False)

        Reads current record information from file

        Parameters
        ----------
        skip : bool, optional
            If set to True, skip to next record after reading record
            header data
        """

        rec = ZDF_Record()
        rec.pos = self.__file.tell()

        # Read record id and check for EOF
        rec.id = self.__read_uint32()

        if (rec.id is False):
            # If end of file return false
            return False

        # Read name and length
        rec.name = self.__read_string()
        rec.len  = self.__read_uint64()

        # If requested, skip over to next record
        if (skip):
            self.__file.seek(rec.len, 1)

        return rec
    
    def __record_skip( self, rec ):
        self.__file.seek(rec.len, 1)

# -----------------------------------------------------------------------------
# Read string
# -----------------------------------------------------------------------------

    def read_string(self, rec = False):
        """read_string(rec = False)

        Reads string record from data file
        
        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        string : str
            String data
        """
        if ( rec is False ):
            rec = self.read_record()
        
        fstring = self.__read_string()
        return fstring

# -----------------------------------------------------------------------------
# Read iteration
# -----------------------------------------------------------------------------

    def read_iteration(self, rec = False):
        """read_iteration( rec = False )

        Read iteration record from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        iteration : ZDF_Iteration()
            Iteration data
        """
        if ( rec is False ):
            rec = self.read_record()
        iteration = ZDF_Iteration()
        iteration.name = rec.name
        iteration.n = self.__read_int32()
        iteration.t = self.__read_float64()
        iteration.tunits = self.__read_string()
        return iteration

# -----------------------------------------------------------------------------
# Read grid info
# -----------------------------------------------------------------------------

    def read_grid_info(self, rec = False):
        """read_grid_info( rec = False )

        Read grid information record from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        info : ZDF_Grid_Info()
            Grid information data
        """
        if ( rec is False ):
            rec = self.read_record()

        # Maximum supported version
        max_version = 0x00000001

        # Get version
        version = rec.version()
        if ( version > max_version ):
            print( '(*error*) ZDF: Grid info version is higher than supported.' , file=sys.stderr)
            print( '(*error*) ZDF: Please update the code to a newer version.' , file=sys.stderr)
            return False

        info = ZDF_Grid_Info()
        
        info.name = rec.name
        info.ndims = self.__read_uint32()
        info.nx = self.__read_uint64_arr( info.ndims )

        info.label = self.__read_string()
        info.units = self.__read_string()
        info.has_axis = self.__read_int32()

        if ( info.has_axis ):
            for i in range(info.ndims):
                ax = ZDF_Grid_Axis()
                if (version > 0):
                    ax.name  = self.__read_string()
                else:
                    ax.name = 'axis_{}'.format(i)
                ax.type  = self.__read_int32()
                ax.min   = self.__read_float64()
                ax.max   = self.__read_float64()
                ax.label = self.__read_string()
                ax.units = self.__read_string()
                info.axis.append(ax)

        return info

# -----------------------------------------------------------------------------
# Read particle info
# -----------------------------------------------------------------------------
    
    def read_part_info(self, rec = False):
        """read_part_info( rec = False )

        Read particle information record from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        iteration : ZDF_Part_Info()
            Iteration data
        """
        if ( rec is False ):
            rec = self.read_record()

        # Maximum supported version
        max_version = 0x00000002

        # Get version
        version = rec.version()
        if ( version > max_version ):
            print( '(*error*) ZDF: Particles info version is higher than supported.' , file=sys.stderr)
            print( '(*error*) ZDF: Please update the code to a newer version.' , file=sys.stderr)
            return False
        
        info = ZDF_Part_Info()
        info.name = rec.name
        info.label = self.__read_string()
        
        if ( version >= 1 ):
            # version 1
            info.nparts = self.__read_uint64()
            info.nquants = self.__read_uint32()
            
            for i in range(info.nquants):
                info.quants.append( self.__read_string() )
            for q in info.quants:
                info.qlabels[q] = self.__read_string()            
            for q in info.quants:
                info.qunits[q] = self.__read_string()
            
        else:
            # version 0
            info.nquants = self.__read_uint32()

            for i in range(info.nquants):
                info.quants.append( self.__read_string() )

            # version 0 does not have label information
            for q in info.quants:
                info.qlabels[q] = q

            for q in info.quants:
                info.qunits[q] = self.__read_string()

            info.nparts = self.__read_uint64()

        return info

# -----------------------------------------------------------------------------
# Read dataset
# -----------------------------------------------------------------------------

    def read_dataset(self, rec = False):
        """read_dataset()

        Read dataset from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        data : numpy.ndarray
            Numpy ndarray with data
        """
        if ( rec is False ):
            rec = self.read_record()
            
        if ( self.record_type(rec.id) != 'dataset' ):
            print( '(*error*) ZDF: Expected dataset record but found {} instead.'.format(self.record_type(rec.id)),
                  file=sys.stderr)
            return False
            
        
        # Maximum supported version
        max_version = 0x00000002
        
        # Get version
        version = rec.version()
        if ( version > max_version ):
            print( '(*error*) ZDF: Dataset version is higher than supported.' , file=sys.stderr)
            print( '(*error*) ZDF: Please update the code to a newer version.' , file=sys.stderr)
            return False
        
        # Version 0x0001 includes id tag
        if ( version >= 1 ):
            id = self.__read_uint32()
        else:
            id = 0
        
        data_type = self.__read_int32()
        ndims = self.__read_uint32()
        nx        = self.__read_uint64_arr(ndims)
        data      = self.__read_arr( data_type, nx )

        return data

# -----------------------------------------------------------------------------
# Retrieve list of file contents / print file contents
# -----------------------------------------------------------------------------

    def list(self, printRec=False):
        """list( printRec=False )

        Gets a list of file contents and optionally prints it to screen

        Parameters
        ----------
        printRec : bool, optional
            If set to True will print all records found in the file,
            defaults to False.
        """
        # Position file after magic number
        self.__file.seek(4)

        rec_list = []
        while True:
            rec = self.read_record(skip=True)
            if (rec is False):
                break
            else:
                rec_list.append(rec)

        if (printRec and (len(rec_list) > 0)):
            print('Position     Size(bytes)  Type        Name')
            print('-----------------------------------------------------')
            for rec in rec_list:
                print('{:#010x}   {:#010x}   {:11}  {}'.format(
                    rec.pos, rec.len,
                    self.record_type(rec.id), rec.name)
                     )

        return rec_list

# -----------------------------------------------------------------------------
# High level interfaces
# -----------------------------------------------------------------------------

class ZDF_Info:
    """ZDF_Info()
    ZDF File information

    Attributes
    ----------
    type : {'grid','particles'}
        Type of ZDF file
    grid : ZDF_Grid_Info
        Grid information for grid files
    particles : ZDF_Part_Info
        Particle information for particle files
    iteration : ZDF_Iteration
        Iteration information

    """
    def __init__(self):
        self.type = ""
        self.grid = None
        self.particles = None
        self.iteration = None


def info( file_name ):
    """info( file_name )

    Gets metadata for a ZDF file

    Parameters
    ----------
    file_name : str
        File name of ZDF data file, should include path
    
    Returns
    -------
    info : ZDF_Info
        File information. If file is invalid False is returned.
    """
    # Open file
    zdf = ZDFfile( file_name )

    # Check file type and read metadata
    info = ZDF_Info()
    info.type = zdf.read_string()
    if ( info.type == "grid" ):
        info.grid = zdf.read_grid_info()
    elif ( info.type == "particles" ):
        info.particles = zdf.read_part_info()
    else:
        print("File is not a valid ZDF grid or particles file", file=sys.stderr)
        zdf.close()
        return False

    # Read iteration info
    info.iteration = zdf.read_iteration()

    # Close file
    zdf.close()

    return info

def read( file_name ):
    """read( file_name )

    Reads all data in a ZDF file

    Parameters
    ----------
    file_name : str
        File name of ZDF data file, should include path
    
    Returns
    -------
    (data, info) : ( numpy.ndarray | dictionary, ZDF_Info )
        Tuple containing file data and metadata. Data will be a
        numpy.ndarray for grid data, and a dictionary of numpy.array for
        particle data (one entry per quantity). Metadata is returned as a
        ZDF_Info object. If file is invalid False is returned.
    """
    # Open file
    zdf = ZDFfile( file_name )

    # Check file type and read metadata
    info = ZDF_Info()

    info.type = zdf.read_string()
    if ( info.type == "grid" ):
        info.grid = zdf.read_grid_info()
        info.iteration = zdf.read_iteration()
        data = zdf.read_dataset()
    elif ( info.type == "particles" ):
        info.particles = zdf.read_part_info()
        info.iteration = zdf.read_iteration()

        # Read all quantities
        data = dict()
        for q in info.particles.quants:
            data[q] = zdf.read_dataset()
    else:
        print("File is not a valid ZDF grid or particles file", file=sys.stderr)
        zdf.close()
        return False

    #Close file
    zdf.close()

    return (data,info)

def list(file_name, printRec=False):
    """list( printRec=False )

    Gets a list of file contents and optionally print it to screen

    Parameters
    ----------
    file_name : str
        File name of ZDF data file, should include path
        
    printRec : bool, optional
        If set to True will print all records found in the file,
        defaults to False.
    """
    zdf = ZDFfile(file_name)
    zdf.list(printRec)
    zdf.close()

