#!/usr/bin/env python3
"""
zdf - Read and write ZDF files

Reading
-------
    data, info = zdf.read( 'e1-000100.zdf' )      # grid, particle or track files
    info = zdf.info( 'e1-000100.zdf' )            # metadata only
    zdf.list( 'e1-000100.zdf' )                   # print file contents

Writing
-------
    zdf.save_grid( 'e1.zdf', e1,
        info = zdf.ZDF_Grid_Info( name = 'e1', label = 'E_1',
            axis = [ zdf.ZDF_Grid_Axis( name = 'x', min = 0, max = 10, label = 'x_1' ),
                     zdf.ZDF_Grid_Axis( name = 'y', min = 0, max = 5,  label = 'x_2' ) ] ),
        iteration = zdf.ZDF_Iteration( n = 100, t = 10.0, tunits = '1/\\omega_p' ) )

    zdf.save_particles( 'electrons.zdf', { 'x1' : x1, 'p1' : p1 } )

Lower level access (including chunked datasets) is available through the ZDFfile class,
which can be used as a context manager:

    with zdf.ZDFfile( 'data.zdf', 'w' ) as f:
        f.add_string( 'TYPE', 'grid' )
        ...

Array dimension order
---------------------
ZDF stores data with the first dimension (x) varying fastest. Data is returned as (and
expected to be) C ordered numpy arrays, so the numpy shape is the reverse of the ZDF
dimensions: a grid with nx = [nx, ny] has shape (ny, nx), and the last numpy axis is x.
Axis information (ZDF_Grid_Info.axis) is listed in ZDF order, i.e. axis[0] is x.
"""

import os
import sys
import copy
import struct
import numpy as np

__all__ = [
    'ZDF_Record', 'ZDF_Iteration', 'ZDF_Grid_Axis', 'ZDF_Grid_Info', 'ZDF_Part_Info',
    'ZDF_Tracks_Info', 'ZDF_Dataset', 'ZDF_Info', 'ZDFfile',
    'info', 'read', 'save_grid', 'save_particles',
    # 'list' is deliberately not exported, to avoid shadowing the builtin; use zdf.list()
]

# -----------------------------------------------------------------------------
# Format constants
# -----------------------------------------------------------------------------

#: ZDF magic number
_ZDF_MAGIC = b'ZDF1'

#: Maximum number of dimensions
ZDF_MAX_DIMS = 3

#: ZDF data type ids (same as enum zdf_data_type in zdf.h) and numpy equivalents
_ZDF_DTYPES = {
     1 : np.dtype('<i1'),   # zdf_int8
     2 : np.dtype('<u1'),   # zdf_uint8
     3 : np.dtype('<i2'),   # zdf_int16
     4 : np.dtype('<u2'),   # zdf_uint16
     5 : np.dtype('<i4'),   # zdf_int32
     6 : np.dtype('<u4'),   # zdf_uint32
     7 : np.dtype('<i8'),   # zdf_int64
     8 : np.dtype('<u8'),   # zdf_uint64
     9 : np.dtype('<f4'),   # zdf_float32
    10 : np.dtype('<f8'),   # zdf_float64
    11 : np.dtype('<c8'),   # zdf_complex64 (same as C++ std::complex<float>)
    12 : np.dtype('<c16'),  # zdf_complex128 (same as C++ std::complex<double>)
}

# Record ids (including version)
_ZDF_INT32_ID       = 0x00010000
_ZDF_DOUBLE_ID      = 0x00020000
_ZDF_STRING_ID      = 0x00030000
_ZDF_DATASET_ID     = 0x00100002
_ZDF_CDSET_START_ID = 0x00110000
_ZDF_CDSET_CHUNK_ID = 0x00120000
_ZDF_CDSET_END_ID   = 0x00130000
_ZDF_ITERATION_ID   = 0x00200001
_ZDF_GRID_INFO_ID   = 0x00210001
_ZDF_PART_INFO_ID   = 0x00220002
_ZDF_TRACK_INFO_ID  = 0x00230001

_ZDF_RECORD_TYPES = {
    0x00010000: "int",
    0x00020000: "double",
    0x00030000: "string",
    0x00100000: "dataset",
    0x00110000: "cdset_start",
    0x00120000: "cdset_chunk",
    0x00130000: "cdset_end",
    0x00200000: "iteration",
    0x00210000: "grid_info",
    0x00220000: "part_info",
    0x00230000: "track_info",
}

def _rndup( n ):
    """Rounds up to a multiple of 4 bytes (the ZDF file unit)"""
    return ( ( n + 3 ) // 4 ) * 4

def _zdf_type_name( rec_id ):
    """Returns record type name from record id"""
    return _ZDF_RECORD_TYPES.get( rec_id & 0xFFFF0000, "unknown" )

def _zdf_dtype( data_type ):
    """Returns numpy dtype for ZDF data type id, or None if not supported"""
    return _ZDF_DTYPES.get( int( data_type ) )

def _zdf_type_id( dtype ):
    """Returns ZDF data type id for numpy dtype, or None if not supported"""
    dtype = np.dtype( dtype )
    for type_id, zdt in _ZDF_DTYPES.items():
        if ( dtype.kind == zdt.kind and dtype.itemsize == zdt.itemsize ):
            return type_id
    return None

# -----------------------------------------------------------------------------
# Serialization helpers (little endian, padded to 4 byte units)
# -----------------------------------------------------------------------------

def _pack_u32( v ): return struct.pack( '<I', int(v) )
def _pack_i32( v ): return struct.pack( '<i', int(v) )
def _pack_u64( v ): return struct.pack( '<Q', int(v) )
def _pack_f64( v ): return struct.pack( '<d', float(v) )

def _pack_string( s ):
    """Serializes string: uint32 length followed by data padded to 4 bytes"""
    data = ( s if s is not None else '' ).encode( 'utf-8' )
    return _pack_u32( len(data) ) + data + b'\0' * ( _rndup( len(data) ) - len(data) )

def _zdf_array( data, name = 'data' ):
    """Converts data to a C contiguous little endian array of a supported type

    Returns
    -------
    (array, type_id)
    """
    a = np.asarray( data )
    type_id = _zdf_type_id( a.dtype )
    if ( type_id is None ):
        raise TypeError( "ZDF: unsupported data type '{}' for {}".format( a.dtype, name ) )
    a = np.ascontiguousarray( a, dtype = _ZDF_DTYPES[type_id] )
    return a, type_id

# -----------------------------------------------------------------------------
# Metadata classes
# -----------------------------------------------------------------------------

class ZDF_Record:
    """ZDF_Record()

    ZDF Datafile record information

    Attributes
    ----------
    pos : int
        Position in ZDF file (record header)
    data_pos : int
        Position in ZDF file of the record data (after the header)
    id : int
        Type id
    name : str
        Record name
    len : int
        Additional record length in bytes
    """
    def __init__( self ):
        self.pos = -1
        self.data_pos = -1
        self.id = -1
        self.name = ''
        self.len = -1

    def version(self):
        """version()

        Gets record version number

        Returns
        -------
        version : int
            Record version number
        """
        return( self.id & 0x0000FFFF )

    def type( self ):
        """type()

        Gets record type from tag

        Returns
        -------
        type : {'int', 'double', 'string', 'dataset', 'cdset_start', 'cdset_chunk', 'cdset_end',
                'iteration', 'grid_info', 'part_info', 'track_info', 'unknown'}
            Type of record
        """
        return _zdf_type_name( self.id )

    def __str__( self ):
        return "<ZDF_Record> {} '{}' at {:#x}, {} bytes".format(
            self.type(), self.name, self.pos, self.len )


class ZDF_Iteration:
    """ZDF_Iteration( name = 'ITERATION', n = 0, t = 0., tunits = '' )

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
    def __init__( self, name = 'ITERATION', n = 0, t = 0., tunits = '' ):
        self.name = name
        self.n = n
        self.t = t
        self.tunits = tunits

    def __str__( self ):
        msg = "<ZDF_Iteration>\n"
        msg += "name    : {}\n".format(self.name)
        msg += "n       : {}\n".format(self.n)
        msg += "t       : {}\n".format(self.t)
        msg += "tunits  : {}\n".format(self.tunits)
        return msg


class ZDF_Grid_Axis:
    """ZDF_Grid_Axis( name = '', type = 0, min = 0., max = 0., label = '', units = '' )

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
    def __init__( self, name = '', type = 0, min = 0., max = 0., label = '', units = '' ):
        self.name = name
        self.type = type
        self.min = min
        self.max = max
        self.label = label
        self.units = units

    def __str__( self ):
        msg = "<ZDF_Grid_Axis>\n"
        msg += "name     : {}\n".format(self.name)
        msg += "type     : {}\n".format(self.type)
        msg += "label    : {}\n".format(self.label)
        msg += "min      : {}\n".format(self.min)
        msg += "max      : {}\n".format(self.max)
        msg += "units    : {}\n".format(self.units)
        return msg


class ZDF_Grid_Info:
    """ZDF_Grid_Info( name = '', ndims = 0, nx = None, label = '', units = '', axis = None )

    Grid dataset information

    Attributes
    ----------
    name : str
        Dataset name
    ndims : int
        Dimensionality of dataset
    nx : list of int (ndims)
        Number of grid points in each direction (ZDF order, x first)
    label : str
        Dataset label
    units : str
        Dataset units
    has_axis : bool
        True if the dataset includes axis information
    axis : list of ZDF_Grid_Axis (ndims)
        Information on each axis (ZDF order, x first), when available
    """
    def __init__( self, name = '', ndims = 0, nx = None, label = '', units = '', axis = None ):
        self.name = name
        self.ndims = ndims
        self.nx = nx if nx is not None else []
        self.label = label
        self.units = units
        self.axis = axis if axis is not None else []
        self.has_axis = len( self.axis ) > 0

    def __str__( self ):
        msg = "<ZDF_Grid_Info>\n"
        msg += "name     : {}\n".format(self.name)
        msg += "label    : {}\n".format(self.label)
        msg += "ndims    : {}\n".format(self.ndims)
        msg += "nx       : {}\n".format(self.nx)
        msg += "units    : {}\n".format(self.units)
        msg += "has_axis : {}\n".format(bool(self.has_axis))
        if ( self.has_axis ):
            for axis in self.axis:
                msg += str(axis)
        return msg

class ZDF_Part_Info:
    """ZDF_Part_Info( name = '', label = '', quants = None, qlabels = None, qunits = None,
                      nparts = 0 )

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
    def __init__( self, name = '', label = '', quants = None, qlabels = None, qunits = None,
                  nparts = 0 ):
        self.name = name
        self.label = label
        self.quants = quants if quants is not None else []
        self.nquants = len( self.quants )
        self.qlabels = qlabels if qlabels is not None else dict()
        self.qunits = qunits if qunits is not None else dict()
        self.nparts = nparts

    def __str__( self ):
        msg = "<ZDF_Part_Info>\n"
        msg += "name    : {}\n".format(self.name)
        msg += "label   : {}\n".format(self.label)
        msg += "nparts  : {}\n".format(self.nparts)
        msg += "nquants : {}\n".format(self.nquants)
        msg += "quants  : {}\n".format(self.quants)
        msg += "qlabels : {}\n".format(self.qlabels)
        msg += "qunits  : {}\n".format(self.qunits)
        return msg

class ZDF_Tracks_Info:
    """ZDF_Tracks_Info()

    Tracks dataset information

    Attributes
    ----------
    name : str
        Track dataset name
    label : str
        Track dataset label
    ntracks : int
        Number of tracks in dataset
    ndump : int
        Number of data dumps
    niter : int
        Number of iterations between points
    nquants : int
        Number of quantities per particle
    quants : list of str (nquants)
        Name of individual quantities
    qlabels : list of str (nquants)
        Labels for each quantity
    qunits : list of str (nquants)
        Units for each quantity
    """
    def __init__( self ):
        self.name = ''
        self.label = ''
        self.ntracks = 0
        self.ndump = 0
        self.niter = 0
        self.nquants = 0
        self.quants = []
        self.qlabels = []
        self.qunits = []

    def __str__( self ):
        msg = "<ZDF_Tracks_Info>\n"
        msg += "name    : {}\n".format(self.name)
        msg += "label   : {}\n".format(self.label)
        msg += "ntracks : {}\n".format(self.ntracks)
        msg += "ndump   : {}\n".format(self.ndump)
        msg += "niter   : {}\n".format(self.niter)
        msg += "nquants : {}\n".format(self.nquants)
        msg += "quants  : {}\n".format(self.quants)
        msg += "qlabels : {}\n".format(self.qlabels)
        msg += "qunits  : {}\n".format(self.qunits)
        return msg

class ZDF_Dataset:
    """ZDF_Dataset

    Chunked dataset being written, returned by ZDFfile.start_cdset()

    Attributes
    ----------
    name : str
        Dataset name
    id : int
        Dataset id
    data_type : int
        ZDF data type id
    nx : list of int
        Dataset dimensions (ZDF order, x first)
    offset : int
        File position of the dataset header
    """
    def __init__( self, name, id, data_type, nx, offset ):
        self.name = name
        self.id = id
        self.data_type = data_type
        self.nx = nx
        self.offset = offset

    @property
    def shape( self ):
        """Dataset shape (numpy order)"""
        return tuple( reversed( self.nx ) )

    @property
    def dtype( self ):
        """Dataset numpy data type"""
        return _ZDF_DTYPES[ self.data_type ]

    def __str__( self ):
        return "<ZDF_Dataset> '{}' id = {}, dtype = {}, shape = {}".format(
            self.name, self.id, self.dtype, self.shape )

# -----------------------------------------------------------------------------
# ZDF file
# -----------------------------------------------------------------------------

class ZDFfile:
    """ZDFfile( file_name, mode = 'r' )

    ZDF data file class. Can be used as a context manager.

    Parameters
    ----------
    file_name : str
        File name of ZDF data file, should include path
    mode : {'r', 'w', 'a'}, optional
        'r' (default) opens an existing file for reading, 'w' creates a new file (overwriting
        any existing file), 'a' opens an existing file for appending new data

    Raises
    ------
    ValueError
        If the file is not a ZDF file, or the mode is invalid
    """
    def __init__(self, file_name, mode = 'r'):

        if ( mode not in ('r','w','a') ):
            raise ValueError( "ZDF: invalid file mode '{}', must be one of 'r', 'w' or 'a'".format(mode) )

        self.mode = mode
        self.ndatasets = 0
        self.__file = None

        if ( mode == 'w' ):
            self.__file = open( file_name, "w+b" )
            self.__file.write( _ZDF_MAGIC )
        else:
            self.__file = open( file_name, "rb" if mode == 'r' else "r+b" )

            # Check magic number
            magic = self.__file.read(4)
            if ( magic != _ZDF_MAGIC ):
                self.__file.close()
                raise ValueError( "ZDF: {} is not a proper ZDF file".format( file_name ) )

            if ( mode == 'a' ):
                # Find existing datasets (new datasets get unique ids) and go to end of file
                self.ndatasets = self.__scan_datasets()
                self.__file.seek( 0, 2 )

    def __enter__( self ):
        return self

    def __exit__( self, exc_type, exc_value, traceback ):
        self.close()
        return False

    def close(self):
        """close()

        Closes ZDF file
        """
        if ( self.__file is not None ):
            self.__file.close()
            self.__file = None

    def record_type(self, typeTag):
        """record_type( typeTag )

        Gets record type name from record id
        """
        return _zdf_type_name( typeTag )

# -----------------------------------------------------------------------------
# Elemental datatypes (read)
# -----------------------------------------------------------------------------

    def __read_scalar( self, fmt, size ):
        data = self.__file.read( size )
        if ( len(data) < size ):
            return None
        return struct.unpack( fmt, data )[0]

    def __read_uint32(self):  return self.__read_scalar( '<I', 4 )
    def __read_int32(self):   return self.__read_scalar( '<i', 4 )
    def __read_uint64(self):  return self.__read_scalar( '<Q', 8 )
    def __read_int64(self):   return self.__read_scalar( '<q', 8 )
    def __read_float32(self): return self.__read_scalar( '<f', 4 )
    def __read_float64(self): return self.__read_scalar( '<d', 8 )

    def __read_string(self):

        length = self.__read_uint32()
        if ( length is None ):
            return None

        if (length > 0):
            # Data is always written in blocks of 4 bytes
            data = self.__file.read( _rndup( length ) )
            if ( len(data) < length ):
                return None
            fstring = data[:length].decode( 'utf-8', errors = 'replace' )
        else:
            fstring = ""

        return fstring

    def __read_array( self, dtype, count ):
        """Reads count elements of type dtype, returns 1D array (or None at EOF)"""
        dtype = np.dtype( dtype )
        nbytes = int(count) * dtype.itemsize
        buf = bytearray( nbytes )
        if ( self.__file.readinto( buf ) != nbytes ):
            return None
        return np.frombuffer( buf, dtype = dtype )

    def __read_arr( self, data_type, nx ):
        """Reads ZDF array of type data_type and dimensions nx (ZDF order)"""
        dtype = _zdf_dtype( data_type )
        if ( dtype is None ):
            print( '(*error*) ZDF: Data type {} not supported.'.format(data_type), file=sys.stderr)
            return None

        shape = tuple( int(n) for n in reversed( nx ) )
        data = self.__read_array( dtype, np.prod( shape, dtype = np.int64 ) )
        if ( data is None ):
            print( '(*error*) ZDF: Unexpected end of file reading data.', file=sys.stderr)
            return None
        return data.reshape( shape )

# -----------------------------------------------------------------------------
# Elemental datatypes (write)
# -----------------------------------------------------------------------------

    def __check_write( self ):
        if ( self.__file is None ):
            raise ValueError( "ZDF: file is closed" )
        if ( self.mode == 'r' ):
            raise IOError( "ZDF: file was opened for reading only" )

    def __write_record( self, rec_id, name, body, data = None ):
        """Writes a record

        Parameters
        ----------
        rec_id : int
            Record id
        name : str
            Record name
        body : bytes
            Record data (header part)
        data : numpy.ndarray, optional
            Array data to write after body (C contiguous, little endian). Padding is added
            as required.

        Returns
        -------
        data_pos : int
            File position of record data (after the record header)
        """
        nbytes = len( body )
        if ( data is not None ):
            nbytes += _rndup( data.nbytes )

        self.__file.write( _pack_u32( rec_id ) + _pack_string( name ) + _pack_u64( nbytes ) )
        data_pos = self.__file.tell()
        self.__file.write( body )

        if ( data is not None and data.size > 0 ):
            self.__file.write( data.reshape(-1).view( np.uint8 ) )
            npad = _rndup( data.nbytes ) - data.nbytes
            if ( npad > 0 ):
                self.__file.write( b'\0' * npad )

        return data_pos

# -----------------------------------------------------------------------------
# Low level interfaces
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Read record header
# -----------------------------------------------------------------------------

    def read_record(self, skip = False):
        """read_record(skip = False)

        Reads current record information from file

        Parameters
        ----------
        skip : bool, optional
            If set to True, skip to next record after reading record
            header data

        Returns
        -------
        rec : ZDF_Record
            Record information, or None if the end of file was reached
        """

        rec = ZDF_Record()
        rec.pos = self.__file.tell()

        # Read record id and check for EOF
        rec.id = self.__read_uint32()
        if ( rec.id is None ):
            return None

        # Read name and length
        rec.name = self.__read_string()
        rec.len  = self.__read_uint64()
        if ( rec.name is None or rec.len is None ):
            print( '(*error*) ZDF: Truncated record header found.', file=sys.stderr)
            return None

        rec.data_pos = self.__file.tell()

        if (skip):
            self.__record_skip( rec )

        return rec

    def __record_skip( self, rec ):
        """Positions the file pointer at the end of the record"""
        # Record lengths are always multiples of 4 bytes; rounding up also handles 8 bit
        # datasets written by older versions of the C library (unpadded length)
        self.__file.seek( rec.data_pos + _rndup( rec.len ), 0 )

    def __scan_datasets( self ):
        """Returns the largest dataset id in the file"""
        self.__file.seek( 4 )
        maxid = 0
        while True:
            rec = self.read_record()
            if ( rec is None ):
                break
            # Regular datasets (version >= 1) and chunked datasets start with the dataset id
            if ( ( rec.type() == 'dataset' and rec.version() >= 1 ) or rec.type() == 'cdset_start' ):
                dset_id = self.__read_uint32()
                if ( dset_id is not None and dset_id > maxid ):
                    maxid = dset_id
            self.__record_skip( rec )
        return maxid

    def __find_record( self, name, types ):
        """Finds the first record with the given name and one of the given types.
        Returns the record (file pointer positioned after the header) or None"""
        self.__file.seek( 4 )
        while True:
            rec = self.read_record()
            if ( rec is None ):
                return None
            if ( rec.name == name and rec.type() in types ):
                return rec
            self.__record_skip( rec )

# -----------------------------------------------------------------------------
# Read string
# -----------------------------------------------------------------------------

    def read_string(self, rec = None):
        """read_string(rec = None)

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
        if ( rec is None ):
            rec = self.read_record()
            if ( rec is None ):
                return None

        fstring = self.__read_string()
        return fstring

# -----------------------------------------------------------------------------
# Read iteration
# -----------------------------------------------------------------------------

    def read_iteration(self, rec = None):
        """read_iteration( rec = None )

        Read iteration record from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        iteration : ZDF_Iteration()
            Iteration data. If the record is not an iteration record, the file
            pointer is restored and False is returned.
        """

        if ( rec is None ):
            rec = self.read_record()
            if ( rec is None ):
                return False

        if ( rec.type() == 'iteration' ):
            iteration = ZDF_Iteration()
            iteration.name = rec.name
            iteration.n = self.__read_int32()
            iteration.t = self.__read_float64()
            iteration.tunits = self.__read_string()

        else:
            # rewind file
            self.__file.seek(rec.pos, 0)
            iteration = False

        return iteration

# -----------------------------------------------------------------------------
# Read grid info
# -----------------------------------------------------------------------------

    def read_grid_info(self, rec = None):
        """read_grid_info( rec = None )

        Read grid information record from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        info : ZDF_Grid_Info()
            Grid information data. On error returns False.
        """
        if ( rec is None ):
            rec = self.read_record()
            if ( rec is None ):
                return False

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
        info.nx = self.__read_array( '<u8', info.ndims )

        info.label = self.__read_string()
        info.units = self.__read_string()
        info.has_axis = self.__read_int32()

        if ( info.has_axis ):
            for i in range(info.ndims):
                ax = ZDF_Grid_Axis()
                if (version > 0):
                    ax.name = self.__read_string()
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

    def read_part_info(self, rec = None):
        """read_part_info( rec = None )

        Read particle information record from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        info : ZDF_Part_Info()
            Particle information data. On error returns None.
        """
        if ( rec is None ):
            rec = self.read_record()
            if ( rec is None ):
                return None

        # Maximum supported version
        max_version = 0x00000002

        # Get version
        version = rec.version()
        if ( version > max_version ):
            print( '(*error*) ZDF: Particles info version is higher than supported.' , file=sys.stderr)
            print( '(*error*) ZDF: Please update the code to a newer version.' , file=sys.stderr)
            return None

        info      = ZDF_Part_Info()
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
# Read track info
# -----------------------------------------------------------------------------

    def read_track_info(self, rec = None):
        """read_track_info( rec = None )

        Read track information record from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        info : ZDF_Tracks_Info()
            Track information data. On error returns None.
        """

        if ( rec is None ):
            rec = self.read_record()
            if ( rec is None ):
                return None

        # Maximum supported version
        max_version = 0x00000001

        # Get version
        version = rec.version()
        if ( version > max_version ):
            print( '(*error*) ZDF: Tracks info version is higher than supported.' , file=sys.stderr)
            print( '(*error*) ZDF: Please update the code to a newer version.' , file=sys.stderr)
            return None

        info = ZDF_Tracks_Info()

        info.name = rec.name
        info.label    = self.__read_string()
        info.ntracks = self.__read_uint32()
        info.ndump   = self.__read_uint32()
        info.niter   = self.__read_uint32()
        info.nquants = self.__read_uint32()

        for i in range(info.nquants):
            info.quants.append( self.__read_string() )

        for i in range(info.nquants):
            info.qlabels.append( self.__read_string() )

        for i in range(info.nquants):
            info.qunits.append( self.__read_string() )

        # Iteration data is not supported so remove it from metadata
        info.nquants -= 1

        info.quants.pop(0)
        info.qlabels.pop(0)
        info.qunits.pop(0)

        return info

# -----------------------------------------------------------------------------
# Read dataset
# -----------------------------------------------------------------------------

    def read_dataset(self, rec = None):
        """read_dataset( rec = None )

        Read dataset from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data

        Returns
        -------
        data : numpy.ndarray
            Numpy ndarray with data. On error returns None.
        """

        if ( rec is None ):
            rec = self.read_record()
            if ( rec is None ):
                return None

        if ( rec.type() != 'dataset' ):
            print( '(*error*) ZDF: Expected dataset record but found {} instead.'.format(rec.type()),
                  file=sys.stderr)
            return None

        # Maximum supported version
        max_version = 0x00000002

        # Get version
        version = rec.version()
        if ( version > max_version ):
            print( '(*error*) ZDF: Dataset version is higher than supported.' , file=sys.stderr)
            print( '(*error*) ZDF: Please update the code to a newer version.' , file=sys.stderr)
            return None

        # Version 0x0001 includes id tag
        if ( version >= 1 ):
            self.__read_uint32()

        data_type = self.__read_int32()
        ndims     = self.__read_uint32()
        nx        = self.__read_array( '<u8', ndims )
        data      = self.__read_arr( data_type, nx )

        # Skip padding (if any)
        self.__record_skip( rec )

        return data

# -----------------------------------------------------------------------------
# Read chunked dataset
# -----------------------------------------------------------------------------

    def read_cdset(self, rec = None, pos = 0 ):
        """read_cdset( rec = None, pos = 0 )

        Read chunked dataset from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data
        pos : int, optional, default 0
            If set to 1, position file pointer at the end of the cdset_start record after
            reading data. Otherwise the file pointer is left after the cdset_end record.

        Returns
        -------
        data : numpy.ndarray
            Numpy ndarray with data. On error returns None.
        """

        if ( rec is None ):
            rec = self.read_record()
            if ( rec is None ):
                return None

        if ( rec.type() != 'cdset_start' ):
            print( '(*error*) ZDF: Expected cdset_start record but found {} instead.'.format(rec.type()),
                  file=sys.stderr)
            return None

        # Maximum supported version
        max_version = 0x00000001

        # Get version
        version = rec.version()
        if ( version > max_version ):
            print( '(*error*) ZDF: Chunked dataset version is higher than supported.' , file=sys.stderr)
            print( '(*error*) ZDF: Please update the code to a newer version.' , file=sys.stderr)
            return None

        dset_id   = self.__read_uint32()
        data_type = self.__read_int32()
        ndims     = self.__read_uint32()
        nx        = self.__read_array( '<u8', ndims )

        dtype = _zdf_dtype( data_type )
        if ( dtype is None ):
            print( '(*error*) ZDF: Data type {} not supported.'.format(data_type), file=sys.stderr)
            return None

        # Create numpy array
        data = np.zeros( tuple( int(n) for n in reversed(nx) ), dtype = dtype )

        # Chunk names use the dataset id in hexadecimal (as written by the C library)
        chunk_name = "{:08x}-chunk".format(dset_id)
        end_name = "{:08x}-end".format(dset_id)

        # Loop until the end record
        cdset_start_end = self.__file.tell()

        while True:

            # Read next record
            rec = self.read_record()

            # Check if end of file reached
            if ( rec is None ):
                print( '(*warning*) ZDF: End of file reached before the end of dataset {}'.format(dset_id),
                       file=sys.stderr)
                break

            if ( rec.name == end_name ):
                break

            if ( rec.name == chunk_name and rec.type() == 'cdset_chunk' ):
                self.__read_uint32()    # chunk dataset id

                count  = self.__read_array( '<i8', ndims )
                start  = self.__read_array( '<i8', ndims )
                stride = self.__read_array( '<i8', ndims )

                chunk  = self.__read_arr( data_type, count )
                if ( chunk is None ):
                    return None

                # numpy axis k corresponds to ZDF dimension ndims - 1 - k
                idx = tuple( slice( int(start[d]), int(start[d] + count[d] * stride[d]), int(stride[d]) )
                             for d in reversed( range( ndims ) ) )
                data[ idx ] = chunk

            # Move to the end of the record (skipping padding, if any)
            self.__record_skip(rec)

        # If requested, position file pointer at the end of the cdset_start record
        if ( pos == 1 ):
            self.__file.seek( cdset_start_end )

        return data

# -----------------------------------------------------------------------------
# Read arbitrary ZDF element
# -----------------------------------------------------------------------------

    def read_element( self, rec = None, name = None, type_id = None):
        """read_element( rec = None, name = None, type_id = None )

        Reads abitrary zdf element from file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data.
        name : str, optional
            If set the routine will only read the record if it matches the supplied name.
        type_id : str, optional
            If set the routine will only read the record if it matches the supplied type.

        Returns
        -------
        multiple data types: zdf element data
        """
        if ( rec is None ):
            rec = self.read_record()
            if ( rec is None ):
                return None

        if ( name is not None ):
            if (name != rec.name ):
                print("(*warning*) Requested name does not match record name", file=sys.stderr)
                print("(*warning*) expected '{}', found '{}'".format(name, rec.name), file=sys.stderr)
                self.__record_skip(rec)
                return False

        if ( type_id is not None ):
            if (type_id != rec.type() ):
                print("(*warning*) Requested type does not match record type", file=sys.stderr)
                print("(*warning*) expected '{}', found '{}'".format(type_id, rec.type()), file=sys.stderr)
                self.__record_skip(rec)
                return False
        else:
            type_id = rec.type()

        if ( type_id == "int" ):
            data = self.__read_int32()
        elif( type_id == "double" ):
            data = self.__read_float64()
        elif( type_id == "string" ):
            data = self.__read_string()
        elif( type_id == "dataset" ):
            data = self.read_dataset( rec = rec )
        elif( type_id == "cdset_start" ):
            data = self.read_cdset( rec = rec )
        elif( type_id == "cdset_chunk" ):
            print("(*warning*) Dataset chunks are not meant to be read directly", file=sys.stderr)
            self.__record_skip(rec)
            data = None
        elif( type_id == "cdset_end" ):
            print("(*warning*) Dataset end marks have no data", file=sys.stderr)
            self.__record_skip(rec)
            data = None
        elif( type_id == "iteration" ):
            data = self.read_iteration( rec = rec )
        elif( type_id == "grid_info" ):
            data = self.read_grid_info( rec = rec )
        elif( type_id == "part_info" ):
            data = self.read_part_info( rec = rec )
        elif( type_id == "track_info" ):
            data = self.read_track_info( rec = rec )
        else:
            print("(*warning*) Unknown element type, skipping", file=sys.stderr)
            self.__record_skip(rec)
            data = None

        return data

# -----------------------------------------------------------------------------
# Read named dataset
# -----------------------------------------------------------------------------

    def read_named_dataset( self, name ):
        """read_named_dataset( name )

        Reads a dataset (regular or chunked) by name, anywhere in the file

        Parameters
        ----------
        name : str
            Dataset name

        Returns
        -------
        data : numpy.ndarray
            Dataset data, or None if not found
        """
        rec = self.__find_record( name, ('dataset', 'cdset_start') )
        if ( rec is None ):
            print( "(*error*) ZDF: Dataset '{}' not found".format( name ), file=sys.stderr )
            return None
        if ( rec.type() == 'dataset' ):
            return self.read_dataset( rec = rec )
        return self.read_cdset( rec = rec )

# -----------------------------------------------------------------------------
# Read particle data
# -----------------------------------------------------------------------------

    def read_part_data( self, quants ):
        """read_part_data( quants )

        Read particle data. Quantities are located by name, so this works for any
        ordering of the quantity datasets / chunks in the file.

        Parameters
        ----------
        quants : list of str
            Particle quantities in the zdf file

        Returns
        -------
        data : dict
            Dictionary with particle data indexed with quantity names
        """
        data = dict()
        for q in quants:
            data[q] = self.read_named_dataset( q )
        return data

# -----------------------------------------------------------------------------
# Read track data
# -----------------------------------------------------------------------------

    def read_track_data( self, trackInfo ):
        """read_track_data( trackInfo )

        Reads track data from zdf file

        Parameters
        ----------
        trackInfo : ZDF_Tracks_Info
            Tracks dataset information

        Returns
        -------
        trackData : list of numpy.ndarray
            Track Data
        """

        itermap = self.read_named_dataset( 'itermap' )
        data = self.read_named_dataset( 'data' )
        if ( itermap is None or data is None ):
            return None

        trackNp = np.zeros( trackInfo.ntracks, dtype = '<i8' )
        for i in range( itermap.shape[0] ):
            trackID = itermap[i,0]-1
            npoints = itermap[i,1]
            trackNp[trackID] += npoints

        trackData = [None] * trackInfo.ntracks
        for i in range( trackInfo.ntracks ):
            trackData[i] = np.zeros( [ trackNp[i], trackInfo.nquants ], dtype = '<f4' )
            trackNp[i] = 0

        idx = 0
        for i in range( itermap.shape[0] ):
            trackID = itermap[i,0]-1
            npoints = itermap[i,1]
            (trackData[ trackID ])[ trackNp[trackID] : trackNp[trackID]+npoints, : ] = data[ idx : idx+npoints, : ]
            trackNp[trackID] += npoints
            idx += npoints

        return trackData

# -----------------------------------------------------------------------------
# Retrieve list of file contents / print file contents
# -----------------------------------------------------------------------------

    def list(self, printRec = False):
        """list( printRec=False )

        Gets a list of file contents and optionally prints it to screen

        Parameters
        ----------
        printRec : bool, optional
            If set to True will print all records found in the file,
            defaults to False.

        Returns
        -------
        rec_list : list of ZDF_Record
            Records in file
        """
        # Position file after magic number
        self.__file.seek(4)

        rec_list = []
        while True:
            rec = self.read_record(skip=True)
            if (rec is None):
                break
            rec_list.append(rec)

        if (printRec and (len(rec_list) > 0)):
            print('Position     Size(bytes)  Type         Name')
            print('-----------------------------------------------------')
            for rec in rec_list:
                print('{:#010x}   {:#010x}   {:11}  {}'.format(
                    rec.pos, rec.len, rec.type(), rec.name) )

        return rec_list

# -----------------------------------------------------------------------------
# Write metadata
# -----------------------------------------------------------------------------

    def add_string( self, name, value ):
        """add_string( name, value )

        Adds string element to file
        """
        self.__check_write()
        self.__write_record( _ZDF_STRING_ID, name, _pack_string( value ) )

    def add_int32( self, name, value ):
        """add_int32( name, value )

        Adds int32 element to file
        """
        self.__check_write()
        self.__write_record( _ZDF_INT32_ID, name, _pack_i32( value ) )

    def add_double( self, name, value ):
        """add_double( name, value )

        Adds float64 element to file
        """
        self.__check_write()
        self.__write_record( _ZDF_DOUBLE_ID, name, _pack_f64( value ) )

    def add_iteration( self, iteration ):
        """add_iteration( iteration )

        Adds iteration information to file

        Parameters
        ----------
        iteration : ZDF_Iteration
            Iteration information
        """
        self.__check_write()
        body = _pack_i32( iteration.n ) + _pack_f64( iteration.t ) + _pack_string( iteration.tunits )
        self.__write_record( _ZDF_ITERATION_ID, iteration.name if iteration.name else 'ITERATION', body )

    def add_grid_info( self, info ):
        """add_grid_info( info )

        Adds grid information to file

        Parameters
        ----------
        info : ZDF_Grid_Info
            Grid information. ndims and nx must be set (ZDF order, x first). Axis
            information is written if info.axis has ndims entries.
        """
        self.__check_write()

        ndims = int( info.ndims )
        if ( ndims < 1 or ndims > ZDF_MAX_DIMS ):
            raise ValueError( "ZDF: invalid number of grid dimensions ({})".format( ndims ) )
        if ( len( info.nx ) < ndims ):
            raise ValueError( "ZDF: grid info nx must have ndims ({}) values".format( ndims ) )

        axis = info.axis if info.axis is not None else []
        if ( len( axis ) not in ( 0, ndims ) ):
            raise ValueError( "ZDF: grid info must have 0 or ndims ({}) axis".format( ndims ) )

        body = _pack_u32( ndims )
        for i in range( ndims ):
            body += _pack_u64( info.nx[i] )
        body += _pack_string( info.label ) + _pack_string( info.units )
        body += _pack_i32( 1 if len(axis) > 0 else 0 )
        for ax in axis:
            body += _pack_string( ax.name ) + _pack_i32( ax.type )
            body += _pack_f64( ax.min ) + _pack_f64( ax.max )
            body += _pack_string( ax.label ) + _pack_string( ax.units )

        self.__write_record( _ZDF_GRID_INFO_ID, info.name, body )

    def add_part_info( self, info ):
        """add_part_info( info )

        Adds particle information to file

        Parameters
        ----------
        info : ZDF_Part_Info
            Particle information. Missing labels default to the quantity name, missing
            units default to ''.
        """
        self.__check_write()

        quants = info.quants
        body = _pack_string( info.label ) + _pack_u64( info.nparts ) + _pack_u32( len(quants) )
        for q in quants:
            body += _pack_string( q )
        for q in quants:
            body += _pack_string( info.qlabels.get( q, q ) )
        for q in quants:
            body += _pack_string( info.qunits.get( q, '' ) )

        self.__write_record( _ZDF_PART_INFO_ID, info.name, body )

# -----------------------------------------------------------------------------
# Write datasets
# -----------------------------------------------------------------------------

    def add_dataset( self, name, data ):
        """add_dataset( name, data )

        Adds dataset to file

        Parameters
        ----------
        name : str
            Dataset name
        data : numpy.ndarray
            Data (1 to 3 dimensions, numpy order)

        Returns
        -------
        id : int
            Dataset id
        """
        self.__check_write()

        a, type_id = _zdf_array( data, name )
        if ( a.ndim < 1 or a.ndim > ZDF_MAX_DIMS ):
            raise ValueError( "ZDF: datasets must have 1 to {} dimensions".format( ZDF_MAX_DIMS ) )

        self.ndatasets += 1
        body = _pack_u32( self.ndatasets ) + _pack_i32( type_id ) + _pack_u32( a.ndim )
        for n in reversed( a.shape ):
            body += _pack_u64( n )

        self.__write_record( _ZDF_DATASET_ID, name, body, data = a )
        return self.ndatasets

    def start_cdset( self, name, dtype, shape ):
        """start_cdset( name, dtype, shape )

        Starts a chunked dataset

        Parameters
        ----------
        name : str
            Dataset name
        dtype : numpy dtype
            Data type
        shape : tuple of int
            Global dataset shape (numpy order)

        Returns
        -------
        dset : ZDF_Dataset
            Dataset object, to be used with write_cdset(), extend_cdset() and end_cdset()
        """
        self.__check_write()

        type_id = _zdf_type_id( dtype )
        if ( type_id is None ):
            raise TypeError( "ZDF: unsupported data type '{}' for {}".format( np.dtype(dtype), name ) )
        shape = tuple( int(n) for n in shape )
        if ( len(shape) < 1 or len(shape) > ZDF_MAX_DIMS ):
            raise ValueError( "ZDF: datasets must have 1 to {} dimensions".format( ZDF_MAX_DIMS ) )

        self.ndatasets += 1
        nx = [ n for n in reversed( shape ) ]
        offset = self.__write_record( _ZDF_CDSET_START_ID, name, self.__cdset_header( self.ndatasets, type_id, nx ) )

        return ZDF_Dataset( name, self.ndatasets, type_id, nx, offset )

    def __cdset_header( self, dset_id, type_id, nx ):
        body = _pack_u32( dset_id ) + _pack_i32( type_id ) + _pack_u32( len(nx) )
        for n in nx:
            body += _pack_u64( n )
        return body

    def write_cdset( self, dset, data, start = None ):
        """write_cdset( dset, data, start = None )

        Writes a chunk of a chunked dataset

        Parameters
        ----------
        dset : ZDF_Dataset
            Dataset object returned by start_cdset()
        data : numpy.ndarray
            Chunk data, with the same number of dimensions as the dataset (numpy order).
            Data is converted to the dataset type if required.
        start : tuple of int, optional
            Position of the chunk in the global dataset (numpy order), defaults to 0
        """
        self.__check_write()

        a = np.ascontiguousarray( data, dtype = dset.dtype )
        ndims = len( dset.nx )
        if ( a.ndim != ndims ):
            raise ValueError( "ZDF: chunk must have the same number of dimensions as the dataset" )

        if ( start is None ):
            start = (0,) * ndims
        if ( len(start) != ndims ):
            raise ValueError( "ZDF: chunk start must have {} values".format( ndims ) )

        for k in range( ndims ):
            if ( start[k] < 0 or start[k] + a.shape[k] > dset.shape[k] ):
                raise ValueError( "ZDF: chunk is outside dataset bounds" )

        body = _pack_u32( dset.id )
        for n in reversed( a.shape ):
            body += _pack_u64( n )
        for s in reversed( start ):
            body += _pack_u64( s )
        for i in range( ndims ):
            body += _pack_u64( 1 )

        self.__write_record( _ZDF_CDSET_CHUNK_ID, "{:08x}-chunk".format( dset.id ), body, data = a )

    def extend_cdset( self, dset, shape ):
        """extend_cdset( dset, shape )

        Extends chunked dataset dimensions

        Parameters
        ----------
        dset : ZDF_Dataset
            Dataset object returned by start_cdset()
        shape : tuple of int
            New dataset shape (numpy order), must be >= than the current shape
        """
        self.__check_write()

        shape = tuple( int(n) for n in shape )
        if ( len(shape) != len(dset.nx) or any( n < o for n, o in zip( shape, dset.shape ) ) ):
            raise ValueError( "ZDF: new dataset shape must have the same number of dimensions "
                              "and be >= than the current shape" )

        dset.nx = [ n for n in reversed( shape ) ]
        end = self.__file.seek( 0, 2 )
        self.__file.seek( dset.offset )
        self.__file.write( self.__cdset_header( dset.id, dset.data_type, dset.nx ) )
        self.__file.seek( end )

    def end_cdset( self, dset ):
        """end_cdset( dset )

        Ends a chunked dataset

        Parameters
        ----------
        dset : ZDF_Dataset
            Dataset object returned by start_cdset()
        """
        self.__check_write()
        self.__write_record( _ZDF_CDSET_END_ID, "{:08x}-end".format( dset.id ), b'' )

# -----------------------------------------------------------------------------
# High level interfaces
# -----------------------------------------------------------------------------

class ZDF_Info:
    """ZDF_Info()
    ZDF File information

    Attributes
    ----------
    type : {'grid','particles','tracks-2'}
        Type of ZDF file
    grid : ZDF_Grid_Info
        Grid information for grid files
    particles : ZDF_Part_Info
        Particle information for particle files
    tracks : ZDF_Tracks_Info
        Tracks information for tracks files
    iteration : ZDF_Iteration
        Iteration information

    """
    def __init__(self):
        self.type = ""
        self.grid = None
        self.particles = None
        self.tracks = None
        self.iteration = None

    def __str__( self ):
        msg = "<ZDF_Info>\n"
        msg += "type    : {}\n".format(self.type)
        for item in ( self.grid, self.particles, self.tracks, self.iteration ):
            if ( item ):
                msg += str( item )
        return msg

def _read_metadata( zdf ):
    """Reads file type and metadata, returns ZDF_Info or None"""
    info = ZDF_Info()
    info.type = zdf.read_string()
    if ( info.type == "grid" ):
        info.grid = zdf.read_grid_info()
        info.iteration = zdf.read_iteration()
    elif ( info.type == "particles" ):
        info.particles = zdf.read_part_info()
        info.iteration = zdf.read_iteration()
    elif ( info.type == "tracks-2" ):
        info.tracks = zdf.read_track_info()
    else:
        print("File is not a valid ZDF grid, particles or tracks file", file=sys.stderr)
        return None
    return info

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
        File information. If file is invalid None is returned.
    """
    with ZDFfile( file_name ) as zdf:
        return _read_metadata( zdf )

def read( file_name ):
    """read( file_name )

    Reads all data in a ZDF file

    Parameters
    ----------
    file_name : str
        File name of ZDF data file, should include path

    Returns
    -------
    (data, info) : ( numpy.ndarray | dictionary | list, ZDF_Info )
        Tuple containing file data and metadata. Data will be:
        + a numpy.ndarray for grid data
        + a dictionary of numpy.array for particle data (one entry per quantity).
        + a list of numpy.array for track data (one entry per track)
        Metadata is returned as a ZDF_Info object.
        If file is invalid None is returned.
    """
    with ZDFfile( file_name ) as zdf:
        file_info = _read_metadata( zdf )
        if ( file_info is None ):
            return None

        if ( file_info.type == "grid" ):
            data = zdf.read_element()
        elif ( file_info.type == "particles" ):
            data = zdf.read_part_data( file_info.particles.quants )
        else:
            data = zdf.read_track_data( file_info.tracks )

    return (data, file_info)

def list(file_name):
    """list( file_name )

    Prints a list of all zdf file contents

    Parameters
    ----------
    file_name : str
        File name of ZDF data file, should include path
    """
    with ZDFfile( file_name ) as zdf:
        zdf.list(True)

def _create_path( file_name ):
    """Creates the directory for file_name if required"""
    path = os.path.dirname( file_name )
    if ( path ):
        os.makedirs( path, exist_ok = True )

def save_grid( file_name, data, info = None, iteration = None ):
    """save_grid( file_name, data, info = None, iteration = None )

    Saves grid data to a ZDF file. The file can be read with zdf.read() and by
    the C / C++ / Fortran ZDF libraries.

    Parameters
    ----------
    file_name : str
        File name, should include path. The directory is created if required.
    data : numpy.ndarray
        Grid data (1 to 3 dimensions, numpy order: the last axis is x)
    info : ZDF_Grid_Info, optional
        Grid metadata. ndims and nx are set from the data (the object is not
        modified). Axis information, if present, must be in ZDF order (x first).
        If not set, the grid name is taken from the file name.
    iteration : ZDF_Iteration, optional
        Iteration metadata
    """
    a, type_id = _zdf_array( data )
    if ( a.ndim < 1 or a.ndim > ZDF_MAX_DIMS ):
        raise ValueError( "ZDF: grid data must have 1 to {} dimensions".format( ZDF_MAX_DIMS ) )

    if ( info is None ):
        info = ZDF_Grid_Info( name = os.path.splitext( os.path.basename( file_name ) )[0] )
    else:
        info = copy.copy( info )

    info.ndims = a.ndim
    info.nx = [ n for n in reversed( a.shape ) ]

    _create_path( file_name )
    with ZDFfile( file_name, 'w' ) as zdf:
        zdf.add_string( "TYPE", "grid" )
        zdf.add_grid_info( info )
        if ( iteration is not None ):
            zdf.add_iteration( iteration )
        zdf.add_dataset( info.name, a )

def save_particles( file_name, data, info = None, iteration = None ):
    """save_particles( file_name, data, info = None, iteration = None )

    Saves particle data to a ZDF file. The file can be read with zdf.read() and by
    the C / C++ / Fortran ZDF libraries.

    Parameters
    ----------
    file_name : str
        File name, should include path. The directory is created if required.
    data : dict
        Particle data, one 1D array per quantity, all with the same length
    info : ZDF_Part_Info, optional
        Particle metadata. If set, info.quants selects the quantities (and order)
        to save; nparts is set from the data (the object is not modified). If not
        set, all quantities in data are saved and the species name is taken from
        the file name.
    iteration : ZDF_Iteration, optional
        Iteration metadata
    """
    if ( info is None ):
        info = ZDF_Part_Info( name = os.path.splitext( os.path.basename( file_name ) )[0],
                              quants = [ q for q in data ] )
    else:
        info = copy.copy( info )

    arrays = dict()
    for q in info.quants:
        if ( q not in data ):
            raise KeyError( "ZDF: quantity '{}' not found in particle data".format( q ) )
        a, type_id = _zdf_array( data[q], q )
        if ( a.ndim != 1 ):
            raise ValueError( "ZDF: particle quantity '{}' must be a 1D array".format( q ) )
        arrays[q] = a

    sizes = set( a.size for a in arrays.values() )
    if ( len( sizes ) > 1 ):
        raise ValueError( "ZDF: all particle quantities must have the same length" )

    info.nquants = len( info.quants )
    info.nparts = sizes.pop() if sizes else 0

    _create_path( file_name )
    with ZDFfile( file_name, 'w' ) as zdf:
        zdf.add_string( "TYPE", "particles" )
        zdf.add_part_info( info )
        if ( iteration is not None ):
            zdf.add_iteration( iteration )
        for q in info.quants:
            zdf.add_dataset( q, arrays[q] )
