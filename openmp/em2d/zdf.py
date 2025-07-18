#!/usr/bin/env python

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
        self.pos = -1
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
        self.t = 0.
        self.tunits = ""

    def __str__( self ):
        msg = "<ZDF_Iteration>\n"
        msg += "name    : {}\n".format(self.name)
        msg += "n       : {}\n".format(self.n)
        msg += "t       : {}\n".format(self.t)
        msg += "tunits  : {}\n".format(self.tunits)
        return msg


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

    def __str__( self ):
        msg = "<ZDF_Grid_Axis>\n"
        msg += "name     : {}\n".format(self.name)
        msg += "label    : {}\n".format(self.label)
        msg += "min      : {}\n".format(self.min)
        msg += "max      : {}\n".format(self.max)
        msg += "units    : {}\n".format(self.units)
        return msg


class ZDF_Grid_Info:
    """ZDF_Grid_Info()

    Grid dataset information

    Attributes
    ----------
    name : str
        Dataset name
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

    def __str__( self ):
        msg = "<ZDF_Grid_info>\n"
        msg += "name     : {}\n".format(self.name)
        msg += "label    : {}\n".format(self.label)
        msg += "ndims    : {}\n".format(self.ndims)
        msg += "nx       : {}\n".format(self.nx)
        msg += "units    : {}\n".format(self.units)
        if ( self.has_axis ):
            for axis in self.axis:
                msg += axis
        else:
            msg += "has_axis : {}\n".format(self.has_axis)
        return msg

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

    def __str__( self ):
        msg = "<ZDF_Part_info>\n"
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
    qlabels : dictionary
        Labels for each quantity
    qunits : dictionary
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
        msg += "qlabels  : {}\n".format(self.qlabels)
        msg += "qunits  : {}\n".format(self.qunits)
        return msg

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

    def __read_complex64_arr(self, nx):
        size = np.prod(nx)
        data = np.fromfile(self.__file,dtype='<c8',count=size)
        data.shape = np.flip(nx)
        return data

    def __read_complex128_arr(self, nx):
        size = np.prod(nx)
        data = np.fromfile(self.__file,dtype='<c16',count=size)
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
        elif ( dtype == 11 ):
            data = self.__read_complex64_arr(nx)           
        elif ( dtype == 12 ):
            data = self.__read_complex128_arr(nx)           
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

    def read_record(self, skip = False):
        """read_record(skip = False)

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

        if (skip):
            self.__file.seek(rec.len, 1)

        return rec
    
    def __record_skip( self, rec ):
        self.__file.seek(rec.len, 1)

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
            Iteration data
        """

        if ( rec is None ):
            rec = self.read_record()
        
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
            Grid information data
        """
        if ( rec is None ):
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

        if ( rec is None ):
            rec = self.read_record()
            
        if ( self.record_type(rec.id) != 'dataset' ):
            print( '(*error*) ZDF: Expected dataset record but found {} instead.'.format(self.record_type(rec.id)),
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
            id = self.__read_uint32()
        else:
            id = 0
        
        data_type = self.__read_int32()
        ndims     = self.__read_uint32()
        nx        = self.__read_uint64_arr(ndims)
        data      = self.__read_arr( data_type, nx )

        return data

# -----------------------------------------------------------------------------
# Read chunked dataset
# -----------------------------------------------------------------------------
    
    def read_cdset(self, rec = None, pos = 0 ):
        """read_cdset()

        Read chunked dataset from data file

        Parameters
        ----------
        rec : ZDF_Record, optional
            If not set the routine will read the record before reading the data
        pos : int, optional, default 0
            If set to 1, position file pointer at the end of the cdset_start record after reading data

        Returns
        -------
        data : numpy.ndarray
            Numpy ndarray with data. On error returns None.
        """

        if ( rec is None ):
            rec = self.read_record()

        if ( self.record_type(rec.id) != 'cdset_start' ):
            print( '(*error*) ZDF: Expected cdset_start record but found {} instead.'.format(self.record_type(rec.id)),
                  file=sys.stderr)
            return None

        # Maximum supported version
        max_version = 0x00000001
        
        # Get version
        version = rec.version()
        if ( version > max_version ):
            print( '(*error*) ZDF: Chunked dataset version is higher than supported.' , file=sys.stderr)
            print( '(*error*) ZDF: Please update the code to a newer version.' , file=sys.stderr)
            return -1
        
        id        = self.__read_uint32()
        data_type = self.__read_int32()
        ndims     = self.__read_uint32()
        nx        = self.__read_uint64_arr(ndims)

        size = np.prod(nx)
  
        # Create numpy array
        dt = {
            5 :'int32',
            3 :'uint32',
            7 :'int64',
            8 :'uint64',
            9 :'float32',
           10 :'float64',
           11 :'complex64',     # same as C++ std::complex<float>
           12 :'complex128',    # same as C++ std::complex<double>
        }       
        
        data = np.zeros( np.flip(nx), dtype = dt[data_type] )
        
        chunk_name = "{:#08}-chunk".format(id)
        end_name = "{:#08}-end".format(id)
        
        # Loop until the end record
        cdset_start_end = self.__file.tell()
        
        name = ""
        while ( name != end_name ):
            
            # Read next record
            rec = self.read_record()
            
            # Check if end of file reached
            if ( rec is None ):
                break
            
            name = rec.name
                        
            if ( name == chunk_name ):
                chunk_id = self.__read_uint32()
                
                count  = self.__read_int64_arr(ndims)
                start  = self.__read_int64_arr(ndims)
                stride = self.__read_int64_arr(ndims)
                
                chunk  = self.__read_arr( data_type, count )

                if ( ndims == 1 ) :
                    data[start[0]:start[0]+count[0]:stride[0]] = chunk
                elif ( ndims == 2 ) :
                    data[start[1]:start[1]+count[1]:stride[1],
                         start[0]:start[0]+count[0]:stride[0]] = chunk
                else :
                    data[start[2]:start[2]+count[2]:stride[2],
                         start[1]:start[1]+count[1]:stride[1],
                         start[0]:start[0]+count[0]:stride[0]] = chunk
            
            else:
                
                self.__record_skip(rec)
        
        # If requested, position file pointer at the end of the cdset_start record
        if ( pos == 1 ):
            self.__file.seek( cdset_start_end )

        return data    

# -----------------------------------------------------------------------------
# Read arbitrary ZDF element
# -----------------------------------------------------------------------------

    def read_element( self, rec = None, name = None, type_id = None):
        """Reads abitrary zdf element from file

        Args:
            rec (ZDF_Record, optional): If not set the routine will read the record before reading the data.
            name (str, optional): If set the routine will only read the record if it matches the supplied name.
            type_id (int, optional): If set the routine will only read the record if it matches the supplied name.

        Returns:
            multiple data types: zdf element data
        """
        if ( rec is None ):
            rec = self.read_record()
        
        if ( name is not None ):
            if (name != rec.name ):
                print("(*warning*) Requested name does not match record name", file=sys.stderr)
                print("(*warning*) expected '{}', found '{}".format(name, rec.name), file=sys.stderr)
                self.__record_skip(rec)
                return False
        else:
            name = rec.name
        
        if ( type_id is not None ):
            if (type_id != self.record_type(rec.id) ):
                print("(*warning*) Requested type does not match record type", file=sys.stderr)
                print("(*warning*) expected '{}', found '{}".format(type_id, self.record_type(rec.id)), file=sys.stderr)
                self.__record_skip(rec)
                return False
        else:
            type_id = self.record_type(rec.id)
        
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
# Read particle data
# -----------------------------------------------------------------------------
    
    def read_part_data( self, quants ):
        """Read particle data

        Args:
            quants (list(str)): Particle quantities in the zdf file

        Returns:
            dict(): Dictionary with particle data indexed with quantity names
        """

        # Read all quantities
        data = dict()
        
        for q in quants:
            rec  = self.read_record()
            
            # Sanity check, this should never happen
            if ( rec.name != q ):
                print("(*error*) Expecting {} record, {} found".format(
                    q, rec.name ) )
                      
            # Particle data can be either a standard dataset or a chunked dataset
            type_id = self.record_type(rec.id)
            if ( type_id == "dataset" ):
                data[q] = self.read_dataset( rec = rec )
            elif ( type_id == "cdset_start" ):
                # Read chunked dataset and position file pointer after the
                # cdset_start record
                data[q] = self.read_cdset( rec = rec, pos = 1 )
            else:
                print("(*error*) Unable to read particle data, {} record found".format(type_id),
                      file=sys.stderr)
                data[q] = None
        
        return data

# -----------------------------------------------------------------------------
# Read track data
# -----------------------------------------------------------------------------
    
    def read_track_data( self, trackInfo ):
        """read_track_data()

        Reads track data from zdf file

        Args:
            trackInfo (ZDF_Tracks_Info): Tracks dataset information

        Returns:
            list(numpy.array): Track Data
        """

        rec  = self.read_record()
        if ( rec.name != 'itermap' ):
                print("(*error*) Expecting itermap record, {} found".format(rec.name ) ,
                      file=sys.stderr)
        itermap = self.read_cdset( rec = rec, pos = 1 )

        rec  = self.read_record()
        if ( rec.name != 'data' ):
                print("(*error*) Expecting data record, {} found".format(rec.name ) ,
                      file=sys.stderr)
        data = self.read_cdset( rec = rec )
        
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
        """
        # Position file after magic number
        self.__file.seek(4)

        rec_list = []
        while True:
            rec = self.read_record(skip=True)
            if (rec is None):
                break
            else:
                rec_list.append(rec)

        if (printRec and (len(rec_list) > 0)):
            print('Position     Size(bytes)  Type         Name')
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
    if ( info.type == "grid" ):
        return __str__(self.grid)
    elif ( info.type == "particles" ):
        return __str__(self.particles)
    elif ( info.type == "tracks-2" ):
        return __str__(self.tracks)
    elif ( info.type == "iteration" ):
        return __str__(self.iteration)
    else:
        msg = "<ZDF_Grid_info>\n"
        msg += "undefined\n"
        return msg

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
    # Open file
    zdf = ZDFfile( file_name )

    # Check file type and read metadata
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
        zdf.close()
        return None

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
    (data, info) : ( numpy.ndarray | dictionary | list, ZDF_Info )
        Tuple containing file data and metadata. Data will be:
        + a numpy.ndarray for grid data
        + a dictionary of numpy.array for particle data (one entry per quantity).
        + a list of numpy.array for track data (one entry per track)
        Metadata is returned as a ZDF_Info object.
        If file is invalid None is returned.
    """
    # Open file
    zdf = ZDFfile( file_name )

    # Check file type and read metadata
    info = ZDF_Info()

    info.type = zdf.read_string()
    if ( info.type == "grid" ):
        info.grid = zdf.read_grid_info()
        info.iteration = zdf.read_iteration()
        data = zdf.read_element()
    elif ( info.type == "particles" ):
        info.particles = zdf.read_part_info()
        info.iteration = zdf.read_iteration()
        data = zdf.read_part_data( info.particles.quants )
    elif ( info.type == "tracks-2" ):
        info.tracks = zdf.read_track_info()
        data = zdf.read_track_data( info.tracks )
    else:
        print("File is not a valid ZDF grid, particles or tracks file", file=sys.stderr)
        zdf.close()
        return None

    # Close file
    zdf.close()

    return (data,info)

def list(file_name):
    """Prints a list of all zdf file contents

    Args:
        file_name (std): File name of ZDF data file, should include path
    """
    zdf = ZDFfile(file_name)
    zdf.list(True)
    zdf.close()

