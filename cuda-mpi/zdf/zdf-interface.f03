!------------------------------------------------------------------------------
! Fortran 2003 interface routines for ZDF
!
! Notes:
! - All routines accept an optional `ierr` argument, set to 0 on success and -1 on
!   error. If `ierr` is not present, an error is fatal (error stop), since continuing
!   would typically crash inside the C library (e.g. writing to a file that failed
!   to open).
! - Strings are passed to C with trailing blanks removed, using zdf_c_str().
! - Data arrays must be in host memory.
!------------------------------------------------------------------------------
module zdf

use iso_c_binding
use iso_fortran_env, only : error_unit

implicit none

integer, private, parameter :: p_single = kind(1.0e0)
integer, private, parameter :: p_double = kind(1.0d0)
integer, private, parameter :: p_byte   = selected_int_kind(2)
integer, private, parameter :: p_int64  = c_int64_t

integer, parameter :: zdf_str_maxlen = 256

integer, parameter :: zdf_max_dims = 3

integer, parameter :: zdf_file_create = 0
integer, parameter :: zdf_file_read   = 1
integer, parameter :: zdf_file_update = 2

integer, parameter :: zdf_axis_linear = 0
integer, parameter :: zdf_axis_log10  = 1
integer, parameter :: zdf_axis_log2   = 2

enum, bind(c)
    enumerator :: zdf_null, zdf_int8, zdf_uint8, zdf_int16, zdf_uint16, zdf_int32, &
                   zdf_uint32, zdf_int64, zdf_uint64, zdf_float32, zdf_float64, &
                   zdf_complex64, zdf_complex128
endenum

! Maximum number of quantities in particle / track info
integer, private, parameter :: p_max_quants = 256

integer, parameter :: zdf_max_quants = p_max_quants

type, bind(C) :: t_zdf_file
    type(c_ptr) :: fp = C_NULL_PTR
    integer(c_int) :: mode
    integer(c_int32_t) :: ndatasets
end type

type :: t_zdf_dataset
    character(len = zdf_str_maxlen ) :: name = 'DATASET'
    integer(c_int32_t) :: data_type = 0
    integer(c_int32_t) :: ndims = 0
    integer(c_int64_t), dimension(zdf_max_dims) :: count = 0
    type(c_ptr) :: data = C_NULL_PTR
    integer(c_int64_t) :: id = -1
    integer(c_int64_t) :: offset = 0
end type

type :: t_zdf_grid_axis
    integer :: type = zdf_axis_linear
    real(p_double) :: min = 0.0
    real(p_double) :: max = 1.0
    character(len = zdf_str_maxlen ) :: name = ''
    character(len = zdf_str_maxlen ) :: label = ''
    character(len = zdf_str_maxlen ) :: units = ''
end type

type :: t_zdf_iteration
    character(len = zdf_str_maxlen ) :: name = 'ITERATION'
    integer :: n
    real(p_double) :: t
    character(len = zdf_str_maxlen ) :: time_units = ''
end type

type :: t_zdf_grid_info
    integer :: ndims = -1
    integer, dimension(zdf_max_dims) :: count = 0
    character(len = zdf_str_maxlen ) :: name = 'GRID'
    character(len = zdf_str_maxlen ) :: label = ''
    character(len = zdf_str_maxlen ) :: units = ''
    type(t_zdf_grid_axis), dimension(zdf_max_dims) :: axis
end type

type :: t_zdf_part_info
    character(len = zdf_str_maxlen ) :: name = 'PARTICLES'
    character(len = zdf_str_maxlen ) :: label = ''
    integer(p_int64) :: np
    integer :: nquants
    character(len = zdf_str_maxlen ), dimension(p_max_quants) :: quants  = ''
    character(len = zdf_str_maxlen ), dimension(p_max_quants) :: qlabels = ''
    character(len = zdf_str_maxlen ), dimension(p_max_quants) :: qunits  = ''
    logical :: has_tags
end type

type, bind(C) :: t_zdf_chunk
    integer(c_int64_t), dimension(zdf_max_dims) :: count
    integer(c_int64_t), dimension(zdf_max_dims) :: start
    integer(c_int64_t), dimension(zdf_max_dims) :: stride
    type(c_ptr) :: data
end type

type :: t_zdf_track_info
    character(len = zdf_str_maxlen ) :: name = "TRACKS"
    character(len = zdf_str_maxlen ) :: label = ""
    integer :: ntracks
    integer :: ndump
    integer :: niter
    integer :: nquants
    character(len = zdf_str_maxlen ), dimension(p_max_quants) :: quants  = ''
    character(len = zdf_str_maxlen ), dimension(p_max_quants) :: qlabels = ''
    character(len = zdf_str_maxlen ), dimension(p_max_quants) :: qunits  = ''
end type

! C types to be used in C function calls
type, bind(C) :: t_zdf_dataset_c
    type(c_ptr) :: name = C_NULL_PTR
    integer(c_int32_t) :: data_type = 0
    integer(c_int32_t) :: ndims = 0
    integer(c_int64_t), dimension(zdf_max_dims) :: count = 0
    type(c_ptr) :: data = C_NULL_PTR
    integer(c_int64_t) :: id = -1
    integer(c_int64_t) :: offset = 0
end type

type, bind(C) :: t_zdf_iteration_c
    type(c_ptr) :: name = C_NULL_PTR
    integer(c_int32_t) :: n
    real(c_double) :: t
    type(c_ptr) :: time_units = C_NULL_PTR
end type

type, bind(C) :: t_zdf_grid_axis_c
    type(c_ptr) :: name = C_NULL_PTR
    integer(c_int) :: type
    real(c_double) :: min, max
    type(c_ptr) :: label = C_NULL_PTR
    type(c_ptr) :: units = C_NULL_PTR
end type

type, bind(C) :: t_zdf_grid_info_c
    type(c_ptr) :: name = C_NULL_PTR
    integer(c_int32_t) :: ndims
    integer(c_int64_t), dimension(zdf_max_dims) :: count
    type(c_ptr) :: label = C_NULL_PTR
    type(c_ptr) :: units = C_NULL_PTR
    type(c_ptr) :: axis = C_NULL_PTR
end type

type, bind(C) :: t_zdf_part_info_c
    type(c_ptr) :: name = C_NULL_PTR
    type(c_ptr) :: label = C_NULL_PTR
    integer(c_int64_t) :: np
    integer(c_int32_t) :: nquants
    type(c_ptr) :: quants = C_NULL_PTR
    type(c_ptr) :: qlabels = C_NULL_PTR
    type(c_ptr) :: qunits = C_NULL_PTR
end type

type, bind(C) :: t_zdf_track_info_c
    type(c_ptr) :: name = C_NULL_PTR
    type(c_ptr) :: label = C_NULL_PTR
    integer(c_int32_t) :: ntracks
    integer(c_int32_t) :: ndump
    integer(c_int32_t) :: niter
    integer(c_int32_t) :: nquants
    type(c_ptr) :: quants = C_NULL_PTR
    type(c_ptr) :: qlabels = C_NULL_PTR
    type(c_ptr) :: qunits = C_NULL_PTR
end type

!------------------------------------------------------------------------------
! Buffers holding null terminated copies of Fortran strings. The C structures
! point to these (using c_loc), so they must remain in scope while the C routine
! is called. These, and the zdf_*_to_c / zdf_dataset_from_c / zdf_c_str /
! zdf_set_err utilities, are meant for internal use by the zdf and zdf_parallel
! modules. Buffers are one character longer than the Fortran
! string components, so the null terminator always fits.
!------------------------------------------------------------------------------

integer, private, parameter :: p_cstr_len = zdf_str_maxlen + 1

type :: t_zdf_iter_buf
    character(kind=c_char, len=p_cstr_len) :: name, time_units
end type

type :: t_zdf_grid_buf
    character(kind=c_char, len=p_cstr_len) :: name, label, units
    type(t_zdf_grid_axis_c), dimension(zdf_max_dims) :: axis
    character(kind=c_char, len=p_cstr_len), dimension(zdf_max_dims) :: axis_name, axis_label, axis_units
end type

type :: t_zdf_quants_buf
    character(kind=c_char, len=p_cstr_len), dimension(:), allocatable :: quants, qlabels, qunits
    type(c_ptr), dimension(:), allocatable :: pquants, pqlabels, pqunits
end type

type :: t_zdf_part_buf
    character(kind=c_char, len=p_cstr_len) :: name, label
    type(t_zdf_quants_buf) :: q
end type

type :: t_zdf_dataset_buf
    character(kind=c_char, len=p_cstr_len) :: name
end type

!------------------------------------------------------------------------------
! C library interfaces
!------------------------------------------------------------------------------
interface

    function zdf_open_file_c( zdf, filename, mode ) bind(C, name="zdf_open_file")
        import
        integer(c_int) :: zdf_open_file_c
        type(t_zdf_file) :: zdf
        character(kind=c_char), dimension(*), intent(in) :: filename
        integer(c_int), value :: mode
    end function

    function zdf_close_file_c( zdf ) bind(C, name="zdf_close_file")
        import
        integer(c_int) :: zdf_close_file_c
        type(t_zdf_file) :: zdf
    end function

    function zdf_add_string_c( zdf, name, str ) bind(C, name="zdf_add_string")
        import
        integer(c_size_t) :: zdf_add_string_c
        type(t_zdf_file) :: zdf
        character(kind=c_char), dimension(*), intent(in) :: name, str
    end function

    function zdf_add_int32_c( zdf, name, val ) bind(C, name="zdf_add_int32")
        import
        integer(c_size_t) :: zdf_add_int32_c
        type(t_zdf_file) :: zdf
        character(kind=c_char), dimension(*), intent(in) :: name
        integer(c_int32_t), value :: val
    end function

    function zdf_add_double_c( zdf, name, val ) bind(C, name="zdf_add_double")
        import
        integer(c_size_t) :: zdf_add_double_c
        type(t_zdf_file) :: zdf
        character(kind=c_char), dimension(*), intent(in) :: name
        real(c_double), value :: val
    end function

    function zdf_add_iteration_c( zdf, iter ) bind(C, name="zdf_add_iteration")
        import
        integer(c_size_t) :: zdf_add_iteration_c
        type(t_zdf_file) :: zdf
        type(t_zdf_iteration_c), intent(in) :: iter
    end function

    function zdf_add_grid_info_c( zdf, info ) bind(C, name="zdf_add_grid_info")
        import
        integer(c_size_t) :: zdf_add_grid_info_c
        type(t_zdf_file) :: zdf
        type(t_zdf_grid_info_c), intent(in) :: info
    end function

    function zdf_add_part_info_c( zdf, info ) bind(C, name="zdf_add_part_info")
        import
        integer(c_size_t) :: zdf_add_part_info_c
        type(t_zdf_file) :: zdf
        type(t_zdf_part_info_c), intent(in) :: info
    end function

    function zdf_add_track_info_c( zdf, info ) bind(C, name="zdf_add_track_info")
        import
        integer(c_size_t) :: zdf_add_track_info_c
        type(t_zdf_file) :: zdf
        type(t_zdf_track_info_c), intent(in) :: info
    end function

    function zdf_add_dataset_c( zdf, dataset ) bind(C, name="zdf_add_dataset")
        import
        integer(c_size_t) :: zdf_add_dataset_c
        type(t_zdf_file) :: zdf
        type(t_zdf_dataset_c) :: dataset
    end function

    function zdf_start_cdset_c( zdf, dataset ) bind(C, name="zdf_start_cdset")
        import
        integer(c_size_t) :: zdf_start_cdset_c
        type(t_zdf_file) :: zdf
        type(t_zdf_dataset_c) :: dataset
    end function

    function zdf_write_cdset_c( zdf, dataset, chunk ) bind(C, name="zdf_write_cdset")
        import
        integer(c_size_t) :: zdf_write_cdset_c
        type(t_zdf_file) :: zdf
        type(t_zdf_dataset_c), intent(in) :: dataset
        type(t_zdf_chunk), intent(in) :: chunk
    end function

    function zdf_end_cdset_c( zdf, dataset ) bind(C, name="zdf_end_cdset")
        import
        integer(c_size_t) :: zdf_end_cdset_c
        type(t_zdf_file) :: zdf
        type(t_zdf_dataset_c), intent(in) :: dataset
    end function

    function zdf_open_dataset_c( zdf, dataset ) bind(C, name="zdf_open_dataset")
        import
        integer(c_size_t) :: zdf_open_dataset_c
        type(t_zdf_file) :: zdf
        type(t_zdf_dataset_c) :: dataset
    end function

    function zdf_extend_dataset_c( zdf, dataset, new_count ) bind(C, name="zdf_extend_dataset")
        import
        integer(c_int) :: zdf_extend_dataset_c
        type(t_zdf_file) :: zdf
        type(t_zdf_dataset_c) :: dataset
        integer(c_int64_t), dimension(*), intent(in) :: new_count
    end function

    function zdf_open_grid_file_c( zdf, info, iter, path ) bind(C, name="zdf_open_grid_file")
        import
        integer(c_int) :: zdf_open_grid_file_c
        type(t_zdf_file) :: zdf
        type(t_zdf_grid_info_c), intent(in) :: info
        type(t_zdf_iteration_c), intent(in) :: iter
        character(kind=c_char), dimension(*), intent(in) :: path
    end function

    function zdf_save_grid_c( data, data_type, info, iter, path ) bind(C, name="zdf_save_grid")
        import
        integer(c_int) :: zdf_save_grid_c
        type(c_ptr), value :: data
        integer(c_int), value :: data_type
        type(t_zdf_grid_info_c), intent(in) :: info
        type(t_zdf_iteration_c), intent(in) :: iter
        character(kind=c_char), dimension(*), intent(in) :: path
    end function

    function zdf_open_part_file_c( zdf, info, iter, path ) bind(C, name="zdf_open_part_file")
        import
        integer(c_int) :: zdf_open_part_file_c
        type(t_zdf_file) :: zdf
        type(t_zdf_part_info_c), intent(in) :: info
        type(t_zdf_iteration_c), intent(in) :: iter
        character(kind=c_char), dimension(*), intent(in) :: path
    end function

    function zdf_add_quant_part_file_c( zdf, name, data, np ) bind(C, name="zdf_add_quant_part_file")
        import
        integer(c_int) :: zdf_add_quant_part_file_c
        type(t_zdf_file) :: zdf
        character(kind=c_char), dimension(*), intent(in) :: name
        type(c_ptr), value :: data
        integer(c_int64_t), value :: np
    end function

end interface

private :: zdf_quants_to_c, zdf_save_grid_generic

!------------------------------------------------------------------------------
! Fortran interfaces
!------------------------------------------------------------------------------
interface zdf_sizeof
    module procedure zdf_sizeof
end interface

interface zdf_open_file
    module procedure zdf_open_file
end interface

interface zdf_close_file
    module procedure zdf_close_file
end interface

interface zdf_open_grid_file
    module procedure zdf_open_grid_file
end interface

interface zdf_open_part_file
    module procedure zdf_open_part_file
end interface

interface zdf_add
    module procedure zdf_add_string
    module procedure zdf_add_int32
    module procedure zdf_add_r8
    module procedure zdf_add_iteration
    module procedure zdf_add_grid_info
    module procedure zdf_add_part_info
    module procedure zdf_add_track_info
    module procedure zdf_add_dataset
end interface

! Chunked dataset Interface
interface zdf_start_cdset
    module procedure zdf_start_cdset
end interface

interface zdf_write_cdset
    module procedure zdf_write_cdset
end interface

interface zdf_end_cdset
    module procedure zdf_end_cdset
end interface

interface zdf_open_cdset
    module procedure zdf_open_cdset
end interface

interface zdf_extend_cdset
    module procedure zdf_extend_cdset
end interface

! High level interfaces
interface zdf_save_grid
    module procedure zdf_save_grid_1D_r4
    module procedure zdf_save_grid_2D_r4
    module procedure zdf_save_grid_3D_r4
    module procedure zdf_save_grid_1D_r8
    module procedure zdf_save_grid_2D_r8
    module procedure zdf_save_grid_3D_r8
end interface

interface zdf_add_quant_part_file
    module procedure zdf_add_quant_part_file
end interface

contains

!------------------------------------------------------------------------------
! *************************** Internal utilities ******************************
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!> Returns a null terminated C string from a Fortran string (trailing blanks
!! are removed). The result can be used directly as an actual argument for C
!! routines taking a `character(kind=c_char), dimension(*)` dummy argument, or
!! assigned to a buffer referenced from a C structure using c_loc().
!! @param   fstr    Fortran string
!! @return          C string, of length len_trim(fstr) + 1
pure function zdf_c_str( fstr ) result( cstr )

    character(len=*), intent(in) :: fstr
    character(kind=c_char, len=len_trim(fstr)+1) :: cstr

    cstr = trim(fstr) // C_NULL_CHAR

end function zdf_c_str

!------------------------------------------------------------------------------
!> Sets error code / handles error
!! @param   ok      Operation was successful
!! @param   routine Routine name (for error message)
!! @param   ierr    (optional) Error code, 0 on success, -1 on error. If not
!!                  present errors are fatal.
subroutine zdf_set_err( ok, routine, ierr )

    logical, intent(in) :: ok
    character(len=*), intent(in) :: routine
    integer, intent(out), optional :: ierr

    if ( present( ierr ) ) then
        if ( ok ) then
            ierr = 0
        else
            ierr = -1
        endif
    else if ( .not. ok ) then
        write( error_unit, '(A,A,A)' ) '(*error*) ', routine, ' failed, aborting.'
        error stop 1
    endif

end subroutine zdf_set_err

!------------------------------------------------------------------------------
!> Prepares C iteration info
subroutine zdf_iter_to_c( iter, iter_c, buf )

    type(t_zdf_iteration), intent(in) :: iter
    type(t_zdf_iteration_c), intent(out) :: iter_c
    type(t_zdf_iter_buf), intent(inout), target :: buf

    buf % name = zdf_c_str( iter % name )
    buf % time_units = zdf_c_str( iter % time_units )

    iter_c % name       = c_loc( buf % name )
    iter_c % n          = int( iter % n, c_int32_t )
    iter_c % t          = real( iter % t, c_double )
    iter_c % time_units = c_loc( buf % time_units )

end subroutine zdf_iter_to_c

!------------------------------------------------------------------------------
!> Prepares C grid info
subroutine zdf_grid_to_c( info, info_c, buf, ok )

    type(t_zdf_grid_info), intent(in) :: info
    type(t_zdf_grid_info_c), intent(out) :: info_c
    type(t_zdf_grid_buf), intent(inout), target :: buf
    logical, intent(out) :: ok

    integer :: i

    ok = ( info % ndims >= 1 .and. info % ndims <= zdf_max_dims )
    if ( .not. ok ) then
        write( error_unit, '(A,I0)' ) '(*error*) Invalid number of grid dimensions: ', info % ndims
        return
    endif

    buf % name = zdf_c_str( info % name )
    buf % label = zdf_c_str( info % label )
    buf % units = zdf_c_str( info % units )

    info_c % name  = c_loc( buf % name )
    info_c % label = c_loc( buf % label )
    info_c % units = c_loc( buf % units )

    info_c % ndims = int( info % ndims, c_int32_t )
    info_c % count = 0
    do i = 1, info % ndims
        info_c % count(i) = int( info % count(i), c_int64_t )

        buf % axis_name(i) = zdf_c_str( info % axis(i) % name )
        buf % axis_label(i) = zdf_c_str( info % axis(i) % label )
        buf % axis_units(i) = zdf_c_str( info % axis(i) % units )

        buf % axis(i) % type  = int( info % axis(i) % type, c_int )
        buf % axis(i) % min   = real( info % axis(i) % min, c_double )
        buf % axis(i) % max   = real( info % axis(i) % max, c_double )
        buf % axis(i) % name  = c_loc( buf % axis_name(i) )
        buf % axis(i) % label = c_loc( buf % axis_label(i) )
        buf % axis(i) % units = c_loc( buf % axis_units(i) )
    enddo

    info_c % axis = c_loc( buf % axis )

end subroutine zdf_grid_to_c

!------------------------------------------------------------------------------
!> Prepares C arrays of quantity names, labels and units
subroutine zdf_quants_to_c( nquants, quants, qlabels, qunits, buf, ok )

    integer, intent(in) :: nquants
    character(len=zdf_str_maxlen), dimension(:), intent(in) :: quants, qlabels, qunits
    type(t_zdf_quants_buf), intent(inout), target :: buf
    logical, intent(out) :: ok

    integer :: i, n

    ok = ( nquants >= 0 .and. nquants <= p_max_quants )
    if ( .not. ok ) then
        write( error_unit, '(A,I0,A,I0,A)' ) '(*error*) Invalid number of quantities: ', nquants, &
            ' (maximum is ', p_max_quants, ')'
        return
    endif

    ! Always allocate at least 1 element, so that c_loc() is valid
    n = max( nquants, 1 )
    allocate( buf % quants( n ), buf % qlabels( n ), buf % qunits( n ) )
    allocate( buf % pquants( n ), buf % pqlabels( n ), buf % pqunits( n ) )

    do i = 1, nquants
        buf % quants(i) = zdf_c_str( quants(i) )
        buf % qlabels(i) = zdf_c_str( qlabels(i) )
        buf % qunits(i) = zdf_c_str( qunits(i) )

        buf % pquants(i)  = c_loc( buf % quants(i) )
        buf % pqlabels(i) = c_loc( buf % qlabels(i) )
        buf % pqunits(i)  = c_loc( buf % qunits(i) )
    enddo

end subroutine zdf_quants_to_c

!------------------------------------------------------------------------------
!> Prepares C particle info
subroutine zdf_part_to_c( info, info_c, buf, ok )

    type(t_zdf_part_info), intent(in) :: info
    type(t_zdf_part_info_c), intent(out) :: info_c
    type(t_zdf_part_buf), intent(inout), target :: buf
    logical, intent(out) :: ok

    call zdf_quants_to_c( info % nquants, info % quants, info % qlabels, info % qunits, buf % q, ok )
    if ( .not. ok ) return

    buf % name = zdf_c_str( info % name )
    buf % label = zdf_c_str( info % label )

    info_c % name    = c_loc( buf % name )
    info_c % label   = c_loc( buf % label )
    info_c % np      = int( info % np, c_int64_t )
    info_c % nquants = int( info % nquants, c_int32_t )
    info_c % quants  = c_loc( buf % q % pquants )
    info_c % qlabels = c_loc( buf % q % pqlabels )
    info_c % qunits  = c_loc( buf % q % pqunits )

end subroutine zdf_part_to_c

!------------------------------------------------------------------------------
!> Prepares C tracks info
subroutine zdf_track_to_c( info, info_c, buf, ok )

    type(t_zdf_track_info), intent(in) :: info
    type(t_zdf_track_info_c), intent(out) :: info_c
    type(t_zdf_part_buf), intent(inout), target :: buf
    logical, intent(out) :: ok

    call zdf_quants_to_c( info % nquants, info % quants, info % qlabels, info % qunits, buf % q, ok )
    if ( .not. ok ) return

    buf % name = zdf_c_str( info % name )
    buf % label = zdf_c_str( info % label )

    info_c % name    = c_loc( buf % name )
    info_c % label   = c_loc( buf % label )
    info_c % ntracks = int( info % ntracks, c_int32_t )
    info_c % ndump   = int( info % ndump, c_int32_t )
    info_c % niter   = int( info % niter, c_int32_t )
    info_c % nquants = int( info % nquants, c_int32_t )
    info_c % quants  = c_loc( buf % q % pquants )
    info_c % qlabels = c_loc( buf % q % pqlabels )
    info_c % qunits  = c_loc( buf % q % pqunits )

end subroutine zdf_track_to_c

!------------------------------------------------------------------------------
!> Prepares C dataset
subroutine zdf_dataset_to_c( dataset, dataset_c, buf )

    type(t_zdf_dataset), intent(in) :: dataset
    type(t_zdf_dataset_c), intent(out) :: dataset_c
    type(t_zdf_dataset_buf), intent(inout), target :: buf

    buf % name = zdf_c_str( dataset % name )

    dataset_c % name      = c_loc( buf % name )
    dataset_c % data_type = dataset % data_type
    dataset_c % ndims     = dataset % ndims
    dataset_c % count     = dataset % count
    dataset_c % data      = dataset % data
    dataset_c % id        = dataset % id
    dataset_c % offset    = dataset % offset

end subroutine zdf_dataset_to_c

!------------------------------------------------------------------------------
!> Copies values set by the C library back into the Fortran dataset
subroutine zdf_dataset_from_c( dataset_c, dataset )

    type(t_zdf_dataset_c), intent(in) :: dataset_c
    type(t_zdf_dataset), intent(inout) :: dataset

    dataset % data_type = dataset_c % data_type
    dataset % ndims     = dataset_c % ndims
    dataset % count     = dataset_c % count
    dataset % id        = dataset_c % id
    dataset % offset    = dataset_c % offset

end subroutine zdf_dataset_from_c

!------------------------------------------------------------------------------
! ****************************** File routines ********************************
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!> Opens ZDF file
!! @param   this        ZDF file
!! @param   filename    File name
!! @param   access_mode zdf_file_create, zdf_file_read or zdf_file_update
!! @param   ierr        (optional) Error code
subroutine zdf_open_file( this, filename, access_mode, ierr )

    type( t_zdf_file ), intent(inout) :: this
    character(len = *), intent(in)  :: filename
    integer, intent(in)             :: access_mode
    integer, intent(out), optional  :: ierr

    integer(c_int) :: ret

    ret = zdf_open_file_c( this, zdf_c_str(filename), int( access_mode, c_int ) )
    call zdf_set_err( ret /= 0, 'zdf_open_file', ierr )

end subroutine zdf_open_file

!------------------------------------------------------------------------------
!> Closes ZDF file
!! @param   this        ZDF file
!! @param   ierr        (optional) Error code
subroutine zdf_close_file( this, ierr )

    type( t_zdf_file ), intent(inout) :: this
    integer, intent(out), optional  :: ierr

    integer(c_int) :: ret

    ret = zdf_close_file_c( this )
    call zdf_set_err( ret /= 0, 'zdf_close_file', ierr )

end subroutine zdf_close_file

!------------------------------------------------------------------------------
! ****************************** Metadata *************************************
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!> Adds string element
subroutine zdf_add_string( file, name, str, ierr )

    type(t_zdf_file), intent(inout) :: file
    character(len=*), intent(in) :: name
    character(len=*), intent(in) :: str
    integer, intent(out), optional :: ierr

    integer(c_size_t) :: ret

    ret = zdf_add_string_c( file, zdf_c_str(name), zdf_c_str(str) )
    call zdf_set_err( ret > 0, 'zdf_add_string', ierr )

end subroutine zdf_add_string

!------------------------------------------------------------------------------
!> Adds int32 element
subroutine zdf_add_int32( file, name, val, ierr )

    type(t_zdf_file), intent(inout) :: file
    character(len=*), intent(in) :: name
    integer, intent(in) :: val
    integer, intent(out), optional :: ierr

    integer(c_size_t) :: ret

    ret = zdf_add_int32_c( file, zdf_c_str(name), int( val, c_int32_t ) )
    call zdf_set_err( ret > 0, 'zdf_add_int32', ierr )

end subroutine zdf_add_int32

!------------------------------------------------------------------------------
!> Adds float64 element
subroutine zdf_add_r8( file, name, val, ierr )

    type(t_zdf_file), intent(inout) :: file
    character(len=*), intent(in) :: name
    real(p_double), intent(in) :: val
    integer, intent(out), optional :: ierr

    integer(c_size_t) :: ret

    ret = zdf_add_double_c( file, zdf_c_str(name), real( val, c_double ) )
    call zdf_set_err( ret > 0, 'zdf_add_double', ierr )

end subroutine zdf_add_r8

!------------------------------------------------------------------------------
!> Adds iteration metadata
subroutine zdf_add_iteration( file, iter, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_iteration), intent(in) :: iter
    integer, intent(out), optional :: ierr

    type(t_zdf_iteration_c) :: iter_c
    type(t_zdf_iter_buf), target :: buf
    integer(c_size_t) :: ret

    call zdf_iter_to_c( iter, iter_c, buf )
    ret = zdf_add_iteration_c( file, iter_c )
    call zdf_set_err( ret > 0, 'zdf_add_iteration', ierr )

end subroutine zdf_add_iteration

!------------------------------------------------------------------------------
!> Adds grid information metadata
subroutine zdf_add_grid_info( file, info, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_grid_info), intent(in) :: info
    integer, intent(out), optional :: ierr

    type(t_zdf_grid_info_c) :: info_c
    type(t_zdf_grid_buf), target :: buf
    logical :: ok

    call zdf_grid_to_c( info, info_c, buf, ok )
    if ( ok ) ok = ( zdf_add_grid_info_c( file, info_c ) > 0 )
    call zdf_set_err( ok, 'zdf_add_grid_info', ierr )

end subroutine zdf_add_grid_info

!------------------------------------------------------------------------------
!> Adds particle information metadata
subroutine zdf_add_part_info( file, info, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_part_info), intent(in) :: info
    integer, intent(out), optional :: ierr

    type( t_zdf_part_info_c ) :: info_c
    type( t_zdf_part_buf ), target :: buf
    logical :: ok

    call zdf_part_to_c( info, info_c, buf, ok )
    if ( ok ) ok = ( zdf_add_part_info_c( file, info_c ) > 0 )
    call zdf_set_err( ok, 'zdf_add_part_info', ierr )

end subroutine zdf_add_part_info

!------------------------------------------------------------------------------
!> Adds tracks information metadata
subroutine zdf_add_track_info( file, info, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_track_info), intent(in) :: info
    integer, intent(out), optional :: ierr

    type( t_zdf_track_info_c ) :: info_c
    type( t_zdf_part_buf ), target :: buf
    logical :: ok

    call zdf_track_to_c( info, info_c, buf, ok )
    if ( ok ) ok = ( zdf_add_track_info_c( file, info_c ) > 0 )
    call zdf_set_err( ok, 'zdf_add_track_info', ierr )

end subroutine zdf_add_track_info

!------------------------------------------------------------------------------
! ****************************** Datasets *************************************
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!> Adds dataset to file. The dataset id and file offset are updated.
subroutine zdf_add_dataset( file, dataset, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_dataset), intent(inout) :: dataset
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: dataset_c
    type(t_zdf_dataset_buf), target :: buf
    integer(c_size_t) :: ret

    call zdf_dataset_to_c( dataset, dataset_c, buf )
    ret = zdf_add_dataset_c( file, dataset_c )
    call zdf_dataset_from_c( dataset_c, dataset )
    call zdf_set_err( ret > 0, 'zdf_add_dataset', ierr )

end subroutine zdf_add_dataset

!------------------------------------------------------------------------------
!> Start ZDF chunked dataset. The dataset id and file offset are updated.
!! @param   file        ZDF file
!! @param   dataset     ZDF chunked dataset
!! @param   ierr        (optional) Error code
subroutine zdf_start_cdset( file, dataset, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_dataset), intent(inout) :: dataset
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: dataset_c
    type(t_zdf_dataset_buf), target :: buf
    integer(c_size_t) :: ret

    call zdf_dataset_to_c( dataset, dataset_c, buf )
    ret = zdf_start_cdset_c( file, dataset_c )
    call zdf_dataset_from_c( dataset_c, dataset )
    call zdf_set_err( ret > 0, 'zdf_start_cdset', ierr )

end subroutine zdf_start_cdset

!------------------------------------------------------------------------------
!> Write ZDF chunked dataset
!> @param file      ZDF file
!> @param cdset     Chunked dataset
!> @param chunk     Data chunk
!! @param ierr      (optional) Error code
subroutine zdf_write_cdset( file, cdset, chunk, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_dataset), intent(in) :: cdset
    type(t_zdf_chunk), intent(in)   :: chunk
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: cdset_c
    type(t_zdf_dataset_buf), target :: buf
    integer(c_size_t) :: ret

    call zdf_dataset_to_c( cdset, cdset_c, buf )
    ret = zdf_write_cdset_c( file, cdset_c, chunk )
    call zdf_set_err( ret > 0, 'zdf_write_cdset', ierr )

end subroutine zdf_write_cdset

!------------------------------------------------------------------------------
!> Ends access to chunked dataset
!!
!! @param   file    ZDF file
!! @param   cdset   ZDF chunked dataset
!! @param   ierr    (optional) Error code
subroutine zdf_end_cdset( file, cdset, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_dataset), intent(in) :: cdset
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: cdset_c
    type(t_zdf_dataset_buf), target :: buf
    integer(c_size_t) :: ret

    call zdf_dataset_to_c( cdset, cdset_c, buf )
    ret = zdf_end_cdset_c( file, cdset_c )
    call zdf_set_err( ret > 0, 'zdf_end_cdset', ierr )

end subroutine zdf_end_cdset

!------------------------------------------------------------------------------
!> Extends dataset dimensions
!!
!! @param   file        ZDF file
!! @param   cdset       ZDF chunked dataset
!! @param   new_count   New dataset dimensions
!! @param   ierr        (optional) Error code
subroutine zdf_extend_cdset( file, cdset, new_count, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_dataset), intent(inout) :: cdset
    integer(c_int64_t), dimension(:), intent(in) :: new_count
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: cdset_c
    type(t_zdf_dataset_buf), target :: buf
    integer(c_int64_t), dimension(zdf_max_dims) :: count
    logical :: ok

    ok = ( cdset % ndims >= 1 .and. cdset % ndims <= zdf_max_dims .and. size( new_count ) >= cdset % ndims )
    if ( ok ) then
        count = cdset % count
        count( 1:cdset % ndims ) = new_count( 1:cdset % ndims )
        call zdf_dataset_to_c( cdset, cdset_c, buf )
        ok = ( zdf_extend_dataset_c( file, cdset_c, count ) /= 0 )
        call zdf_dataset_from_c( cdset_c, cdset )
    else
        write( error_unit, '(A)' ) '(*error*) zdf_extend_cdset: invalid dimensions'
    endif
    call zdf_set_err( ok, 'zdf_extend_cdset', ierr )

end subroutine zdf_extend_cdset

!------------------------------------------------------------------------------
!> Opens dataset for reading. The dataset is selected by name, and the
!! dataset type, dimensions, id and file offset are read from file.
!!
!! @param   file        ZDF file
!! @param   dataset     ZDF dataset (name must be set)
!! @param   ierr        (optional) Error code. Use this to check if a dataset
!!                      exists without aborting.
subroutine zdf_open_cdset( file, dataset, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_dataset), intent(inout) :: dataset
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: dataset_c
    type(t_zdf_dataset_buf), target :: buf
    integer(c_size_t) :: ret

    call zdf_dataset_to_c( dataset, dataset_c, buf )
    ret = zdf_open_dataset_c( file, dataset_c )
    if ( ret > 0 ) call zdf_dataset_from_c( dataset_c, dataset )
    call zdf_set_err( ret > 0, 'zdf_open_cdset', ierr )

end subroutine zdf_open_cdset

!------------------------------------------------------------------------------
! *************************** High Level Interfaces ***************************
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!> Saves grid file (generic version, data must be contiguous)
subroutine zdf_save_grid_generic( data, data_type, info, iteration, path, ierr )

    type(c_ptr), intent(in) :: data
    integer, intent(in)     :: data_type
    type(t_zdf_grid_info), intent(in) :: info
    type(t_zdf_iteration), intent(in) :: iteration
    character(len=*), intent(in) :: path
    integer, intent(out), optional :: ierr

    type(t_zdf_iteration_c) :: iter_c
    type(t_zdf_iter_buf), target :: iter_buf
    type(t_zdf_grid_info_c) :: info_c
    type(t_zdf_grid_buf), target :: grid_buf
    logical :: ok

    call zdf_iter_to_c( iteration, iter_c, iter_buf )
    call zdf_grid_to_c( info, info_c, grid_buf, ok )
    if ( ok ) then
        ok = ( zdf_save_grid_c( data, int( data_type, c_int ), info_c, iter_c, &
                                zdf_c_str(path) ) /= 0 )
    endif
    call zdf_set_err( ok, 'zdf_save_grid', ierr )

end subroutine zdf_save_grid_generic

!------------------------------------------------------------------------------
!> Saves grid file. Grid dimensions in info are set from the data array.
!! Non-contiguous arrays (e.g. strided sections) are copied by the compiler.
subroutine zdf_save_grid_1D_r4( data, info, iteration, path, ierr )

    real(p_single), dimension( : ), contiguous, target, intent(in) :: data
    type(t_zdf_grid_info), intent(inout) :: info
    type(t_zdf_iteration), intent(in) :: iteration
    character(len=*), intent(in) :: path
    integer, intent(out), optional :: ierr

    info % ndims = 1
    info % count(1:1) = shape( data )
    call zdf_save_grid_generic( c_loc(data), zdf_float32, info, iteration, path, ierr )

end subroutine zdf_save_grid_1D_r4

subroutine zdf_save_grid_2D_r4( data, info, iteration, path, ierr )

    real(p_single), dimension( :, : ), contiguous, target, intent(in) :: data
    type(t_zdf_grid_info), intent(inout) :: info
    type(t_zdf_iteration), intent(in) :: iteration
    character(len=*), intent(in) :: path
    integer, intent(out), optional :: ierr

    info % ndims = 2
    info % count(1:2) = shape( data )
    call zdf_save_grid_generic( c_loc(data), zdf_float32, info, iteration, path, ierr )

end subroutine zdf_save_grid_2D_r4

subroutine zdf_save_grid_3D_r4( data, info, iteration, path, ierr )

    real(p_single), dimension( :, :, : ), contiguous, target, intent(in) :: data
    type(t_zdf_grid_info), intent(inout) :: info
    type(t_zdf_iteration), intent(in) :: iteration
    character(len=*), intent(in) :: path
    integer, intent(out), optional :: ierr

    info % ndims = 3
    info % count(1:3) = shape( data )
    call zdf_save_grid_generic( c_loc(data), zdf_float32, info, iteration, path, ierr )

end subroutine zdf_save_grid_3D_r4

subroutine zdf_save_grid_1D_r8( data, info, iteration, path, ierr )

    real(p_double), dimension( : ), contiguous, target, intent(in) :: data
    type(t_zdf_grid_info), intent(inout) :: info
    type(t_zdf_iteration), intent(in) :: iteration
    character(len=*), intent(in) :: path
    integer, intent(out), optional :: ierr

    info % ndims = 1
    info % count(1:1) = shape( data )
    call zdf_save_grid_generic( c_loc(data), zdf_float64, info, iteration, path, ierr )

end subroutine zdf_save_grid_1D_r8

subroutine zdf_save_grid_2D_r8( data, info, iteration, path, ierr )

    real(p_double), dimension( :, : ), contiguous, target, intent(in) :: data
    type(t_zdf_grid_info), intent(inout) :: info
    type(t_zdf_iteration), intent(in) :: iteration
    character(len=*), intent(in) :: path
    integer, intent(out), optional :: ierr

    info % ndims = 2
    info % count(1:2) = shape( data )
    call zdf_save_grid_generic( c_loc(data), zdf_float64, info, iteration, path, ierr )

end subroutine zdf_save_grid_2D_r8

subroutine zdf_save_grid_3D_r8( data, info, iteration, path, ierr )

    real(p_double), dimension( :, :, : ), contiguous, target, intent(in) :: data
    type(t_zdf_grid_info), intent(inout) :: info
    type(t_zdf_iteration), intent(in) :: iteration
    character(len=*), intent(in) :: path
    integer, intent(out), optional :: ierr

    info % ndims = 3
    info % count(1:3) = shape( data )
    call zdf_save_grid_generic( c_loc(data), zdf_float64, info, iteration, path, ierr )

end subroutine zdf_save_grid_3D_r8

!------------------------------------------------------------------------------
!> Opens grid file (path/name-iteration.zdf) and adds TYPE, GRID and ITERATION
!! metadata
subroutine zdf_open_grid_file( file, info, iteration, path, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_grid_info), intent(in) :: info
    type(t_zdf_iteration), intent(in) :: iteration
    character(len=*), intent(in) :: path
    integer, intent(out), optional :: ierr

    type(t_zdf_iteration_c) :: iter_c
    type(t_zdf_iter_buf), target :: iter_buf
    type(t_zdf_grid_info_c) :: info_c
    type(t_zdf_grid_buf), target :: grid_buf
    logical :: ok

    call zdf_iter_to_c( iteration, iter_c, iter_buf )
    call zdf_grid_to_c( info, info_c, grid_buf, ok )
    if ( ok ) then
        ok = ( zdf_open_grid_file_c( file, info_c, iter_c, zdf_c_str(path) ) /= 0 )
    endif
    call zdf_set_err( ok, 'zdf_open_grid_file', ierr )

end subroutine zdf_open_grid_file

!------------------------------------------------------------------------------
!> Opens particle file (path/particles-name-iteration.zdf) and adds TYPE,
!! PARTICLES and ITERATION metadata
subroutine zdf_open_part_file( file, info, iteration, path, ierr )

    type(t_zdf_file), intent(inout) :: file
    type(t_zdf_part_info), intent(in) :: info
    type(t_zdf_iteration), intent(in) :: iteration
    character(len=*), intent(in) :: path
    integer, intent(out), optional :: ierr

    type(t_zdf_iteration_c) :: iter_c
    type(t_zdf_iter_buf), target :: iter_buf
    type(t_zdf_part_info_c) :: info_c
    type(t_zdf_part_buf), target :: part_buf
    logical :: ok

    call zdf_iter_to_c( iteration, iter_c, iter_buf )
    call zdf_part_to_c( info, info_c, part_buf, ok )
    if ( ok ) then
        ok = ( zdf_open_part_file_c( file, info_c, iter_c, zdf_c_str(path) ) /= 0 )
    endif
    call zdf_set_err( ok, 'zdf_open_part_file', ierr )

end subroutine zdf_open_part_file

!------------------------------------------------------------------------------
!> Adds individual particle quantity to file
!! @param   file    ZDF file
!! @param   name    Quantity name
!! @param   data    Quantity data
!! @param   np      (optional) Number of particles to write, defaults to size(data)
!! @param   ierr    (optional) Error code
subroutine zdf_add_quant_part_file( file, name, data, np, ierr )

    type(t_zdf_file), intent(inout) :: file
    character(len=*), intent(in) :: name
    real(p_single), dimension(:), contiguous, target, intent(in) :: data
    integer, intent(in), optional :: np
    integer, intent(out), optional :: ierr

    integer(c_int64_t) :: n
    logical :: ok

    n = size( data, kind = c_int64_t )
    if ( present( np ) ) n = int( np, c_int64_t )

    ok = ( n >= 0 .and. n <= size( data, kind = c_int64_t ) )
    if ( ok ) then
        if ( n > 0 ) then
            ok = ( zdf_add_quant_part_file_c( file, zdf_c_str(name), c_loc( data ), n ) /= 0 )
        else
            ok = ( zdf_add_quant_part_file_c( file, zdf_c_str(name), C_NULL_PTR, n ) /= 0 )
        endif
    else
        write( error_unit, '(A)' ) '(*error*) zdf_add_quant_part_file: np is larger than data size'
    endif
    call zdf_set_err( ok, 'zdf_add_quant_part_file', ierr )

end subroutine zdf_add_quant_part_file

!------------------------------------------------------------------------------
! ******************************** Utilities **********************************
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
!> Returns storage size in bytes for each datatype
!! @param   zdf_type
!! @return  Storage size in bytes
function zdf_sizeof( data_type ) result(sizeof)

    integer, intent(in) :: data_type
    integer :: sizeof

    select case ( data_type )
    case( zdf_int8, zdf_uint8 )
        sizeof = 1
    case( zdf_int16, zdf_uint16 )
        sizeof = 2
    case( zdf_int32, zdf_uint32, zdf_float32 )
        sizeof = 4
    case( zdf_int64, zdf_uint64, zdf_float64, zdf_complex64 )
        sizeof = 8
    case( zdf_complex128 )
        sizeof = 16
    case default
        sizeof = 0
    end select

end function zdf_sizeof

!------------------------------------------------------------------------------
!> Returns ZDF datatype equivalent to Fortran real kind
!! @param   rkind   Real kind type parameter
!! @return          Equivalent ZDF datatype
function freal_to_zdftype( rkind )

    integer, intent(in) :: rkind
    integer :: freal_to_zdftype

    select case(rkind)
    case(p_single)
        freal_to_zdftype = zdf_float32
    case(p_double)
        freal_to_zdftype = zdf_float64
    case default
        freal_to_zdftype = zdf_null
    end select

end function freal_to_zdftype


end module zdf

!------------------------------------------------------------------------------


#ifdef __ZDF_TEST__

program zdf_test

    use zdf
    use iso_c_binding
    implicit none

    type(t_zdf_grid_info) :: grid
    type(t_zdf_iteration) :: iter
    type(t_zdf_file) :: file
    type(t_zdf_dataset) :: cdset
    type(t_zdf_chunk) :: chunk

    real(kind(1.0e0)), dimension(16,8) :: data
    real(kind(1.0e0)), dimension(8,4), target :: tile

    integer :: i, j

    iter % n = 1
    iter % t = 0.0
    iter % time_units = 'time units'

    do j = 1, size(data,2)
        do i = 1, size(data,1)
            data(i,j) = i + (j-1) * size(data,1)
        enddo
    enddo

    ! Grid
    grid % name  = 'test'
    grid % ndims = 2
    grid % count(1:2) = [16,8]

    grid % label = 'label'
    grid % units = 'units'

    grid % axis(1) % type  = zdf_axis_linear
    grid % axis(1) % min   = -1.0
    grid % axis(1) % max   = +1.0
    grid % axis(1) % label = 'x1'
    grid % axis(1) % units = 'x1 units'

    grid % axis(2) % type  = zdf_axis_linear
    grid % axis(2) % min   = -1.0
    grid % axis(2) % max   = +1.0
    grid % axis(2) % label = 'x2'
    grid % axis(2) % units = 'x2 units'

    ! Save grid all at once
    call zdf_save_grid( data, grid, iter, 'TEST-SAVE' )

    ! Save grid using a chunked dataset
    call zdf_open_grid_file( file, grid, iter, 'TEST-CHUNKED' )

    cdset % name = 'test'
    cdset % data_type = zdf_float32
    cdset % ndims = 2
    cdset % count(1:2) = [16,8]
    call zdf_start_cdset( file, cdset )

    chunk % count(1:2)  = [8,4]
    chunk % stride(1:2) = [1,1]
    chunk % data = c_loc( tile )

    chunk % start(1:2) = [0,0]
    tile = data( 1:8, 1:4 )
    call zdf_write_cdset( file, cdset, chunk )

    chunk % start(1:2) = [8,0]
    tile = data( 9:16, 1:4 )
    call zdf_write_cdset( file, cdset, chunk )

    chunk % start(1:2) = [0,4]
    tile = data( 1:8, 5:8 )
    call zdf_write_cdset( file, cdset, chunk )

    chunk % start(1:2) = [8,4]
    tile = data( 9:16, 5:8 )
    call zdf_write_cdset( file, cdset, chunk )

    call zdf_end_cdset( file, cdset )
    call zdf_close_file( file )

end program zdf_test

#endif
