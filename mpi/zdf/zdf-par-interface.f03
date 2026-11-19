!------------------------------------------------------------------------------
! Fortran 2003 interface routines for parallel (MPI) ZDF
!
! Notes:
! - All routines are collective over the communicator used to open the file, and
!   return the same error code on all ranks.
! - All routines accept an optional `ierr` argument, set to 0 on success and -1 on
!   error. If `ierr` is not present, an error is fatal (error stop).
! - The file handle is an opaque pointer to the C t_zdf_par_file structure, which
!   is allocated / freed by zdf_par_open_file_f03() / zdf_par_close_file_f03() in
!   zdf-parallel-f.c. All other routines call the C library directly.
! - Communicators are Fortran MPI handles (`use mpi`). For mpi_f08 pass comm % MPI_VAL.
! - Data buffers must be in host memory.
!------------------------------------------------------------------------------
module zdf_parallel

use zdf
use iso_c_binding
use iso_fortran_env, only : error_unit

implicit none

integer, private, parameter :: p_double = kind(1.0d0)

!> Use as chunk offset to have the library calculate it
integer(c_int64_t), parameter :: ZDF_GET_OFFSET = -1

enum, bind(c)
    enumerator :: ZDF_MPI, ZDF_INDEPENDENT, ZDF_MPIIO_INDEPENDENT, ZDF_MPIIO_COLLECTIVE
endenum

!> Parallel ZDF file handle
type :: t_zdf_par_file
    !> Opaque pointer to C t_zdf_par_file structure
    type(c_ptr) :: ptr = C_NULL_PTR
    !> Parallel I/O mode (information only)
    integer :: iomode = -1
    !> MPI communicator (information only)
    integer :: comm = -1
end type

!------------------------------------------------------------------------------
! C library interfaces
!------------------------------------------------------------------------------
interface

    function zdf_par_open_file_f03( zdf, filename, amode, comm, iomode ) &
        bind(C, name="zdf_par_open_file_f03")
        import
        integer(c_int) :: zdf_par_open_file_f03
        type(c_ptr) :: zdf
        character(kind=c_char), dimension(*), intent(in) :: filename
        integer(c_int), value :: amode
        ! Fortran MPI handle (MPI_Fint, which is a C int unless MPI was built with
        ! 8 byte Fortran integers)
        integer(c_int), value :: comm
        integer(c_int), value :: iomode
    end function

    function zdf_par_close_file_f03( zdf ) bind(C, name="zdf_par_close_file_f03")
        import
        integer(c_int) :: zdf_par_close_file_f03
        type(c_ptr) :: zdf
    end function

    function zdf_par_add_string_c( zdf, name, str ) bind(C, name="zdf_par_add_string")
        import
        integer(c_int) :: zdf_par_add_string_c
        type(c_ptr), value :: zdf
        character(kind=c_char), dimension(*), intent(in) :: name, str
    end function

    function zdf_par_add_int32_c( zdf, name, val ) bind(C, name="zdf_par_add_int32")
        import
        integer(c_int) :: zdf_par_add_int32_c
        type(c_ptr), value :: zdf
        character(kind=c_char), dimension(*), intent(in) :: name
        integer(c_int32_t), value :: val
    end function

    function zdf_par_add_double_c( zdf, name, val ) bind(C, name="zdf_par_add_double")
        import
        integer(c_int) :: zdf_par_add_double_c
        type(c_ptr), value :: zdf
        character(kind=c_char), dimension(*), intent(in) :: name
        real(c_double), value :: val
    end function

    function zdf_par_add_iteration_c( zdf, iter ) bind(C, name="zdf_par_add_iteration")
        import
        integer(c_int) :: zdf_par_add_iteration_c
        type(c_ptr), value :: zdf
        type(t_zdf_iteration_c), intent(in) :: iter
    end function

    function zdf_par_add_grid_info_c( zdf, info ) bind(C, name="zdf_par_add_grid_info")
        import
        integer(c_int) :: zdf_par_add_grid_info_c
        type(c_ptr), value :: zdf
        type(t_zdf_grid_info_c), intent(in) :: info
    end function

    function zdf_par_add_part_info_c( zdf, info ) bind(C, name="zdf_par_add_part_info")
        import
        integer(c_int) :: zdf_par_add_part_info_c
        type(c_ptr), value :: zdf
        type(t_zdf_part_info_c), intent(in) :: info
    end function

    function zdf_par_start_cdset_c( zdf, dataset ) bind(C, name="zdf_par_start_cdset")
        import
        integer(c_int) :: zdf_par_start_cdset_c
        type(c_ptr), value :: zdf
        type(t_zdf_dataset_c) :: dataset
    end function

    function zdf_par_write_cdset_c( zdf, dataset, chunk, offset ) bind(C, name="zdf_par_write_cdset")
        import
        integer(c_int) :: zdf_par_write_cdset_c
        type(c_ptr), value :: zdf
        type(t_zdf_dataset_c), intent(in) :: dataset
        type(t_zdf_chunk), intent(in) :: chunk
        integer(c_int64_t), value :: offset
    end function

    function zdf_par_end_cdset_c( zdf, dataset ) bind(C, name="zdf_par_end_cdset")
        import
        integer(c_int) :: zdf_par_end_cdset_c
        type(c_ptr), value :: zdf
        type(t_zdf_dataset_c), intent(in) :: dataset
    end function

    function zdf_par_extend_dataset_c( zdf, dataset, new_count ) bind(C, name="zdf_par_extend_dataset")
        import
        integer(c_int) :: zdf_par_extend_dataset_c
        type(c_ptr), value :: zdf
        type(t_zdf_dataset_c) :: dataset
        integer(c_int64_t), dimension(*), intent(in) :: new_count
    end function

end interface

private :: zdf_par_check_open

!------------------------------------------------------------------------------
! Fortran interfaces (extend the generic interfaces of the zdf module)
!------------------------------------------------------------------------------

interface zdf_mpi_type
    module procedure zdf_mpi_type
end interface

interface zdf_iomode_name
    module procedure zdf_iomode_name
end interface

interface zdf_open_file
    module procedure zdf_par_open_file
end interface

interface zdf_close_file
    module procedure zdf_par_close_file
end interface

interface zdf_add
    module procedure zdf_par_add_string
    module procedure zdf_par_add_int32
    module procedure zdf_par_add_double
    module procedure zdf_par_add_iteration
    module procedure zdf_par_add_grid_info
    module procedure zdf_par_add_part_info
end interface

interface zdf_start_cdset
    module procedure zdf_par_start_cdset
end interface

interface zdf_write_cdset
    module procedure zdf_par_write_cdset
end interface

interface zdf_end_cdset
    module procedure zdf_par_end_cdset
end interface

interface zdf_extend_cdset
    module procedure zdf_par_extend_cdset
end interface

contains

!------------------------------------------------------------------------------
!> Checks that the parallel file is open. All ranks hold a handle if the file
!! was opened successfully, so this check is consistent across ranks.
function zdf_par_check_open( file, routine )

    type(t_zdf_par_file), intent(in) :: file
    character(len=*), intent(in) :: routine
    logical :: zdf_par_check_open

    zdf_par_check_open = c_associated( file % ptr )
    if ( .not. zdf_par_check_open ) then
        write( error_unit, '(A,A,A)' ) '(*error*) ', routine, ': parallel ZDF file is not open'
    endif

end function zdf_par_check_open

!------------------------------------------------------------------------------
!> Opens parallel ZDF file
!! @param   file    Parallel ZDF file
!! @param   name    File name
!! @param   amode   Access mode (zdf_file_create, zdf_file_read or zdf_file_update)
!! @param   comm    MPI communicator (Fortran handle)
!! @param   iomode  Parallel I/O mode (ZDF_MPI, ZDF_INDEPENDENT, ZDF_MPIIO_INDEPENDENT
!!                  or ZDF_MPIIO_COLLECTIVE)
!! @param   ierr    (optional) Error code
subroutine zdf_par_open_file( file, name, amode, comm, iomode, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    character(len=*), intent(in) :: name
    integer, intent(in) :: amode
    integer, intent(in) :: comm
    integer, intent(in) :: iomode
    integer, intent(out), optional :: ierr

    logical :: ok

    ok = ( zdf_par_open_file_f03( file % ptr, zdf_c_str(name), int( amode, c_int ), &
                                  int( comm, c_int ), int( iomode, c_int ) ) /= 0 )
    if ( ok ) then
        file % iomode = iomode
        file % comm   = comm
    endif
    call zdf_set_err( ok, 'zdf_par_open_file', ierr )

end subroutine zdf_par_open_file

!------------------------------------------------------------------------------
!> Closes parallel ZDF file
!! @param   file    Parallel ZDF file
!! @param   ierr    (optional) Error code
subroutine zdf_par_close_file( file, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    integer, intent(out), optional :: ierr

    logical :: ok

    ok = zdf_par_check_open( file, 'zdf_par_close_file' )
    if ( ok ) ok = ( zdf_par_close_file_f03( file % ptr ) /= 0 )
    file % iomode = -1
    file % comm   = -1
    call zdf_set_err( ok, 'zdf_par_close_file', ierr )

end subroutine zdf_par_close_file

!------------------------------------------------------------------------------
!> Adds string element to parallel ZDF file
subroutine zdf_par_add_string( file, name, str, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    character(len = *), intent(in) :: name
    character(len = *), intent(in) :: str
    integer, intent(out), optional :: ierr

    logical :: ok

    ok = zdf_par_check_open( file, 'zdf_par_add_string' )
    if ( ok ) ok = ( zdf_par_add_string_c( file % ptr, zdf_c_str(name), zdf_c_str(str) ) /= 0 )
    call zdf_set_err( ok, 'zdf_par_add_string', ierr )

end subroutine zdf_par_add_string

!------------------------------------------------------------------------------
!> Adds int32 element to parallel ZDF file
subroutine zdf_par_add_int32( file, name, value, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    character(len = *), intent(in) :: name
    integer, intent(in) :: value
    integer, intent(out), optional :: ierr

    logical :: ok

    ok = zdf_par_check_open( file, 'zdf_par_add_int32' )
    if ( ok ) ok = ( zdf_par_add_int32_c( file % ptr, zdf_c_str(name), int( value, c_int32_t ) ) /= 0 )
    call zdf_set_err( ok, 'zdf_par_add_int32', ierr )

end subroutine zdf_par_add_int32

!------------------------------------------------------------------------------
!> Adds float64 element to parallel ZDF file
subroutine zdf_par_add_double( file, name, value, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    character(len = *), intent(in) :: name
    real(p_double), intent(in) :: value
    integer, intent(out), optional :: ierr

    logical :: ok

    ok = zdf_par_check_open( file, 'zdf_par_add_double' )
    if ( ok ) ok = ( zdf_par_add_double_c( file % ptr, zdf_c_str(name), real( value, c_double ) ) /= 0 )
    call zdf_set_err( ok, 'zdf_par_add_double', ierr )

end subroutine zdf_par_add_double

!------------------------------------------------------------------------------
!> Adds iteration metadata to parallel ZDF file
subroutine zdf_par_add_iteration( file, iter, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    type(t_zdf_iteration), intent(in) :: iter
    integer, intent(out), optional :: ierr

    type(t_zdf_iteration_c) :: iter_c
    type(t_zdf_iter_buf), target :: buf
    logical :: ok

    ok = zdf_par_check_open( file, 'zdf_par_add_iteration' )
    if ( ok ) then
        call zdf_iter_to_c( iter, iter_c, buf )
        ok = ( zdf_par_add_iteration_c( file % ptr, iter_c ) /= 0 )
    endif
    call zdf_set_err( ok, 'zdf_par_add_iteration', ierr )

end subroutine zdf_par_add_iteration

!------------------------------------------------------------------------------
!> Adds grid information metadata to parallel ZDF file
subroutine zdf_par_add_grid_info( file, info, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    type(t_zdf_grid_info), intent(in) :: info
    integer, intent(out), optional :: ierr

    type(t_zdf_grid_info_c) :: info_c
    type(t_zdf_grid_buf), target :: buf
    logical :: ok

    ! zdf_grid_to_c only depends on info, so the result is the same on all ranks
    ok = zdf_par_check_open( file, 'zdf_par_add_grid_info' )
    if ( ok ) call zdf_grid_to_c( info, info_c, buf, ok )
    if ( ok ) ok = ( zdf_par_add_grid_info_c( file % ptr, info_c ) /= 0 )
    call zdf_set_err( ok, 'zdf_par_add_grid_info', ierr )

end subroutine zdf_par_add_grid_info

!------------------------------------------------------------------------------
!> Adds particle information metadata to parallel ZDF file
subroutine zdf_par_add_part_info( file, info, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    type(t_zdf_part_info), intent(in) :: info
    integer, intent(out), optional :: ierr

    type(t_zdf_part_info_c) :: info_c
    type(t_zdf_part_buf), target :: buf
    logical :: ok

    ok = zdf_par_check_open( file, 'zdf_par_add_part_info' )
    if ( ok ) call zdf_part_to_c( info, info_c, buf, ok )
    if ( ok ) ok = ( zdf_par_add_part_info_c( file % ptr, info_c ) /= 0 )
    call zdf_set_err( ok, 'zdf_par_add_part_info', ierr )

end subroutine zdf_par_add_part_info

!------------------------------------------------------------------------------
!> Start ZDF chunked dataset for parallel I/O. The dataset id and file offset
!! are updated.
!! @param   file        ZDF parallel I/O file
!! @param   dataset     ZDF chunked dataset
!! @param   ierr        (optional) Error code
subroutine zdf_par_start_cdset( file, dataset, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    type(t_zdf_dataset), intent(inout)  :: dataset
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: dataset_c
    type(t_zdf_dataset_buf), target :: buf
    logical :: ok

    ok = zdf_par_check_open( file, 'zdf_par_start_cdset' )
    if ( ok ) then
        call zdf_dataset_to_c( dataset, dataset_c, buf )
        ok = ( zdf_par_start_cdset_c( file % ptr, dataset_c ) /= 0 )
        call zdf_dataset_from_c( dataset_c, dataset )
    endif
    call zdf_set_err( ok, 'zdf_par_start_cdset', ierr )

end subroutine zdf_par_start_cdset

!------------------------------------------------------------------------------
!> Write ZDF chunked dataset using parallel I/O
!! @param   file        ZDF parallel I/O file
!! @param   cdset       ZDF chunked dataset (global information)
!! @param   chunk       Local chunk
!! @param   offset      (optional) Offset of local chunk in file layout, in elements.
!!                      Defaults to ZDF_GET_OFFSET (calculated by the library).
!! @param   ierr        (optional) Error code
subroutine zdf_par_write_cdset( file, cdset, chunk, offset, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    type(t_zdf_dataset), intent(in)  :: cdset
    type(t_zdf_chunk), intent(in) :: chunk
    integer(c_int64_t), intent(in), optional :: offset
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: cdset_c
    type(t_zdf_dataset_buf), target :: buf
    integer(c_int64_t) :: off
    logical :: ok

    off = ZDF_GET_OFFSET
    if ( present( offset ) ) off = offset

    ok = zdf_par_check_open( file, 'zdf_par_write_cdset' )
    if ( ok ) then
        call zdf_dataset_to_c( cdset, cdset_c, buf )
        ok = ( zdf_par_write_cdset_c( file % ptr, cdset_c, chunk, off ) /= 0 )
    endif
    call zdf_set_err( ok, 'zdf_par_write_cdset', ierr )

end subroutine zdf_par_write_cdset

!------------------------------------------------------------------------------
!> Ends access to chunked dataset (using parallel I/O)
!!
!! @param   file    ZDF parallel I/O file
!! @param   cdset   ZDF chunked dataset
!! @param   ierr    (optional) Error code
subroutine zdf_par_end_cdset( file, cdset, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    type(t_zdf_dataset), intent(in)  :: cdset
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: cdset_c
    type(t_zdf_dataset_buf), target :: buf
    logical :: ok

    ok = zdf_par_check_open( file, 'zdf_par_end_cdset' )
    if ( ok ) then
        call zdf_dataset_to_c( cdset, cdset_c, buf )
        ok = ( zdf_par_end_cdset_c( file % ptr, cdset_c ) /= 0 )
    endif
    call zdf_set_err( ok, 'zdf_par_end_cdset', ierr )

end subroutine zdf_par_end_cdset

!------------------------------------------------------------------------------
!> Extends chunked dataset dimensions (using parallel I/O)
!!
!! @param   file        ZDF parallel I/O file
!! @param   cdset       ZDF chunked dataset
!! @param   new_count   New dataset dimensions
!! @param   ierr        (optional) Error code
subroutine zdf_par_extend_cdset( file, cdset, new_count, ierr )

    type(t_zdf_par_file), intent(inout) :: file
    type(t_zdf_dataset), intent(inout)  :: cdset
    integer(c_int64_t), dimension(:), intent(in) :: new_count
    integer, intent(out), optional :: ierr

    type(t_zdf_dataset_c) :: cdset_c
    type(t_zdf_dataset_buf), target :: buf
    integer(c_int64_t), dimension(zdf_max_dims) :: count
    logical :: ok

    ok = zdf_par_check_open( file, 'zdf_par_extend_cdset' )
    if ( ok ) then
        ok = ( cdset % ndims >= 1 .and. cdset % ndims <= zdf_max_dims .and. &
               size( new_count ) >= cdset % ndims )
        if ( .not. ok ) write( error_unit, '(A)' ) '(*error*) zdf_par_extend_cdset: invalid dimensions'
    endif
    if ( ok ) then
        count = cdset % count
        count( 1:cdset % ndims ) = new_count( 1:cdset % ndims )
        call zdf_dataset_to_c( cdset, cdset_c, buf )
        ok = ( zdf_par_extend_dataset_c( file % ptr, cdset_c, count ) /= 0 )
        call zdf_dataset_from_c( cdset_c, cdset )
    endif
    call zdf_set_err( ok, 'zdf_par_extend_cdset', ierr )

end subroutine zdf_par_extend_cdset


!------------------------------------------------------------------------------
!> Gets name of iomode from integer descriptor
!!
!! @param   iomode  Integer descriptor of I/O mode
!! @return          String with I/O mode name
function zdf_iomode_name( iomode )

    integer, intent(in) :: iomode
    character(len=18) :: zdf_iomode_name

    select case(iomode)
    case (ZDF_MPI)
        zdf_iomode_name = "mpi"
    case (ZDF_INDEPENDENT)
        zdf_iomode_name = "independent"
    case (ZDF_MPIIO_INDEPENDENT)
        zdf_iomode_name = "mpi-io independent"
    case (ZDF_MPIIO_COLLECTIVE)
        zdf_iomode_name = "mpi-io collective"
    case default
        zdf_iomode_name = "invalid"
    end select

end function zdf_iomode_name

!------------------------------------------------------------------------------
!> Gets MPI datatype equivalent to ZDF datatype
!!
!! @param   data_type   ZDF datatype id
!! @return              MPI datatype id
function zdf_mpi_type( data_type )

    use mpi

    integer, intent(in) :: data_type
    integer :: zdf_mpi_type

    select case ( data_type )
    case( zdf_int8, zdf_uint8 )
        zdf_mpi_type = MPI_BYTE
    case( zdf_int16, zdf_uint16 )
        zdf_mpi_type = MPI_INTEGER2
    case( zdf_int32, zdf_uint32 )
        zdf_mpi_type = MPI_INTEGER4
    case( zdf_float32 )
        zdf_mpi_type = MPI_REAL
    case( zdf_int64, zdf_uint64 )
        zdf_mpi_type = MPI_INTEGER8
    case( zdf_float64 )
        zdf_mpi_type = MPI_DOUBLE_PRECISION
    case( zdf_complex64 )
        zdf_mpi_type = MPI_COMPLEX
    case( zdf_complex128 )
        zdf_mpi_type = MPI_DOUBLE_COMPLEX
    case default
        zdf_mpi_type = MPI_DATATYPE_NULL
    end select

end function zdf_mpi_type

end module zdf_parallel
