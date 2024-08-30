! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example (4th-order diffusion)
! ******************************************************

! Driver for apply_diffusion() that sets up fields and does timings
program main
    use m_utils, only: timer_start, timer_end, timer_get, is_master, num_rank, write_field_to_file
    use mpi, only: MPI_INIT, MPI_FINALIZE
    implicit none

    ! constants
    integer, parameter :: wp = 4
    
    ! local variables
    integer :: nx, ny, nz, num_iter
    logical :: scan

    integer :: num_halo = 2
    real (kind=wp) :: alpha = 1.0_wp / 32.0_wp

    real (kind=wp), allocatable :: in_field(:, :, :)
    real (kind=wp), allocatable :: out_field(:, :, :)

    integer :: timer_work
    real (kind=8) :: runtime

    integer :: cur_setup, num_setups = 1
    integer :: nx_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)
    integer :: ny_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)

#ifdef CRAYPAT
    include "pat_apif.h"
    integer :: istat
    call PAT_record(PAT_STATE_OFF, istat)
#endif

    call init()

    if (is_master()) then
        write(*, '(a)') '# ranks nx ny nz num_iter time'
        write(*, '(a)') 'data = np.array( [ \'
    end if

    if (scan) num_setups = size(nx_setups) * size(ny_setups)
    do cur_setup = 0, num_setups - 1

        if (scan) then
            nx = nx_setups(mod(cur_setup, size(ny_setups)) + 1)
            ny = ny_setups(cur_setup / size(ny_setups) + 1)
        end if

        call setup(nx, ny, nz, num_halo, in_field, out_field)

        if (.not. scan .and. is_master()) then
            call write_field_to_file(in_field, num_halo, "in_field.dat")
        end if

        ! warmup caches
        call apply_diffusion(in_field, out_field, alpha, num_iter=1)

        ! time the actual work
#ifdef CRAYPAT
        call PAT_record(PAT_STATE_ON, istat)
#endif
        timer_work = -999
        call timer_start('work', timer_work)

        call apply_diffusion(in_field, out_field, alpha, num_iter=num_iter)
        
        call timer_end(timer_work)
#ifdef CRAYPAT
        call PAT_record(PAT_STATE_OFF, istat)
#endif

        if (.not. scan .and. is_master()) then
            call write_field_to_file(out_field, num_halo, "out_field.dat")
        end if

        call cleanup(in_field, out_field)

        runtime = timer_get(timer_work)
        if (is_master()) then
            write(*, '(a, i5, a, i5, a, i5, a, i5, a, i8, a, e15.7, a)') &
                '[', num_rank(), ',', nx, ',', ny, ',', nz, ',', num_iter, ',', runtime, '], \'
        end if

    end do

    if (is_master()) then
        write(*, '(a)') '] )'
    end if

    call finalize()

contains

    ! Integrate 4th-order diffusion equation by a certain number of iterations.
    subroutine apply_diffusion(in_field, out_field, alpha, num_iter)
        implicit none
        
        ! arguments
        real (kind=wp), intent(inout) :: in_field(:, :, :)
        real (kind=wp), intent(inout) :: out_field(:, :, :)
        real (kind=wp), intent(in) :: alpha
        integer, intent(in) :: num_iter
        
        ! local variables
        real (kind=wp), save, allocatable :: tmp1_field(:, :, :)
        real (kind=wp), save, allocatable :: tmp2_field(:, :, :)
        integer :: iter, i, j, k
        
        ! this is only done the first time this subroutine is called (warmup)
        ! or when the dimensions of the fields change
        if (allocated(tmp1_field) .and. &
            any(shape(tmp1_field) /= (/size(in_field,1), size(in_field,2), size(in_field,3) /) ) then
            deallocate(tmp1_field, tmp2_field)
        end if
        if (.not. allocated(tmp1_field)) then
            allocate(tmp1_field(size(in_field,1), size(in_field,2), size(in_field,3)))
            allocate(tmp2_field(size(in_field,1), size(in_field,2), size(in_field,3)))
            tmp1_field = 0.0_wp
            tmp2_field = 0.0_wp
        end if
        
        do iter = 1, num_iter

            call update_halo(in_field, num_halo, size(in_field,1), size(in_field,2), size(in_field,3))
            
            call laplacian(in_field, tmp1_field, num_halo, extend=1)
            call laplacian(tmp1_field, tmp2_field, num_halo, extend=0)
            
            ! do forward in time step
            do k = 1, size(in_field,3)
            do j = 1 + num_halo, size(in_field,2) + num_halo
            do i = 1 + num_halo, size(in_field,1) + num_halo
                out_field(i, j, k) = in_field(i, j, k) - alpha * tmp2_field(i, j, k)
            end do
            end do
            end do

            ! copy out to in in case this is not the last iteration
            if (iter /= num_iter) then
                do k = 1, size(in_field,3)
                do j = 1 + num_halo, size(in_field,2) + num_halo
                do i = 1 + num_halo, size(in_field,1) + num_halo
                    in_field(i, j, k) = out_field(i, j, k)
                end do
                end do
                end do
            end if

        end do

        call update_halo(out_field, num_halo, size(out_field,1), size(out_field,2), size(out_field,3))
            
    end subroutine apply_diffusion

    ! Compute Laplacian using 2nd-order centered differences.
    subroutine laplacian(field, lap, num_halo, extend)
        implicit none
            
        ! arguments
        real (kind=wp), intent(in) :: field(:, :, :)
        real (kind=wp), intent(inout) :: lap(:, :, :)
        integer, intent(in) :: num_halo, extend
        
        ! local variables
        integer :: i, j, k
            
        do k = 1, size(field,3)
        do j = 1 + num_halo - extend, size(field,2) + num_halo + extend
        do i = 1 + num_halo - extend, size(field,1) + num_halo + extend
            lap(i, j, k) = -4._wp * field(i, j, k)      &
                + field(i - 1, j, k) + field(i + 1, j, k)  &
                + field(i, j - 1, k) + field(i, j + 1, k)
        end do
        end do
        end do

    end subroutine laplacian

    ! Update the halo-zone using an up/down and left/right strategy.
    subroutine update_halo(field, num_halo, nx, ny, nz)
        implicit none
            
        ! arguments
        real (kind=wp), intent(inout) :: field(:, :, :)
        integer, intent(in) :: num_halo, nx, ny, nz
        
        ! local variables
        integer :: i, j, k
            
        ! bottom edge (without corners)
        do k = 1, nz
        do j = 1, num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j + ny, k)
        end do
        end do
        end do
            
        ! top edge (without corners)
        do k = 1, nz
        do j = ny + num_halo + 1, ny + 2 * num_halo
        do i = 1 + num_halo, nx + num_halo
            field(i, j, k) = field(i, j - ny, k)
        end do
        end do
        end do
        
        ! left edge (including corners)
        do k = 1, nz
        do j = 1, ny + 2 * num_halo
        do i = 1, num_halo
            field(i, j, k) = field(i + nx, j, k)
        end do
        end do
        end do
                
        ! right edge (including corners)
        do k = 1, nz
        do j = 1, ny + 2 * num_halo
        do i = nx + num_halo + 1, nx + 2 * num_halo
            field(i, j, k) = field(i - nx, j, k)
        end do
        end do
        end do
        
    end subroutine update_halo

    ! initialize at program start
    subroutine init()
        implicit none
        integer :: ierror

        ! initialize MPI environment
        call MPI_INIT(ierror)
        call error(ierror /= 0, 'Problem with MPI_INIT', code=ierror)

        call read_cmd_line_arguments()

    end subroutine init

    ! setup everything before work
    subroutine setup(nx, ny, nz, num_halo, in_field, out_field)
        use m_utils, only : timer_init
        implicit none

        ! arguments
        integer, intent(in) :: nx, ny, nz, num_halo
        real (kind=wp), intent(out) :: in_field(nx + 2 * num_halo, ny + 2 * num_halo, nz)
        real (kind=wp), intent(out) :: out_field(nx + 2 * num_halo, ny + 2 * num_halo, nz)
        
        ! local variables
        integer :: i, j, k

        call timer_init()

        allocate(in_field(nx + 2 * num_halo, ny + 2 * num_halo, nz))
        in_field = 0.0_wp
        do k = 1 + nz / 4, 3 * nz / 4
        do j = 1 + num_halo + ny / 4, num_halo + 3 * ny / 4
        do i = 1 + num_halo + nx / 4, num_halo + 3 * nx / 4
            in_field(i, j, k) = 1.0_wp
        end do
        end do
        end do

        allocate(out_field(nx + 2 * num_halo, ny + 2 * num_halo, nz))
        out_field = in_field

    end subroutine setup

    ! read and parse the command line arguments
    subroutine read_cmd_line_arguments()
        use m_utils, only : error
        implicit none

        ! local variables
        integer iarg, num_arg
        character(len=256) :: arg, arg_val

        ! setup defaults
        nx = -1
        ny = -1
        nz = -1
        num_iter = -1
        scan = .false.

        num_arg = command_argument_count()
        iarg = 1
        do while (iarg <= num_arg)
            call get_command_argument(iarg, arg)
            select case (arg)
            case ("--nx")
                call error(iarg + 1 > num_arg, "Missing value for --nx argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --nx argument")
                read(arg_val, *) nx
                iarg = iarg + 1
            case ("--ny")
                call error(iarg + 1 > num_arg, "Missing value for --ny argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --ny argument")
                read(arg_val, *) ny
                iarg = iarg + 1
            case ("--nz")
                call error(iarg + 1 > num_arg, "Missing value for --nz argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --nz argument")
                read(arg_val, *) nz
                iarg = iarg + 1
            case ("--num_iter")
                call error(iarg + 1 > num_arg, "Missing value for --num_iter argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --num_iter argument")
                read(arg_val, *) num_iter
                iarg = iarg + 1
            case ("--scan")
                scan = .true.
            case default
                call error(.true., "Unknown command line argument encountered: " // trim(arg))
            end select
            iarg = iarg + 1
        end do

        ! make sure everything is set
        if (.not. scan) then
            call error(nx == -1, 'You have to specify nx')
            call error(ny == -1, 'You have to specify ny')
        end if
        call error(nz == -1, 'You have to specify nz')
        call error(num_iter == -1, 'You have to specify num_iter')

        ! check consistency of values
        if (.not. scan) then
            call error(nx < 0 .or. nx > 1024*1024, "Please provide a reasonable value of nx")
            call error(ny < 0 .or. ny > 1024*1024, "Please provide a reasonable value of ny")
        end if
        call error(nz < 0 .or. nz > 1024, "Please provide a reasonable value of nz")
        call error(num_iter < 1 .or. num_iter > 1024*1024, "Please provide a reasonable value of num_iter")

    end subroutine read_cmd_line_arguments

    ! cleanup at end of work
    subroutine cleanup(in_field, out_field)
        implicit none
        
        ! arguments
        real (kind=wp), intent(inout) :: in_field(:, :, :)
        real (kind=wp), intent(inout) :: out_field(:, :, :)
        
        deallocate(in_field, out_field)

    end subroutine cleanup

    ! finalize at end of program
    subroutine finalize()
        implicit none
        integer :: ierror

        call MPI_FINALIZE(ierror)
        call error(ierror /= 0, 'Problem with MPI_FINALIZE', code=ierror)

    end subroutine finalize

end program main
