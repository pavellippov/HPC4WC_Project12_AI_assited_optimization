! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example (4th-order diffusion)
! ******************************************************

! Driver for apply_diffusion() that sets up fields and does timings
    program main
        use mpi
        use m_utils, only: timer_start, timer_end, timer_get, is_master, write_field_to_file
        implicit none

        ! constants
        integer, parameter :: wp = 4

        ! local
        integer :: nx, ny, nz, num_iter
        integer :: num_halo = 2
        real(kind=wp) :: alpha = 1.0_wp / 32.0_wp

        real(kind=wp), allocatable :: in_field(:, :, :)
        real(kind=wp), allocatable :: out_field(:, :, :)

        integer :: timer_work
        real(kind=8) :: runtime

        integer :: cur_setup, num_setups = 1
        integer :: nx_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)
        integer :: ny_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)

        ! MPI variables
        integer :: ierror, rank, size, nx_local, ny_local
        integer :: x_offset, y_offset
        integer :: mpi_cart, dims(2), periods(2), coords(2), reorder

        call MPI_INIT(ierror)
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)

        ! 2D Cartesian grid of processes
        dims(1) = 0
        dims(2) = 0
        periods(1) = .false.
        periods(2) = .false.
        reorder = .true.

        call MPI_DIMS_CREATE(size, 2, dims, ierror)
        call MPI_CART_CREATE(MPI_COMM_WORLD, 2, dims, periods, reorder, mpi_cart, ierror)
        call MPI_CART_COORDS(mpi_cart, rank, 2, coords, ierror)

        ! Set up local domain sizes
        nx_local = nx / dims(1)
        ny_local = ny / dims(2)

        x_offset = coords(1) * nx_local
        y_offset = coords(2) * ny_local

        if (rank == 0) then
            write(*, '(a)') '# ranks nx ny nz num_iter time'
            write(*, '(a)') 'data = np.array( [ \'
        end if

        ! Iterate over setups
        do cur_setup = 0, num_setups - 1
            if (scan) then
                nx = nx_setups(modulo(cur_setup, size(ny_setups)) + 1)
                ny = ny_setups(cur_setup / size(ny_setups) + 1)
            end if

            call setup(nx_local, ny_local, nz)

            ! Warmup caches
            call apply_diffusion(in_field, out_field, alpha, num_iter=1)

            ! Time the actual work
            timer_work = -999
            call timer_start('work', timer_work)

            call apply_diffusion(in_field, out_field, alpha, num_iter=num_iter)

            call timer_end(timer_work)

            call cleanup()

            runtime = timer_get(timer_work)
            if (rank == 0) then
                write(*, '(a, i5, a, i5, a, i5, a, i5, a, i8, a, e15.7, a)') &
                    '[', size, ',', nx, ',', ny, ',', nz, ',', num_iter, ',', runtime, '], \'
            end if
        end do

        if (rank == 0) then
            write(*, '(a)') '] )'
        end if

        call finalize()

    contains

        ! Apply diffusion operation
        subroutine apply_diffusion(in_field, out_field, alpha, num_iter)
            implicit none
            real(kind=wp), intent(inout) :: in_field(:, :, :)
            real(kind=wp), intent(inout) :: out_field(:, :, :)
            real(kind=wp), intent(in) :: alpha
            integer, intent(in) :: num_iter
            real(kind=wp), allocatable :: tmp1_field(:, :, :)
            real(kind=wp), allocatable :: tmp2_field(:, :, :)
            integer :: iter, i, j, k
            integer :: mpi_request(4), mpi_status(4)
            integer :: north, south, east, west

            ! Obtain neighboring ranks
            call MPI_CART_SHIFT(mpi_cart, 0, 1, west, east, ierror)
            call MPI_CART_SHIFT(mpi_cart, 1, 1, north, south, ierror)

            ! Allocate temp fields
            if (.not. allocated(tmp1_field)) then
                allocate(tmp1_field(nx_local + 2 * num_halo, ny_local + 2 * num_halo, nz))
                allocate(tmp2_field(nx_local + 2 * num_halo, ny_local + 2 * num_halo, nz))
                tmp1_field = 0.0_wp
                tmp2_field = 0.0_wp
            end if

            ! Iterative diffusion steps
            do iter = 1, num_iter
                call update_halo(in_field, north, south, east, west)

                call laplacian(in_field, tmp1_field, num_halo, extend=1)
                call laplacian(tmp1_field, tmp2_field, num_halo, extend=0)

                ! Time-stepping
                do k = 1, nz
                    do j = 1 + num_halo, ny_local + num_halo
                        do i = 1 + num_halo, nx_local + num_halo
                            out_field(i, j, k) = in_field(i, j, k) - alpha * tmp2_field(i, j, k)
                        end do
                    end do
                end do

                if (iter /= num_iter) then
                    in_field = out_field
                end if
            end do

            call update_halo(out_field, north, south, east, west)

        end subroutine apply_diffusion

        ! Halo exchange between neighboring ranks
        subroutine update_halo(field, north, south, east, west)
            implicit none
            real(kind=wp), intent(inout) :: field(:, :, :)
            integer, intent(in) :: north, south, east, west
            integer :: ierror, mpi_status(4)
            real(kind=wp) :: north_buf(nx_local), south_buf(nx_local)
            real(kind=wp) :: east_buf(ny_local), west_buf(ny_local)

            ! Communicate with north and south neighbors
            call MPI_SENDRECV(field(1+num_halo, num_halo+1, :), nx_local, MPI_REAL, north, 1, &
                              south_buf, nx_local, MPI_REAL, south, 1, mpi_cart, mpi_status(1), ierror)
            call MPI_SENDRECV(field(1+num_halo, num_halo+ny_local, :), nx_local, MPI_REAL, south, 2, &
                              north_buf, nx_local, MPI_REAL, north, 2, mpi_cart, mpi_status(2), ierror)

            ! Communicate with east and west neighbors
            call MPI_SENDRECV(field(num_halo+1, 1+num_halo, :), ny_local, MPI_REAL, west, 3, &
                              east_buf, ny_local, MPI_REAL, east, 3, mpi_cart, mpi_status(3), ierror)
            call MPI_SENDRECV(field(num_halo+nx_local, 1+num_halo, :), ny_local, MPI_REAL, east, 4, &
                              west_buf, ny_local, MPI_REAL, west, 4, mpi_cart, mpi_status(4), ierror)
        end subroutine update_halo
        

    ! initialize at program start
    ! (init MPI, read command line arguments)
    subroutine init()
        use mpi, only : MPI_INIT
        use m_utils, only : error
        implicit none

        ! local
        integer :: ierror

        ! initialize MPI environment
        call MPI_INIT(ierror)
        call error(ierror /= 0, 'Problem with MPI_INIT', code=ierror)

        call read_cmd_line_arguments()

    end subroutine init


    ! setup everything before work
    ! (init timers, allocate memory, initialize fields)
    subroutine setup()
        use m_utils, only : timer_init
        implicit none

        ! local
        integer :: i, j, k

        call timer_init()

        allocate( in_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
        in_field = 0.0_wp
        do k = 1 + nz / 4, 3 * nz / 4
        do j = 1 + num_halo + ny / 4, num_halo + 3 * ny / 4
        do i = 1 + num_halo + nx / 4, num_halo + 3 * nx / 4
            in_field(i, j, k) = 1.0_wp
        end do
        end do
        end do

        allocate( out_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
        out_field = in_field

    end subroutine setup


    ! read and parse the command line arguments
    ! (read values, convert type, ensure all required arguments are present,
    !  ensure values are reasonable)
    subroutine read_cmd_line_arguments()
        use m_utils, only : error
        implicit none

        ! local
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
        do while ( iarg <= num_arg )
            call get_command_argument(iarg, arg)
            select case (arg)
            case ("--nx")
                call error(iarg + 1 > num_arg, "Missing value for -nx argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -nx argument")
                read(arg_val, *) nx
                iarg = iarg + 1
            case ("--ny")
                call error(iarg + 1 > num_arg, "Missing value for -ny argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -ny argument")
                read(arg_val, *) ny
                iarg = iarg + 1
            case ("--nz")
                call error(iarg + 1 > num_arg, "Missing value for -nz argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -nz argument")
                read(arg_val, *) nz
                iarg = iarg + 1
            case ("--num_iter")
                call error(iarg + 1 > num_arg, "Missing value for -num_iter argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for -num_iter argument")
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
    ! (report timers, free memory)
    subroutine cleanup()
        implicit none
        
        deallocate(in_field, out_field)

    end subroutine cleanup


    ! finalize at end of program
    ! (finalize MPI)
    subroutine finalize()
        use mpi, only : MPI_FINALIZE
        use m_utils, only : error
        implicit none

        integer :: ierror

        call MPI_FINALIZE(ierror)
        call error(ierror /= 0, 'Problem with MPI_FINALIZE', code=ierror)

    end subroutine finalize


end program main
