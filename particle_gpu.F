module particle_struct
    use iso_c_binding
    implicit none

    type, bind(c) :: gpu_particle
        integer(c_int) :: pidx, procidx
        real(c_double) :: vp(3), xp(3), uf(3), xrhs(3), vrhs(3)
        real(c_double) :: Tp, Tprhs_s, Tprhs_L, Tf, radius
        real(c_double) :: radrhs, qinf, qstar
    end type gpu_particle
end module

module particle_c
    use iso_c_binding, only: c_ptr
    implicit none

    type(c_ptr) :: gpu
    integer :: gpu_master_rank

    interface
        integer function gpudevices() bind(c,name="gpudevices")
            use iso_c_binding, only: c_int
            integer(c_int) :: count
        end function

        type(c_ptr) function newgpu(count,h,w,d,z,xl,yl,zl,vis) bind(c,name="NewGPU")
            use iso_c_binding, only: c_ptr, c_int, c_double
            integer(c_int), VALUE, intent(in)      :: count
            integer(c_int), VALUE, intent(in)      :: h
            integer(c_int), VALUE, intent(in)      :: w
            integer(c_int), VALUE, intent(in)      :: d
            integer(c_int), VALUE, intent(in)      :: z
            real(c_double), VALUE, intent(in)      :: xl
            real(c_double), VALUE, intent(in)      :: yl
            real(c_double), VALUE, intent(in)      :: zl
            real(c_double), VALUE, intent(in)      :: vis
        end function

        subroutine gpucopyfield(gpu,uext,vext,wext,text,qext,z,zz) bind(c,name="ParticleFieldSet")
            use iso_c_binding, only: c_ptr, c_double
            type(c_ptr), VALUE                          :: gpu
            real(c_double), intent(in), dimension(*)    :: uext
            real(c_double), intent(in), dimension(*)    :: vext
            real(c_double), intent(in), dimension(*)    :: wext
            real(c_double), intent(in), dimension(*)    :: text
            real(c_double), intent(in), dimension(*)    :: qext
            real(c_double), intent(in), dimension(*)    :: z
            real(c_double), intent(in), dimension(*)    :: zz
        end subroutine

        subroutine gpuwrite(gpu) bind(c,name="ParticleWrite")
            use particle_struct
            use iso_c_binding, only: c_ptr
            type(c_ptr), VALUE        :: gpu
        end subroutine

        subroutine gpuadd(gpu, position, input) bind(c,name="ParticleAdd")
            use particle_struct
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr), VALUE        :: gpu
            integer(c_int), VALUE     :: position
            type(gpu_particle)        :: input
        end subroutine

        type(gpu_particle) function gpuget(gpu, position) bind(c,name="ParticleGet")
            use particle_struct
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr), VALUE        :: gpu
            integer(c_int), VALUE     :: position
        end function

        subroutine gpuupload(gpu) bind(c,name="ParticleUpload")
            use iso_c_binding, only: c_ptr
            type(c_ptr), VALUE, intent(in)  :: gpu
        end subroutine

        subroutine gpuinterpolate(gpu,dx,dy,nnz,ox,oy,oz) bind(c,name="ParticleInterpolate")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr), VALUE, intent(in)      :: gpu
            real(c_double), VALUE, intent(in)   :: dx
            real(c_double), VALUE, intent(in)   :: dy
            integer(c_int), VALUE, intent(in)   :: nnz
            integer(c_int), VALUE, intent(in)   :: ox
            integer(c_int), VALUE, intent(in)   :: oy
            integer(c_int), VALUE, intent(in)   :: oz
        end subroutine

        subroutine gpustep(gpu, it, istage, dt) bind(c,name="ParticleStep")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr), VALUE, intent(in)      :: gpu
            integer(c_int), VALUE, intent(in)   :: it
            integer(c_int), VALUE, intent(in)   :: istage
            real(c_double), VALUE, intent(in)   :: dt
        end subroutine

        subroutine gpunonperiodic(gpu) bind(c,name="ParticleUpdateNonPeriodic")
            use iso_c_binding, only: c_ptr, c_double
            type(c_ptr), VALUE, intent(in)      :: gpu
        end subroutine

        subroutine gpuperiodic(gpu) bind(c,name="ParticleUpdatePeriodic")
            use iso_c_binding, only: c_ptr, c_double
            type(c_ptr), VALUE, intent(in)      :: gpu
        end subroutine

        subroutine gpustatistics(gpu,dx,dy,ox,oy,oz,partcount_t,dzw) bind(c,name="ParticleCalculateStatistics")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr), VALUE, intent(in)              :: gpu
            real(c_double), VALUE, intent(in)   :: dx
            real(c_double), VALUE, intent(in)   :: dy
            integer(c_int), VALUE, intent(in)   :: ox
            integer(c_int), VALUE, intent(in)   :: oy
            integer(c_int), VALUE, intent(in)   :: oz
            real(c_double), intent(inout), dimension(*)    :: partcount_t
            real(c_double), intent(inout), dimension(*)    :: dzw
        end subroutine

        subroutine gpudownload(gpu) bind(c,name="ParticleDownloadHost")
            use iso_c_binding, only: c_ptr
            type(c_ptr), VALUE, intent(in)  :: gpu
        end subroutine
    end interface
contains
    subroutine select_gpu_master(processors)
            integer, intent(in)     :: processors
            include 'mpif.h'

            integer :: gpu_count = 0, i = 0, myid = 0, ierr = 0
            integer, dimension(processors) :: gpus

            call mpi_comm_rank(mpi_comm_world,myid,ierr)

            gpu_count = gpudevices()
            call mpi_gather(gpu_count,1,mpi_integer,gpus,1,mpi_integer,0,mpi_comm_world,ierr)

            if( myid .eq. 0 ) then
            gpu_master_rank = -1
            do  i=1,processors
                if( gpus(i) .gt. 0 ) then
                gpu_master_rank = i-1
                exit
                end if
            end  do
            end if
        end subroutine

        subroutine initialize_gpu()
            use pars, only: maxnx,maxny,maxnz,xl,yl,zl
            use con_stats, only: z
            use particles

            !gpu = newgpu(tnumpart,maxnx+6,maxny+6,maxnz+2,size(z), xl,yl,zl,delta_vis)
            gpu = newgpu(tnumpart,maxnx,maxny,maxnz,size(z), xl,yl,zl,delta_vis)
        end subroutine

        subroutine transfer_particles()
            use pars, only: numprocs, myid
            use particles
            use particle_struct

            include 'mpif.h'

            integer :: ierr, i, iCurrent
            integer, allocatable :: pCounts(:), pDispls(:)
            type(particle), allocatable :: pCurrent(:), pTotal(:)
            type(gpu_particle) :: gpuParticle

            allocate(pCurrent(numpart), pTotal(tnumpart))
            allocate(pCounts(numprocs), pDispls(numprocs))

            iCurrent = 1
            part => first_particle
            do while (associated(part))
                pCurrent(iCurrent) = part
                iCurrent = iCurrent + 1
                part => part%next
            end do
            iCurrent = iCurrent - 1

            call mpi_gather(iCurrent, 1, mpi_integer, pCounts, 1, mpi_integer, gpu_master_rank, mpi_comm_world, ierr)

            if (myid .eq. gpu_master_rank) then
                iCurrent = 0
                do  i=1,numprocs
                    pDispls(i) = iCurrent
                    iCurrent = iCurrent + pCounts(i)
                end do
            end if

            call mpi_gatherv(pCurrent, numpart, particletype, pTotal, pCounts, pDispls, particletype, gpu_master_rank, mpi_comm_world, ierr)

            if (myid .eq. gpu_master_rank) then
                do i = 1,tnumpart
                    gpuParticle%pidx = pTotal(i)%pidx
                    gpuParticle%procidx = pTotal(i)%procidx

                    gpuParticle%vp(1:3) = pTotal(i)%vp(1:3)
                    gpuParticle%xp(1:3) = pTotal(i)%xp(1:3)
                    gpuParticle%uf(1:3) = pTotal(i)%uf(1:3)
                    gpuParticle%xrhs(1:3) = pTotal(i)%xrhs(1:3)
                    gpuParticle%vrhs(1:3) = pTotal(i)%vrhs(1:3)

                    gpuParticle%Tp = pTotal(i)%Tp
                    gpuParticle%Tprhs_s = pTotal(i)%Tprhs_s
                    gpuParticle%Tprhs_L = pTotal(i)%Tprhs_L
                    gpuParticle%Tf = pTotal(i)%Tf
                    gpuParticle%radius = pTotal(i)%radius
                    gpuParticle%radrhs = pTotal(i)%radrhs
                    gpuParticle%qinf = pTotal(i)%qinf
                    gpuParticle%qstar = pTotal(i)%qstar

                    call gpuadd(gpu, i - 1, gpuParticle)
                end do
                call gpuupload(gpu)
            end if
        end subroutine

        subroutine calculate_statistics()
            use pars, only: nnz, nny, nnx, numprocs, myid, dzw
            use con_data, only: dx, dy
            use particles
            use particle_struct

            include 'mpif.h'

            if( myid .eq. gpu_master_rank ) then
                call gpustatistics(gpu,dx,dy,nnz+2,nny,nnx,partcount_t,dzw)
            end if

            ! Distribute Statistics to Processors
        end subroutine

        subroutine compare_particles()
            use pars, only: numprocs, myid
            use particles
            use particle_struct

            include 'mpif.h'

            integer :: ierr, i, j, k, iCurrent
            integer, allocatable :: pCounts(:), pDispls(:)
            type(particle), allocatable :: pCurrent(:), pTotal(:)
            type(gpu_particle) :: gPart

            allocate(pCurrent(numpart), pTotal(tnumpart))
            allocate(pCounts(numprocs), pDispls(numprocs))

            iCurrent = 1
            part => first_particle
            do while (associated(part))
                pCurrent(iCurrent) = part
                iCurrent = iCurrent + 1
                part => part%next
            end do
            iCurrent = iCurrent - 1

            call mpi_gather(iCurrent, 1, mpi_integer, pCounts, 1, mpi_integer, gpu_master_rank, mpi_comm_world, ierr)

            if (myid .eq. gpu_master_rank) then
                iCurrent = 0
                do  i=1,numprocs
                    pDispls(i) = iCurrent
                    iCurrent = iCurrent + pCounts(i)
                end do
            end if

            call mpi_gatherv(pCurrent, numpart, particletype, pTotal, pCounts, pDispls, particletype, gpu_master_rank, mpi_comm_world, ierr)

            if (myid .eq. gpu_master_rank) then
                call gpudownload(gpu)
                do i = 1,tnumpart
                    gPart = gpuget(gpu, i-1)

9001                format(a10, 3x, 'expected:', T25, f14.6, T50, 'actual:', 1x, f14.6)
9002                format(a10, 1x, i1, 1x, 'expected:', T25, f14.6, T50, 'actual:', 1x, f14.6)

                    do k = 1,tnumpart
                        if (pTotal(k)%pidx .eq. gPart%pidx) then
                            write(*,*) "Particle:", i, "Processor:", pTotal(k)%procidx, "ID:", pTotal(k)%pidx

                            do j = 1,3
                                if( dabs(pTotal(k)%vp(j)-gPart%vp(j)) .gt. 1e-5 ) then
                                    write(*,9002) "vp", j, pTotal(k)%vp(j), gPart%vp(j)
                                end if
                            end do

                            do j = 1,3
                                if( dabs(pTotal(k)%xp(j)-gPart%xp(j)) .gt. 1e-5 ) then
                                    write(*,9002) "xp", j, pTotal(k)%xp(j), gPart%xp(j)
                                end if
                            end do

                            do j = 1,3
                                if( dabs(pTotal(k)%uf(j) - gPart%uf(j)) .gt. 1e-5 ) then
                                    write(*,9002) "uf", j, pTotal(k)%uf(j), gPart%uf(j)
                                end if
                            end do

                            do j = 1,3
                                if( dabs(pTotal(k)%xrhs(j)-gPart%xrhs(j)) .gt. 1e-5) then
                                    write(*,9002) "xrhs", j, pTotal(k)%xrhs(j), gPart%xrhs(j)
                                end if
                            end do

                            do j = 1,3
                                if(dabs(pTotal(k)%vrhs(j)-gPart%vrhs(j)) .gt. 1e-5) then
                                    write(*,9002) "vrhs", j, pTotal(k)%vrhs(j), gPart%vrhs(j)
                                end if
                            end do

                            if( dabs(pTotal(k)%Tp - gPart%Tp) .gt. 1e-5 ) then
                                write(*,9001) "Tp", pTotal(k)%Tp, gPart%Tp
                            end if

                            if(dabs(pTotal(k)%Tprhs_s-gPart%Tprhs_s) .gt. 1e-5) then
                                write(*,9001) "Tprhs_s", pTotal(k)%Tprhs_s, gPart%Tprhs_s
                            end if

                            if( dabs(pTotal(k)%Tprhs_L-gPart%Tprhs_L) .gt. 1e-5) then
                                write(*,9001) "Tprhs_L", pTotal(k)%Tprhs_L, gPart%Tprhs_L
                            end if

                            if( dabs(pTotal(k)%Tf - gPart%Tf) .gt. 1e-5 ) then
                                write(*,9001) "Tf", pTotal(k)%Tf, gPart%Tf
                            end if

                            if(dabs(pTotal(k)%radius - gPart%radius) .gt. 1e-5) then
                                write(*,9001) "radius", pTotal(k)%radius, gPart%radius
                            end if

                            if(dabs(pTotal(k)%radrhs-gPart%radrhs) .gt. 1e-5) then
                                write(*,9001) "radrhs", pTotal(k)%radrhs, gPart%radrhs
                            end if

                            if(dabs(pTotal(k)%qinf-gPart%qinf) .gt. 1e-5) then
                                write(*,9001) "qinf", pTotal(k)%qinf, gPart%qinf
                            end if

                            if(dabs(pTotal(k)%qstar-gPart%qstar) .gt. 1e-5) then
                                write(*,9001) "qstar", pTotal(k)%qstar, gPart%qstar
                            end if

                            write(*,*) ""
                        end if
                    end do
                end do
            end if
        end subroutine
end module