module particle_struct
    use iso_c_binding
    implicit none

    type, bind(c) :: gpu_particle
        integer(c_int) :: pidx, procidx
        real(c_double) :: vp(3), xp(3), uf(3), xrhs(3), vrhs(3)
        real(c_double) :: Tp, Tprhs_s, Tprhs_L, Tf, radius
        real(c_double) :: radrhs, qinf, qstar
    end type gpu_particle

    type, bind(c) :: gpu_parameters
        integer(c_int) :: Evaporation, LinearInterpolation

        real(c_double) :: rhoa, nuf, Cpa, Pra, Sc

        real(c_double) :: rhow, part_grav, Cpp, Mw, Ru, Ms, Sal, Gam, Ion, Os

        real(c_double) :: radius_mass
    end type gpu_parameters
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

        type(c_ptr) function newgpu(count,h,w,d,xl,yl,zl,z,zz,params) bind(c,name="NewGPU")
            use particle_struct, only: gpu_parameters
            use iso_c_binding, only: c_ptr, c_int, c_double
            integer(c_int), VALUE, intent(in)      :: count
            integer(c_int), VALUE, intent(in)      :: h
            integer(c_int), VALUE, intent(in)      :: w
            integer(c_int), VALUE, intent(in)      :: d
            real(c_double), VALUE, intent(in)      :: xl
            real(c_double), VALUE, intent(in)      :: yl
            real(c_double), VALUE, intent(in)      :: zl
            real(c_double), intent(in), dimension(*)    :: z
            real(c_double), intent(in), dimension(*)    :: zz
            type(gpu_parameters)                   :: params
        end function

        subroutine gpucopyfield(gpu,uext,vext,wext,text,qext) bind(c,name="ParticleFieldSet")
            use iso_c_binding, only: c_ptr, c_float, c_double

#ifdef BUILD_FIELD_DOUBLE
            integer, parameter :: c_prec = c_double
#else
            integer, parameter :: c_prec = c_float
#endif
            type(c_ptr), VALUE                          :: gpu
            real(c_prec), intent(in), dimension(*)    :: uext
            real(c_prec), intent(in), dimension(*)    :: vext
            real(c_prec), intent(in), dimension(*)    :: wext
            real(c_prec), intent(in), dimension(*)    :: text
            real(c_prec), intent(in), dimension(*)    :: qext
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

        subroutine gpuinterpolate(gpu,dx,dy) bind(c,name="ParticleInterpolate")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr), VALUE, intent(in)      :: gpu
            real(c_double), VALUE, intent(in)   :: dx
            real(c_double), VALUE, intent(in)   :: dy
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

        subroutine gpustatistics(gpu,dx,dy,ox,oy,oz,dzw) bind(c,name="ParticleCalculateStatistics")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr), VALUE, intent(in)              :: gpu
            real(c_double), VALUE, intent(in)           :: dx
            real(c_double), VALUE, intent(in)           :: dy
            integer(c_int), VALUE, intent(in)           :: ox
            integer(c_int), VALUE, intent(in)           :: oy
            integer(c_int), VALUE, intent(in)           :: oz
            real(c_double), intent(inout), dimension(*) :: dzw
        end subroutine

        subroutine gpudownload(gpu) bind(c,name="ParticleDownload")
            use iso_c_binding, only: c_ptr
            type(c_ptr), VALUE, intent(in)  :: gpu
        end subroutine

        subroutine gpufillstatistics(gpu,partcount,vpsum,vpsumsq,rsum,tsum,tf_sum,qf_sum,qstar_sum,single_stats) bind(c,name="ParticleFillStatistics")
            use iso_c_binding, only: c_ptr,c_double
            type(c_ptr), VALUE, intent(in)  :: gpu
            real(c_double), intent(inout), dimension(*) :: partcount
            real(c_double), intent(inout), dimension(*) :: vpsum
            real(c_double), intent(inout), dimension(*) :: vpsumsq
            real(c_double), intent(inout), dimension(*) :: rsum
            real(c_double), intent(inout), dimension(*) :: tsum
            real(c_double), intent(inout), dimension(*) :: tf_sum
            real(c_double), intent(inout), dimension(*) :: qf_sum
            real(c_double), intent(inout), dimension(*) :: qstar_sum
            real(c_double), intent(inout), dimension(*) :: single_stats
        end subroutine

        subroutine gpuparticlegenerate(gpu,processors,ncpus,seed,temperature,radius,qinfp) bind(c,name="ParticleGenerate")
            use iso_c_binding, only: c_ptr,c_int,c_double
            type(c_ptr), VALUE, intent(in)  :: gpu
            integer(c_int), VALUE, intent(in)           :: processors
            integer(c_int), VALUE, intent(in)           :: ncpus
            integer(c_int), VALUE, intent(in)           :: seed
            real(c_double), VALUE, intent(in)           :: temperature
            real(c_double), VALUE, intent(in)           :: radius
            real(c_double), VALUE, intent(in)           :: qinfp
        end subroutine
    end interface
contains
        subroutine select_gpu_master()
            use pars, only: numprocs, myid

            include 'mpif.h'

            integer :: gpu_count, i, ierr
            integer, dimension(numprocs) :: gpus

            gpu_count = gpudevices()
            call mpi_gather(gpu_count,1,mpi_integer,gpus,1,mpi_integer,0,mpi_comm_world,ierr)

            if( myid .eq. 0 ) then
                gpu_master_rank = -1
                do  i=1,numprocs
                    if( gpus(i) .gt. 0 ) then
                        gpu_master_rank = i-1
                        exit
                    end if
                end  do
            end if

            ! Distribute to all processors
            call mpi_bcast(gpu_master_rank, 1, mpi_integer, 0, mpi_comm_world, ierr)
        end subroutine

        subroutine initialize_gpu()
            use pars, only: maxnx,maxny,maxnz,xl,yl,zl,myid,ievap,ilin,numprocs,ncpu_s
            use con_stats, only: z, zz
            use particles
            use particle_struct, only: gpu_parameters

            type(gpu_parameters) :: parameters

            if( myid .eq. gpu_master_rank ) then
                ! Setup Parameters
                parameters%Evaporation = ievap
                parameters%LinearInterpolation = ilin

                parameters%rhoa = rhoa
                parameters%nuf = nuf
                parameters%Cpa = Cpa
                parameters%Pra = Pra
                parameters%Sc = Sc

                parameters%rhow = rhow
                parameters%part_grav = part_grav
                parameters%Cpp = Cpp
                parameters%Mw = Mw
                parameters%Ru = Ru
                parameters%Ms = Ms
                parameters%Sal = Sal
                parameters%Gam = Gam
                parameters%Ion = Ion
                parameters%Os = Os

                parameters%radius_mass = radius_init

                ! Create GPU Instance
                gpu = newgpu(tnumpart,maxnx+5,maxny+5,maxnz+2,xl,yl,zl,z,zz,parameters)
                call gpuparticlegenerate(gpu,numprocs,ncpu_s,1080,Tp_init,radius_init,qf_init)
            end if
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

        subroutine assemble_gpu_data
            use pars
            use fields
            use class_Profiler
            use con_stats, only: z, zz
            use, intrinsic :: iso_fortran_env

            implicit none
            include 'mpif.h'

            integer :: i, j, k
            integer :: receive_size,send_size
            integer :: istart,iend,kstart,kend
            integer :: buf_size(4),buf_dims(4)
            integer :: iproc,istatus(mpi_status_size),ierr

#ifdef BUILD_FIELD_DOUBLE
            integer, parameter :: prec = REAL64
            integer, parameter :: mpi_prec = mpi_real8
#else
            integer, parameter :: prec = REAL32
            integer, parameter :: mpi_prec = mpi_real4
#endif
            real(prec),allocatable :: receive_buf(:,:,:),send_buf(:,:,:)

            !The fields to be assembeled on GPU proc: (2 periodic halos to the left, 3 to right, top/bottom in nz)
            real(prec) :: u_full(-1:maxnx+3,-1:maxny+3,0:maxnz+1)
            real(prec) :: v_full(-1:maxnx+3,-1:maxny+3,0:maxnz+1)
            real(prec) :: w_full(-1:maxnx+3,-1:maxny+3,0:maxnz+1)
            real(prec) :: T_full(-1:maxnx+3,-1:maxny+3,0:maxnz+1)
            real(prec) :: q_full(-1:maxnx+3,-1:maxny+3,0:maxnz+1)

            type(Profiler) :: tTransfer, tHalo, tUpload

            !NOTE: THE Z HALOS (0 and nz+1) WILL BE ZERO, BUT THIS SHOULDN'T MATTER SINCE THEY DON'T GET USED
            u_full = 0.0
            v_full = 0.0
            w_full = 0.0
            T_full = 0.0
            q_full = 0.0

            !The GPU proc (assumed proc 0 here) receives, everyone else sends
            if (myid == gpu_master_rank) then
                call tTransfer%start(gpu_master_rank, .false.)
                do iproc=0,numprocs-1
                    if (myid .ne. iproc) then
                        !Figure out how much data is coming:
                        call mpi_recv(buf_size,4,mpi_integer,iproc,1,mpi_comm_world,istatus,ierr)

                        !Allocate the receive buffer;
                        buf_dims(1) = nnx
                        buf_dims(2) = buf_size(2)-buf_size(1) + 1
                        buf_dims(3) = buf_size(4)-buf_size(3) + 1

                        allocate(receive_buf(buf_dims(1),buf_dims(2),buf_dims(3)))
                        receive_size = buf_dims(1)*buf_dims(2)*buf_dims(3)

                        !u-velocity
                        !receive the buffer:
                        call mpi_recv(receive_buf,receive_size,mpi_prec,iproc,1,mpi_comm_world,istatus,ierr)

                        !Now put into full field
                        u_full(1:nnx,buf_size(1):buf_size(2),buf_size(3):buf_size(4)) = receive_buf(1:buf_dims(1),1:buf_dims(2),1:buf_dims(3))

                        !v-velocity
                        !receive the buffer:
                        call mpi_recv(receive_buf,receive_size,mpi_prec,iproc,1,mpi_comm_world,istatus,ierr)

                        !Now put into full field
                        v_full(1:nnx,buf_size(1):buf_size(2),buf_size(3):buf_size(4)) = receive_buf(1:buf_dims(1),1:buf_dims(2),1:buf_dims(3))

                        !w-velocity
                        !receive the buffer:
                        call mpi_recv(receive_buf,receive_size,mpi_prec,iproc,1,mpi_comm_world,istatus,ierr)

                        !Now put into full field
                        w_full(1:nnx,buf_size(1):buf_size(2),buf_size(3):buf_size(4)) = receive_buf(1:buf_dims(1),1:buf_dims(2),1:buf_dims(3))

                        !temperature
                        !receive the buffer:
                        call mpi_recv(receive_buf,receive_size,mpi_prec,iproc,1,mpi_comm_world,istatus,ierr)

                        !Now put into full field
                        T_full(1:nnx,buf_size(1):buf_size(2),buf_size(3):buf_size(4)) = receive_buf(1:buf_dims(1),1:buf_dims(2),1:buf_dims(3))

                        !humidity
                        !receive the buffer:
                        call mpi_recv(receive_buf,receive_size,mpi_prec,iproc,1,mpi_comm_world,istatus,ierr)

                        !Now put into full field
                        q_full(1:nnx,buf_size(1):buf_size(2),buf_size(3):buf_size(4)) = receive_buf(1:buf_dims(1),1:buf_dims(2),1:buf_dims(3))

                        deallocate(receive_buf)
                    else !the GPU proc simply puts its portion into full
                        u_full(1:nnx,iys:iye,izs:ize) = u(1:nnx,iys:iye,izs:ize)
                        v_full(1:nnx,iys:iye,izs:ize) = v(1:nnx,iys:iye,izs:ize)
                        w_full(1:nnx,iys:iye,izs:ize) = w(1:nnx,iys:iye,izs:ize)
                        T_full(1:nnx,iys:iye,izs:ize) = t(1:nnx,iys:iye,1,izs:ize)
                        q_full(1:nnx,iys:iye,izs:ize) = t(1:nnx,iys:iye,2,izs:ize)
                    end if
                end do
                call tTransfer%finish(gpu_master_rank, "Recieve Time: ", .false.)

                !Finally fill the halos so that GPU doesn't have to conditionally search for periodicity
                call tHalo%start(gpu_master_rank, .false.)
                u_full(-1:0,1:maxny,0:maxnz+1) = u_full(maxnx-1:maxnx,1:maxny,0:maxnz+1)
                u_full(maxnx+1:maxnx+3,1:maxny,0:maxnz+1) = u_full(1:3,1:maxny,0:maxnz+1)
                u_full(-1:maxnx+3,-1:0,0:maxnz+1) = u_full(-1:maxnx+3,maxny-1:maxny,0:maxnz+1)
                u_full(-1:maxnx+3,maxny+1:maxny+3,0:maxnz+1) = u_full(-1:maxnx+3,1:3,0:maxnz+1)

                v_full(-1:0,1:maxny,0:maxnz+1) = v_full(maxnx-1:maxnx,1:maxny,0:maxnz+1)
                v_full(maxnx+1:maxnx+3,1:maxny,0:maxnz+1) = v_full(1:3,1:maxny,0:maxnz+1)
                v_full(-1:maxnx+3,-1:0,0:maxnz+1) = v_full(-1:maxnx+3,maxny-1:maxny,0:maxnz+1)
                v_full(-1:maxnx+3,maxny+1:maxny+3,0:maxnz+1) = v_full(-1:maxnx+3,1:3,0:maxnz+1)

                w_full(-1:0,1:maxny,0:maxnz+1) = w_full(maxnx-1:maxnx,1:maxny,0:maxnz+1)
                w_full(maxnx+1:maxnx+3,1:maxny,0:maxnz+1) = w_full(1:3,1:maxny,0:maxnz+1)
                w_full(-1:maxnx+3,-1:0,0:maxnz+1) = w_full(-1:maxnx+3,maxny-1:maxny,0:maxnz+1)
                w_full(-1:maxnx+3,maxny+1:maxny+3,0:maxnz+1) = w_full(-1:maxnx+3,1:3,0:maxnz+1)

                T_full(-1:0,1:maxny,0:maxnz+1) = T_full(maxnx-1:maxnx,1:maxny,0:maxnz+1)
                T_full(maxnx+1:maxnx+3,1:maxny,0:maxnz+1) = T_full(1:3,1:maxny,0:maxnz+1)
                T_full(-1:maxnx+3,-1:0,0:maxnz+1) = T_full(-1:maxnx+3,maxny-1:maxny,0:maxnz+1)
                T_full(-1:maxnx+3,maxny+1:maxny+3,0:maxnz+1) = T_full(-1:maxnx+3,1:3,0:maxnz+1)

                q_full(-1:0,1:maxny,0:maxnz+1) = q_full(maxnx-1:maxnx,1:maxny,0:maxnz+1)
                q_full(maxnx+1:maxnx+3,1:maxny,0:maxnz+1) = q_full(1:3,1:maxny,0:maxnz+1)
                q_full(-1:maxnx+3,-1:0,0:maxnz+1) = q_full(-1:maxnx+3,maxny-1:maxny,0:maxnz+1)
                q_full(-1:maxnx+3,maxny+1:maxny+3,0:maxnz+1) = q_full(-1:maxnx+3,1:3,0:maxnz+1)
                call tHalo%finish(gpu_master_rank, "Halo Time: ", .false.)

                !Now the "full" fields are complete and can be transferred to GPU
                call tUpload%start(gpu_master_rank, .false.)
                call gpucopyfield(gpu,u_full,v_full,w_full,T_full,q_full)
                call tUpload%finish(gpu_master_rank, "GPU Transfer Time: ", .false.)
            else
                allocate(send_buf(nnx,iye-iys+1,ize-izs+1))

                !First need to send the size of my buffer:
                buf_size(1) = iys
                buf_size(2) = iye
                buf_size(3) = izs
                buf_size(4) = ize
                call mpi_send(buf_size,4,mpi_integer,gpu_master_rank,1,mpi_comm_world,ierr)

                send_size = nnx*(iye-iys+1)*(ize-izs+1)

                !u-velocity
                send_buf = u(1:nnx,iys:iye,izs:ize)
                call mpi_send(send_buf,send_size,mpi_prec,gpu_master_rank,1,mpi_comm_world,ierr)

                !v-velocity
                send_buf = v(1:nnx,iys:iye,izs:ize)
                call mpi_send(send_buf,send_size,mpi_prec,gpu_master_rank,1,mpi_comm_world,ierr)

                !w-velocity
                send_buf = w(1:nnx,iys:iye,izs:ize)
                call mpi_send(send_buf,send_size,mpi_prec,gpu_master_rank,1,mpi_comm_world,ierr)

                !temperature
                send_buf = t(1:nnx,iys:iye,1,izs:ize)
                call mpi_send(send_buf,send_size,mpi_prec,gpu_master_rank,1,mpi_comm_world,ierr)

                !humidity
                send_buf = t(1:nnx,iys:iye,2,izs:ize)
                call mpi_send(send_buf,send_size,mpi_prec,gpu_master_rank,1,mpi_comm_world,ierr)

                deallocate(send_buf)
            end if
        end subroutine assemble_gpu_data

        subroutine gpu_particle_step(it, istage)
            use class_Profiler
            use pars, only: myid
            use con_data, only: dx, dy, dt

            include 'mpif.h'

            integer :: ierr, it, istage, step, substeps
            type(Profiler) :: tAssemble, tStep

            call tAssemble%start(gpu_master_rank)
            call assemble_gpu_data
            call tAssemble%finish(gpu_master_rank, "Assemble Step Time: ")

            call tStep%start(gpu_master_rank)
            if( myid .eq. gpu_master_rank ) then
                call gpuinterpolate(gpu,dx,dy)
                call gpustep(gpu, it, istage, dt)
                call gpunonperiodic(gpu)
                call gpuperiodic(gpu)
            end if
            call tStep%finish(gpu_master_rank, "GPU Step Time: ")
        end subroutine

        subroutine gpu_particle_substep(it, substeps)
            use class_Profiler
            use pars, only: myid
            use con_data, only: dx, dy, dt

            include 'mpif.h'

            integer :: ierr, it, istage, step, substeps
            type(Profiler) :: tAssemble, tStep

            call tAssemble%start(gpu_master_rank)
            call assemble_gpu_data
            call tAssemble%finish(gpu_master_rank, "Assemble Step Time: ")

            call tStep%start(gpu_master_rank)
            if( myid .eq. gpu_master_rank ) then
                call gpuinterpolate(gpu,dx,dy)
                do step = 1,substeps
                    do istage = 1,3
                        call gpustep(gpu, it, istage, dt/substeps)
                        call gpunonperiodic(gpu)
                        call gpuperiodic(gpu)
                    end do
                end do
            end if
            call tStep%finish(gpu_master_rank, "GPU Step Time: ")
        end subroutine

!        subroutine gpu_particle_substep(it, substeps)
!            use class_Profiler
!            use pars, only: myid
!            use con_data, only: dx, dy, dt
!
!            include 'mpif.h'
!
!            integer :: ierr, it, istage, step, substeps
!            type(Profiler) :: tAssemble, tStep
!
!            call tAssemble%start(gpu_master_rank)
!            call assemble_gpu_data
!            call tAssemble%finish(gpu_master_rank, "Assemble Step Time: ")
!
!            call tStep%start(gpu_master_rank)
!            if( myid .eq. gpu_master_rank ) then
!                call gpuinterpolate(gpu,dx,dy)
!                do step = 1,substeps
!                    do istage = 1,3
!                        call gpustep(gpu, it, istage, dt/substeps)
!                        call gpunonperiodic(gpu)
!                        call gpuperiodic(gpu)
!                    end do
!                end do
!            end if
!            call tStep%finish(gpu_master_rank, "GPU Step Time: ")
!        end subroutine

        subroutine calculate_statistics(stat)
            use pars, only: nnz, nny, nnx, numprocs, myid, dzw
            use con_data, only: dx, dy, nscl
            use particles
            use particle_struct
            use class_Profiler
            implicit none
            include 'mpif.h'

            integer :: iz, istatus, ierr, dim, tdim,sdim
            real :: stat(1:nnz,34 + 5*nscl)
            real(8), allocatable :: pCount(:), vSum(:), vSumSQ(:)
            real(8), allocatable :: rSum(:),tSum(:),tf_Sum(:),qf_Sum(:),qstar_Sum(:)
#ifdef BUILD_CUDA_VERIFY
            integer :: succeeded, failures
#endif

            type(Profiler) :: timer

            dim = nnz + 2
            tdim = dim * 3
            sdim = size(single_stats)
            allocate(pCount(dim), vSum(tdim), vSumSQ(tdim))
            allocate(rSum(dim),tSum(dim),tf_Sum(dim),qf_Sum(dim),qstar_Sum(dim))

            call timer%start(gpu_master_rank)
            if( myid .eq. gpu_master_rank ) then
                call gpustatistics(gpu,dx,dy,nnz,nny,nnx,dzw)
                call gpufillstatistics(gpu,pCount,vSum,vSumSQ,rSum,tSum,tf_Sum,qf_Sum,qstar_Sum,single_stats)

                if( myid .ne. 0 ) then
                    call mpi_send(pCount,dim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(vSum,tdim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(vSumSQ,tdim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(rSum,dim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(tSum,dim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(tf_Sum,dim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(qf_Sum,dim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(qstar_Sum,dim,mpi_real8,0,1,mpi_comm_world,ierr)
                    call mpi_send(single_stats,sdim,mpi_real8,0,1,mpi_comm_world,ierr)
                end if
            else
                if( myid .eq. 0 ) then
                    call mpi_recv(pCount,dim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(vSum,tdim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(vSumSQ,tdim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(rSum,dim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(tSum,dim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(tf_Sum,dim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(qf_Sum,dim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(qstar_Sum,dim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                    call mpi_recv(single_stats,sdim,mpi_real8,gpu_master_rank,1,mpi_comm_world,istatus,ierr)
                end if
            end if

            if( myid .eq. 0 ) then
#ifdef BUILD_CUDA_VERIFY
                failures = 0
9001            format(a20, 3x, 'expected:', T35, f20.16, T60, 'actual:', 1x, f20.16, T90, 'diff:', 1x, f20.16)

                write(*,*) "Z Verify:"
#endif

                do iz=1,nnz
#ifdef BUILD_CUDA_VERIFY
                    succeeded = 1
                    if( dabs(stat(iz,11))-dabs(pCount(iz)) .gt. 1e-9 ) then
                        succeeded = 0
                    end if

                    if( dabs(stat(iz,12))-dabs(vSum((iz-1)*3+1)) .gt. 1e-9 ) then
                        succeeded = 0
                    end if
                    if( dabs(stat(iz,13))-dabs(vSum((iz-1)*3+2)) .gt. 1e-9 ) then
                        succeeded = 0
                    end if
                    if( dabs(stat(iz,14))-dabs(vSum((iz-1)*3+2)) .gt. 1e-9 ) then
                        succeeded = 0
                    end if

                    if( dabs(stat(iz,15))-dabs(vSumSQ((iz-1)*3+1)) .gt. 1e-9 ) then
                        succeeded = 0
                    end if
                    if( dabs(stat(iz,16))-dabs(vSumSQ((iz-1)*3+2)) .gt. 1e-9 ) then
                        succeeded = 0
                    end if
                    if( dabs(stat(iz,17))-dabs(vSumSQ((iz-1)*3+3)) .gt. 1e-9 ) then
                        succeeded = 0
                    end if

                    if( succeeded == 0 ) then
                        failures = failures + 1

                        write(*,*) "Z:", iz

                        write(*,9001) "Particle Count", stat(iz,11), pCount(iz), stat(iz,11) - pCount(iz)

                        write(*,9001) "Velociy Sum (x):", stat(iz,12), vSum((iz-1)*3+1), stat(iz,12) - vSum((iz-1)*3+1)
                        write(*,9001) "Velociy Sum (y):", stat(iz,13), vSum((iz-1)*3+2), stat(iz,13) - vSum((iz-1)*3+2)
                        write(*,9001) "Velociy Sum (z):", stat(iz,14), vSum((iz-1)*3+3), stat(iz,14) - vSum((iz-1)*3+3)

                        write(*,9001) "Velociy Squared (x):", stat(iz,15), vSumSQ((iz-1)*3+1), stat(iz,15) - vSumSQ((iz-1)*3+1)
                        write(*,9001) "Velociy Squared (y):", stat(iz,16), vSumSQ((iz-1)*3+2), stat(iz,16) - vSumSQ((iz-1)*3+2)
                        write(*,9001) "Velociy Squared (z):", stat(iz,17), vSumSQ((iz-1)*3+3), stat(iz,17) - vSumSQ((iz-1)*3+3)

                        write(*,*) ""
                    end if
#endif
                    stat(iz,11) = pCount(iz)
                    stat(iz,12) = vSum((iz-1)*3+1)
                    stat(iz,13) = vSum((iz-1)*3+2)
                    stat(iz,14) = vSum((iz-1)*3+3)
                    stat(iz,15) = vSumSQ((iz-1)*3+1)
                    stat(iz,16) = vSumSQ((iz-1)*3+2)
                    stat(iz,17) = vSumSQ((iz-1)*3+3)

                    stat(iz,28) = rSum(iz)
                    stat(iz,22) = tSum(iz)
                    stat(iz,24) = tf_Sum(iz)
                    stat(iz,25) = qf_Sum(iz)
                    stat(iz,34) = qstar_Sum(iz)
                end do
#ifdef BUILD_CUDA_VERIFY
                write(*,*) "Total Failures: ", failures
#endif
            end if
            call timer%finish(gpu_master_rank, "Statistics Time: ")
        end subroutine

        subroutine setup_particles()
            use pars, only: myid, xl, yl
            use con_data, only: dx, dy
            use particles
            use fields

            integer :: i, j, k

            ! Setup Particles
            part => first_particle
            do while (associated(part))
                part%vp(1:3) = (/0.0, 0.0, 0.0/)
                part%xp(1:3) = (/0.087370281899,0.125141144198,0.018382617541/)
                part%uf(1:3) = (/0.0, 0.0, 0.0/)
                part%xrhs(1:3) = (/0.0, 0.0, 0.0/)
                part%vrhs(1:3) = (/0.0, 0.0, 0.0/)

                part%Tp = 0.0
                part%Tprhs_s = 0.0
                part%Tprhs_L = 0.0
                part%Tf = 0.0
                part%radius = 0.0
                part%radrhs = 0.0
                part%qinf = 0.0
                part%qstar = 0.0

                part => part%next
            end do

            ! Move to master and save
            call transfer_particles
            if (myid .eq. gpu_master_rank) then
                call gpuwrite(gpu)
            end if

            ! Run interpolation
            call fill_ext

            part => first_particle
            do while (associated(part))
                call uf_interp
                part => part%next
            end do

            ! Move to master and save
            call transfer_particles
            if (myid .eq. gpu_master_rank) then
                call gpuwrite(gpu)
            end if

            stop 0
        end subroutine

        subroutine compare_particles()
            use pars, only: numprocs, myid
            use particles
            use particle_struct

            include 'mpif.h'

            integer :: ierr, i, j, k, iCurrent, succeeded, failures
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
                failures = 0

                call gpudownload(gpu)
                do i = 1,tnumpart
                    gPart = gpuget(gpu, i-1)

9001                format(a10, 3x, 'expected:', T25, f14.6, T50, 'actual:', 1x, f14.6)
9002                format(a10, 1x, i1, 1x, 'expected:', T25, f14.6, T50, 'actual:', 1x, f14.6)

                    do k = 1,tnumpart
                        if (pTotal(k)%pidx .eq. gPart%pidx) then
                            succeeded = 1

                            do j = 1,3
                                if( dabs(pTotal(k)%vp(j)-gPart%vp(j)) .gt. 1e-5 ) then
                                    succeeded = 0
                                end if
                            end do

                            do j = 1,3
                                if( dabs(pTotal(k)%xp(j)-gPart%xp(j)) .gt. 1e-5 ) then
                                    succeeded = 0
                                end if
                            end do

                            do j = 1,3
                                if( dabs(pTotal(k)%uf(j) - gPart%uf(j)) .gt. 1e-5 ) then
                                    succeeded = 0
                                end if
                            end do

                            do j = 1,3
                                if( dabs(pTotal(k)%xrhs(j)-gPart%xrhs(j)) .gt. 1e-5) then
                                    succeeded = 0
                                end if
                            end do

                            do j = 1,3
                                if(dabs(pTotal(k)%vrhs(j)-gPart%vrhs(j)) .gt. 1e-5) then
                                    succeeded = 0
                                end if
                            end do

                            if( dabs(pTotal(k)%Tp - gPart%Tp) .gt. 1e-5 ) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%Tprhs_s-gPart%Tprhs_s) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if( dabs(pTotal(k)%Tprhs_L-gPart%Tprhs_L) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if( dabs(pTotal(k)%Tf - gPart%Tf) .gt. 1e-5 ) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%radius - gPart%radius) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%radrhs-gPart%radrhs) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%qinf-gPart%qinf) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if(dabs(pTotal(k)%qstar-gPart%qstar) .gt. 1e-5) then
                                succeeded = 0
                            end if

                            if( succeeded == 0 ) then
                                failures = failures + 1

                                write(*,*) "Particle:", i, "Processor:", pTotal(k)%procidx, "ID:", pTotal(k)%pidx

                                do j = 1,3
                                    write(*,9002) "vp", j, pTotal(k)%vp(j), gPart%vp(j)
                                end do

                                do j = 1,3
                                    write(*,9002) "xp", j, pTotal(k)%xp(j), gPart%xp(j)
                                end do

                                do j = 1,3
                                    write(*,9002) "uf", j, pTotal(k)%uf(j), gPart%uf(j)
                                end do

                                do j = 1,3
                                    write(*,9002) "xrhs", j, pTotal(k)%xrhs(j), gPart%xrhs(j)
                                end do

                                do j = 1,3
                                    write(*,9002) "vrhs", j, pTotal(k)%vrhs(j), gPart%vrhs(j)
                                end do

                                write(*,9001) "Tp", pTotal(k)%Tp, gPart%Tp
                                write(*,9001) "Tprhs_s", pTotal(k)%Tprhs_s, gPart%Tprhs_s
                                write(*,9001) "Tprhs_L", pTotal(k)%Tprhs_L, gPart%Tprhs_L
                                write(*,9001) "Tf", pTotal(k)%Tf, gPart%Tf
                                write(*,9001) "radius", pTotal(k)%radius, gPart%radius
                                write(*,9001) "radrhs", pTotal(k)%radrhs, gPart%radrhs
                                write(*,9001) "qinf", pTotal(k)%qinf, gPart%qinf
                                write(*,9001) "qstar", pTotal(k)%qstar, gPart%qstar

                                write(*,*) ""
                            end if
                        end if
                    end do
                end do

                write(*,*) "Total Failures: ", failures
            end if
        end subroutine
end module
