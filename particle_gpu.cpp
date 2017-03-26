#include "particle_gpu.h"
#include "stdio.h"
#include "assert.h"
#include <iomanip>
#include <fstream>
#include <iostream>

#ifndef BUILD_CUDA
#include "math.h"
#include "string.h"
#include "stdlib.h"
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifdef BUILD_CUDA
#define DEVICE __device__
#define GLOBAL __global__
#define CONSTANT __constant__
#define SHARED extern __shared__
#else
#define DEVICE
#define GLOBAL
#define SHARED
#define CONSTANT
#define cudaTextureObject_t double*
#endif

extern "C" int gpudevices(){
#ifdef BUILD_CUDA
    int nDevices;
    if( cudaGetDeviceCount(&nDevices) == cudaErrorInsufficientDriver ){
        return 0;
    }

    return nDevices;
#else
    return 1;
#endif
}

#ifdef BUILD_CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void cudaTextureObjectHelper( cudaTextureObject_t* tex, double* buff, int buff_size ) {
    // Create the resource description
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = buff;
    resDesc.res.linear.sizeInBytes = buff_size;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 32;

    // Create the texture description
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.normalizedCoords = 0;
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.readMode = cudaReadModeElementType;

    // Create texture object
    cudaCreateTextureObject(tex, &resDesc, &tex_desc, NULL);
}
#endif

CONSTANT Parameters cParams;

DEVICE void GPUFindXYNeighbours(const double dx, const double dy, const Particle* __restrict__ particles, int* __restrict__ neighbours){
    neighbours[0*6+2] = floor(particles[0].xp[0]/dx) + 1;
    neighbours[1*6+2] = floor(particles[0].xp[1]/dy) + 1;

    neighbours[0*6+1] = neighbours[0*6+2]-1;
    neighbours[0*6+0] = neighbours[0*6+1]-1;
    neighbours[0*6+3] = neighbours[0*6+2]+1;
    neighbours[0*6+4] = neighbours[0*6+3]+1;
    neighbours[0*6+5] = neighbours[0*6+4]+1;

    neighbours[1*6+1] = neighbours[1*6+2]-1;
    neighbours[1*6+0] = neighbours[1*6+1]-1;
    neighbours[1*6+3] = neighbours[1*6+2]+1;
    neighbours[1*6+4] = neighbours[1*6+3]+1;
    neighbours[1*6+5] = neighbours[1*6+4]+1;
}

GLOBAL void GGPUFindXYNeighbours(const double dx, const double dy, const Particle* __restrict__ particles, int* __restrict__ neighbours){
    GPUFindXYNeighbours(dx, dy, particles, neighbours);
}

int* ParticleFindXYNeighbours(const double dx, const double dy, const Particle* particle) {
    int *hResult = (int*) malloc(sizeof(int) * 12);

#ifdef BUILD_CUDA
    int *dResult;
    gpuErrchk( cudaMalloc( (void **)&dResult, sizeof(int) * 12 ) );

    Particle *dParticle;
    gpuErrchk( cudaMalloc( (void **)&dParticle, sizeof(Particle) * 1 ) );
    gpuErrchk( cudaMemcpy( dParticle, particle, sizeof(Particle) * 1, cudaMemcpyHostToDevice ) );

    GGPUFindXYNeighbours<<< 1, 1 >>> ( dx, dy, dParticle, dResult);
    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk( cudaMemcpy( hResult, dResult, sizeof(int) * 12 , cudaMemcpyDeviceToHost ) );
#else
    GPUFindXYNeighbours(dx, dy, particle, hResult);
#endif
    return hResult;
}

static __inline__ DEVICE double fetch_double(cudaTextureObject_t t, int pos){
#ifdef BUILD_CUDA
    uint2 p = tex1Dfetch<uint2>(t, pos);
    return __hiloint2double(p.y, p.x);
#else
    return t[pos];
#endif
}

GLOBAL void GPUFieldInterpolateLinear(const int nx, const int ny, const double dx, const double dy, const int nnz, const double* __restrict__ z, const double* __restrict__ zz, cudaTextureObject_t uext, cudaTextureObject_t vext, cudaTextureObject_t wext, cudaTextureObject_t Text, cudaTextureObject_t T2ext, const int pcount, Particle* __restrict__ particles) {
#ifdef BUILD_CUDA
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < pcount; idx += blockDim.x * gridDim.x) {
#else
    for( int idx = 0; idx < pcount; idx++){
#endif

// Setup shared memory for Z and ZZ
#ifdef BUILD_CUDA
    SHARED double shared[];

    double *zShared = shared, *zzShared = &shared[nnz], *dzu = &shared[nnz*2], *dzw = &shared[nnz*3+1];
    if( idx == 0 ){
        for( int i = 0; i < nnz; i++ ){
            zShared[i] = z[i];
            zzShared[i] = zz[i];
        }

        for( int i = 1; i < nnz; i++ ){
            dzu[i] = zzShared[i] - zzShared[i-1];
            dzw[i] = zShared[i] - zShared[i-1];
        }
        dzu[0] = dzu[1];
        dzw[0] = dzw[1];

        dzu[nnz] = dzu[nnz-1];
        dzw[nnz] = dzw[nnz-1];
    }
    __syncthreads();
#else
    const double *zShared = z;
    const double *zzShared = zz;

    double dzu[nnz], dzw[nnz];
    for( int i = 1; i < nnz-1; i++ ){
        dzw[i] = z[i] - z[i-1];
        dzu[i] = zz[i] - zz[i-1];
    }

    dzw[0] = dzw[1];
    dzu[0] = dzu[1];

    dzw[nnz] = dzw[nnz-1];
    dzu[nnz] = dzw[nnz-1];
#endif

    const int ipt = floor(particles[idx].xp[0]/dx) + 1;
    const int jpt = floor(particles[idx].xp[1]/dy) + 1;

    int kpt = 0;
    for( ; kpt < nnz; kpt++ ){
        if (zzShared[kpt] > particles[idx].xp[2]){
            break;
        }
    }
    kpt -= 1;

    int kwpt = 0;
    for( ; kwpt < nnz; kwpt++ ){
        if (zShared[kwpt] > particles[idx].xp[2]) {
            break;
        }
    }
    kwpt -= 1;

    particles[idx].uf[0] = 0.0;
    particles[idx].uf[1] = 0.0;
    particles[idx].uf[2] = 0.0;

    particles[idx].Tf = 0.0;
    particles[idx].qinf = 0.0;
    for( int i = 0; i < 2; i++ ){
        for( int j = 0; j < 2; j++ ){
            for( int k = 0; k < 2; k++ ){
                const int ix = i + ipt;
                const int iy = i + jpt;
                const int izuv = k + kpt;
                const int izw = k + kwpt;

                const double xv = dx * (i+ipt - 1);
                const double yv = dy * (j+jpt - 1);

                const double wtx = 1.0 - (abs(particles[idx].xp[0] - xv) / dx);
                const double wty = 1.0 - (abs(particles[idx].xp[1] - yv) / dy);
                const double wtz = 1.0 - (abs(particles[idx].xp[2] - zzShared[izuv]) / dzu[kpt+1]);
                const double wtzw = 1.0 - (abs(particles[idx].xp[2] - zShared[izw]) / dzw[kwpt+1]);

                particles[idx].uf[0] = particles[idx].uf[0] + fetch_double(uext, (ix+1)+(iy+1)*nx+izuv*ny*nx) * wtx * wty * wtz;
                particles[idx].uf[1] = particles[idx].uf[1] + fetch_double(vext, (ix+1)+(iy+1)*nx+izuv*ny*nx) * wtx * wty * wtz;
                particles[idx].uf[2] = particles[idx].uf[2] + fetch_double(wext, (ix+1)+(iy+1)*nx+izw*ny*nx) * wtx * wty * wtzw;
                particles[idx].Tf = particles[idx].Tf + fetch_double(Text, (ix+1)+(iy+1)*nx+izuv*ny*nx) * wtx * wty * wtz;
                particles[idx].qinf = particles[idx].qinf + fetch_double(T2ext, (ix+1)+(iy+1)*nx+izuv*ny*nx) * wtx * wty * wtz;
            }
        }
    }

    }
}

GLOBAL void GPUFieldInterpolate( const int nx, const int ny, const double dx, const double dy, const int nnz, const double* __restrict__ z, const double* __restrict__ zz, cudaTextureObject_t uext, cudaTextureObject_t vext, cudaTextureObject_t wext, cudaTextureObject_t Text, cudaTextureObject_t T2ext, const int pcount, Particle* __restrict__ particles){
#ifdef BUILD_CUDA
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= pcount ) return;
#else
    for( int idx = 0; idx < pcount; idx++){
#endif

#ifdef BUILD_CUDA
    SHARED double shared[];

    // Shared memory for Z and ZZ
    double *zShared = shared, *zzShared = &shared[nnz];
    if( idx == 0 ){
        for( int i = 0; i < nnz; i++ ){
            zShared[i] = z[i];
            zzShared[i] = zz[i];
        }
    }
    __syncthreads();
#else
    const double *zShared = z;
    const double *zzShared = zz;
#endif

    int ijpts[12];
    GPUFindXYNeighbours(dx, dy, &particles[idx], ijpts);

    int kuvpts[6] = { 0, 0, 0, 0, 0, 0 };
    for( ; kuvpts[2] < nnz; kuvpts[2]++ ){
        if (zzShared[kuvpts[2]] > particles[idx].xp[2]){
            break;
        }
    }
    kuvpts[2] -= 1;

    kuvpts[3] = kuvpts[2]+1;
    kuvpts[4] = kuvpts[3]+1;
    kuvpts[5] = kuvpts[4]+1;
    kuvpts[1] = kuvpts[2]-1;
    kuvpts[0] = kuvpts[1]-1;

    int kwpts[6] = { 0, 0, 0, 0, 0, 0 };
    for( ; kwpts[2] < nnz; kwpts[2]++ ){
        if (zShared[kwpts[2]] > particles[idx].xp[2]) {
            break;
        }
    }
    kwpts[2] -= 1;

    kwpts[3] = kwpts[2]+1;
    kwpts[4] = kwpts[3]+1;
    kwpts[5] = kwpts[4]+1;
    kwpts[1] = kwpts[2]-1;
    kwpts[0] = kwpts[1]-1;

    double wt[4][6] = {
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    };

    double dxvec[2] = { dx, dy };
    for( int iz = 0; iz < 2; iz++ ){
        for( int j = 0; j < 6; j++ ){
            double xjval = dxvec[iz]*(ijpts[iz*6+j]-1);
            double pj = 1.0;
            for( int k = 0; k < 6; k++ ){
                double xkval = dxvec[iz]*(ijpts[iz*6+k]-1);
                if (j != k) {
                    pj = pj*(particles[idx].xp[iz]-xkval)/(xjval-xkval);
                }
            }
            wt[iz][j] = pj;
        }
    }

    int first, last;
    if (kuvpts[2] == 1) {
        first = 2;
        last = 4;
        kuvpts[0] = 0;
        kuvpts[1] = 0;
    } else if (kuvpts[2] == 0) {
        first = 3;
        last = 5;
        kuvpts[0] = 0;
        kuvpts[1] = 0;
        kuvpts[2] = 0;
    } else if (kuvpts[2] < 0) {
        first = 0;
        last = 0;
        kuvpts[0] = 0;
        kuvpts[1] = 0;
        kuvpts[2] = 0;
    } else if (kuvpts[2] == 2) {
        first = 1;
        last = 5;
    } else if (kuvpts[2] == nnz-2) {
        first = 1;
        last = 3;
        kuvpts[3] = nnz-2;
        kuvpts[4] = nnz-2;
        kuvpts[5] = nnz-2;
    } else if ( kuvpts[2] > nnz-2) {
        first = 0;
        last = 0;
        kuvpts[3] = nnz-2;
        kuvpts[4] = nnz-2;
        kuvpts[5] = nnz-2;
    } else if (kuvpts[2] == nnz-3) {
        first = 2;
        last = 4;
        kuvpts[4] = nnz-2;
        kuvpts[5] = nnz-2;
    } else if (kuvpts[2] == nnz-4) {
        first = 1;
        last = 5;
    } else {
        first = 0;
        last = 6;
    }

    for( int j = first; j < last; j++){
        double xjval = zzShared[kuvpts[j]];
        double pj = 1.0;
        for( int k = first; k < last; k++ ){
            double xkval = zzShared[kuvpts[k]];
            if (j != k) {
                pj = pj*(particles[idx].xp[2]-xkval)/(xjval-xkval);
            }
        }
        wt[2][j] = pj;
    }

    if (kwpts[2] == 0) {
        first = 2;
        last = 4;
        kwpts[0] = 0;
        kwpts[1] = 0;
    } else if (kwpts[2] < 0) {
        first = 0;
        last = 0;
        kwpts[0] = 0;
        kwpts[1] = 0;
        kwpts[2] = 0;
    } else if (kwpts[2] == 1) {
        first = 1;
        last = 5;
        kwpts[0] = 0;
    } else if (kwpts[2] >= nnz - 2 ){
        first = 0;
        last = 0;
        kwpts[3] = nnz-2;
        kwpts[4] = nnz-2;
        kwpts[5] = nnz-2;
    } else if (kwpts[2] == nnz-3) {
        first = 2;
        last = 4;
        kwpts[4] = nnz-2;
        kwpts[5] = nnz-2;
    } else if (kwpts[2] == nnz-4) {
        first = 1;
        last = 5;
        kwpts[0] = 0;
    } else {
        first = 0;
        last = 6;
    }

    for( int j = first; j < last; j++){
        double xjval = zShared[kwpts[j]];
        double pj = 1.0;
        for( int k = first; k < last; k++ ){
            double xkval = zShared[kwpts[k]];
            if (j != k){
                pj = pj*(particles[idx].xp[2]-xkval)/(xjval-xkval);
            }
        }
        wt[3][j] = pj;
    }

    particles[idx].uf[0] = 0.0;
    particles[idx].uf[1] = 0.0;
    particles[idx].uf[2] = 0.0;

    particles[idx].Tf = 0.0;
    particles[idx].qinf = 0.0;
    for( int k = 0; k < 6; k++ ){
        for( int j = 0; j < 6; j++ ){
            for( int i = 0; i < 6; i++ ){
                const int ix = ijpts[0*6+i];
                const int iy = ijpts[1*6+j];
                const int izuv = kuvpts[k];
                const int izw = kwpts[k];

                particles[idx].uf[0] = particles[idx].uf[0] + fetch_double(uext, (ix+1)+(iy+1)*nx+izuv*ny*nx) * wt[0][i] * wt[1][j] * wt[2][k];
                particles[idx].uf[1] = particles[idx].uf[1] + fetch_double(vext, (ix+1)+(iy+1)*nx+izuv*ny*nx) * wt[0][i] * wt[1][j] * wt[2][k];
                particles[idx].uf[2] = particles[idx].uf[2] + fetch_double(wext, (ix+1)+(iy+1)*nx+izw*ny*nx) * wt[0][i] * wt[1][j] * wt[3][k];
                particles[idx].Tf = particles[idx].Tf + fetch_double(Text, (ix+1)+(iy+1)*nx+izuv*ny*nx) * wt[0][i] * wt[1][j] * wt[2][k];
                particles[idx].qinf = particles[idx].qinf + fetch_double(T2ext, (ix+1)+(iy+1)*nx+izuv*ny*nx) * wt[0][i] * wt[1][j] * wt[2][k];
            }
        }
    }

#ifndef BUILD_CUDA
    }
#endif
}

GLOBAL void GPUUpdateParticles( const int it, const int stage, const double dt, const int pcount, Particle* __restrict__ particles ) {
	const double pi   = 4.0 * atan( 1.0 );
	const double pi2  = 2.0 * pi;
	const double m_s = cParams.Sal / 1000.0 * 4.0 / 3.0 * pi * pow(cParams.radius_mass, 3) * cParams.rhow;

    const double CpaCpp = cParams.Cpa/cParams.Cpp;

    const double zetas[3] = {0.0, -17.0/60.0, -5.0/12.0};
    const double gama[3]  = {8.0/15.0, 5.0/12.0, 3.0/4.0};
    const double g[3] = {0.0, 0.0, cParams.part_grav};

#ifdef BUILD_CUDA
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= pcount ) return;
#else
    for( int idx = 0; idx < pcount; idx++){
#endif

    const int istage = stage - 1;
    if( it == 1 ) {
        for( int j = 0; j < 3; j++ ) {
            particles[idx].vp[j] = particles[idx].uf[j];
        }
        particles[idx].Tp = particles[idx].Tf;
    }

    double diff[3];
    for( int j = 0; j < 3; j++ ) {
        diff[j] = particles[idx].vp[j] - particles[idx].uf[j];
    }
    double diffnorm = sqrt( diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] );
    double Rep = 2.0 * particles[idx].radius * diffnorm / cParams.nuf;
    double Volp = pi2 * 2.0 / 3.0 * ( particles[idx].radius * particles[idx].radius * particles[idx].radius);
    double rhop = ( m_s + Volp * cParams.rhow ) / Volp;
    double taup_i = 18.0 * cParams.rhoa * cParams.nuf / rhop / ( (2.0 * particles[idx].radius) * (2.0 * particles[idx].radius) );

    double corrfac = 1.0 + 0.15 * pow( Rep, 0.687 );
    double Nup = 2.0 + 0.6 * pow( Rep, 0.5 ) * pow( cParams.Pra, 1.0 / 3.0 );
    double Shp = 2.0 + 0.6 * pow( Rep, 0.5 ) * pow( cParams.Sc, 1.0 / 3.0 );

    double TfC = particles[idx].Tf - 273.15;
    double einf = 610.94 * exp( 17.6257 * TfC / ( TfC + 243.04 ) );
    double Lv = ( 25.0 - 0.02274 * 26.0 ) * 100000;
    double Eff_C = 2.0 * cParams.Mw * cParams.Gam / ( cParams.Ru * cParams.rhow * particles[idx].radius * particles[idx].Tp );
    double Eff_S = cParams.Ion * cParams.Os * m_s * cParams.Mw / cParams.Ms / ( Volp * rhop - m_s );
    double estar = einf * exp( cParams.Mw * Lv / cParams.Ru * ( 1.0 / particles[idx].Tf - 1.0 / particles[idx].Tp ) + Eff_C - Eff_S );
    particles[idx].qstar = cParams.Mw / cParams.Ru * estar / particles[idx].Tp / cParams.rhoa;

    double xtmp[3], vtmp[3];
    for( int j = 0; j < 3; j++ ) {
        xtmp[j] = particles[idx].xp[j] + dt * zetas[istage] * particles[idx].xrhs[j];
        vtmp[j] = particles[idx].vp[j] + dt * zetas[istage] * particles[idx].vrhs[j];
    }

    double Tptmp = particles[idx].Tp + dt * zetas[istage] * particles[idx].Tprhs_s;
    Tptmp = Tptmp + dt * zetas[istage] * particles[idx].Tprhs_L;
    double radiustmp = particles[idx].radius + dt * zetas[istage] * particles[idx].radrhs;

    for( int j = 0; j < 3; j++ ) {
        particles[idx].xrhs[j] = particles[idx].vp[j];
    }

    for( int j = 0; j < 3; j++ ) {
        particles[idx].vrhs[j] = corrfac * taup_i * (particles[idx].uf[j] - particles[idx].vp[j]) - g[j];
    }

    if( cParams.Evaporation == 1 ) {
        particles[idx].radrhs = Shp / 9.0 / cParams.Sc * rhop / cParams.rhow * particles[idx].radius * taup_i * ( particles[idx].qinf - particles[idx].qstar );
    } else {
        particles[idx].radrhs = 0.0;
    }

    particles[idx].Tprhs_s = -Nup / 3.0 / cParams.Pra * CpaCpp * rhop / cParams.rhow * taup_i * ( particles[idx].Tp - particles[idx].Tf );
    particles[idx].Tprhs_L = 3.0 * Lv / cParams.Cpp / particles[idx].radius * particles[idx].radrhs;

    for( int j = 0; j < 3; j++ ) {
        particles[idx].xp[j] = xtmp[j] + dt * gama[istage] * particles[idx].xrhs[j];
        particles[idx].vp[j] = vtmp[j] + dt * gama[istage] * particles[idx].vrhs[j];
    }
    particles[idx].Tp = Tptmp + dt * gama[istage] * particles[idx].Tprhs_s;
    particles[idx].Tp = particles[idx].Tp + dt * gama[istage] * particles[idx].Tprhs_L;
    particles[idx].radius = radiustmp + dt * gama[istage] * particles[idx].radrhs;

#ifndef BUILD_CUDA
    }
#endif
}

GLOBAL void GPUUpdateNonperiodic( const double grid_width, const int pcount, Particle* __restrict__ particles ) {
#ifdef BUILD_CUDA
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= pcount ) return;
#else
    for( int idx = 0; idx < pcount; idx++){
#endif

    const double top = grid_width - particles[idx].radius;
    const double bot = 0.0 + particles[idx].radius;

    if( particles[idx].xp[2] > top ){
        particles[idx].xp[2] = top - (particles[idx].xp[2]-top);
        particles[idx].vp[2] = -particles[idx].vp[2];
    }else if( particles[idx].xp[2] < bot ){
        particles[idx].xp[2] = bot + (bot-particles[idx].xp[2]);
        particles[idx].vp[2] = -particles[idx].vp[2];
    }

#ifndef BUILD_CUDA
    }
#endif
}

GLOBAL void GPUUpdatePeriodic( const double grid_width, const double grid_height, const int pcount, Particle* __restrict__ particles ) {
#ifdef BUILD_CUDA
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= pcount ) return;
#else
    for( int idx = 0; idx < pcount; idx++){
#endif

    if( particles[idx].xp[0] > grid_width ){
        particles[idx].xp[0] = particles[idx].xp[0] - grid_width;
    }else if( particles[idx].xp[0] < 0.0 ){
        particles[idx].xp[0] = grid_width + particles[idx].xp[0];
    }

    if( particles[idx].xp[1] > grid_height ){
        particles[idx].xp[1] = particles[idx].xp[1] - grid_height;
    }else if( particles[idx].xp[1] < 0.0 ){
        particles[idx].xp[1] = grid_height + particles[idx].xp[1];
    }

#ifndef BUILD_CUDA
    }
#endif
}

void GPUCalculateStatistics( const int nnz, const double* __restrict__ z, double* __restrict__ partcount_t, double* __restrict__ vpsum_t, double* __restrict__ vpsqrsum_t, const int pcount, Particle* __restrict__ particles ) {
    for( int i = 0; i < pcount; i++ ){
        int kpt = 0;
        for( ; kpt < nnz; kpt++ ){
            if (z[kpt] > particles[i].xp[2]){
                break;
            }
        }
        kpt -= 1;

        for( int idx = 0; idx < nnz; idx++){
            if( kpt == idx ) {
                partcount_t[idx] += 1.0;

                vpsum_t[idx*3+0] += particles[i].vp[0];
                vpsum_t[idx*3+1] += particles[i].vp[1];
                vpsum_t[idx*3+2] += particles[i].vp[2];

                vpsqrsum_t[idx*3+0] += (particles[i].vp[0] * particles[i].vp[0]);
                vpsqrsum_t[idx*3+1] += (particles[i].vp[1] * particles[i].vp[1]);
                vpsqrsum_t[idx*3+2] += (particles[i].vp[2] * particles[i].vp[2]);
            }
        }
    }
}

extern "C" double rand2(int idum, bool reset) {
      const int NTAB = 32;
      static int iv[NTAB], iy = 0, idum2 = 123456789;
      if( reset ) {
          for( int i = 0; i < NTAB; i++ ){
              iv[i] = 0;
          }
          iy = 0;
          idum2 = 123456789;
      }

      int k = 0, IM1 = 2147483563,IM2 = 2147483399,IMM1 = IM1-1,IA1 = 40014,IA2 = 40692,IQ1 = 53668,IQ2 = 52774,IR1 = 12211,IR2 = 3791, NDIV = 1+IMM1/NTAB;
      double AM = 1.0/IM1,EPS = 1.2e-7,RNMX = 1.0-EPS;

      if( idum <= 0 ){
          idum = MAX(-idum,1);
          idum2 = idum;
          for ( int j = NTAB+8; j > 1; j-- ) {
             k = idum/IQ1;
             idum = IA1*(idum-k*IQ1)-k*IR1;
             if (idum < 0) {
                 idum=idum+IM1;
             }
             if (j <= NTAB) {
                 iv[j] = idum;
             }
          }
          iy = iv[1];
      }

      k=idum/IQ1;
      idum=IA1*(idum-k*IQ1)-k*IR1;
      if (idum < 0) {
          idum=idum+IM1;
        }
      k = idum2/IQ2;
      idum2 = IA2*(idum2-k*IQ2)-k*IR2;
      if (idum2 < 0) {
          idum2=idum2+IM2;
      }
      const int j = 1 + iy/NDIV;
      iy = iv[j] - idum2;
      iv[j] = idum;
      if (iy < 1) {
          iy = iy+IMM1;
      }
      return MIN(AM*iy,RNMX);
}

extern "C" GPU* NewGPU(const int particles, const int width, const int height, const int depth, const double fWidth, const double fHeight, const double fDepth, double* z, double* zz, const Parameters* params) {
    GPU* retVal = (GPU*) malloc( sizeof(GPU) );
    SetParameters(retVal, params);

    // Particle Data
    retVal->pCount = particles;
#ifdef BUILD_CUDA
    gpuErrchk( cudaMallocHost((void**)&retVal->hParticles, sizeof(Particle) * particles ) );
#else
    retVal->hParticles = (Particle*) malloc( sizeof(Particle) * particles );
#endif

    // Field Data
    retVal->FieldWidth = fWidth;
    retVal->FieldHeight = fHeight;
    retVal->FieldDepth = fDepth;

    // Grid Data
    retVal->GridWidth = width;
    retVal->GridHeight = height;
    retVal->GridDepth = depth;

    // Statistics
    retVal->hPartCount = (double*) malloc( sizeof(double) * retVal->GridDepth );
    retVal->hVPSum = (double*) malloc( sizeof(double) * retVal->GridDepth * 3);
    retVal->hVPSumSQ = (double*) malloc( sizeof(double) * retVal->GridDepth * 3);

    memset(retVal->hPartCount, 0.0, sizeof(double) * retVal->GridDepth );
    memset(retVal->hVPSum, 0.0, sizeof(double) * retVal->GridDepth * 3);
    memset(retVal->hVPSumSQ, 0.0, sizeof(double) * retVal->GridDepth * 3);

#ifdef BUILD_CUDA
    gpuErrchk( cudaMalloc( (void **)&retVal->dParticles, sizeof(Particle) * retVal->pCount ) );

    gpuErrchk( cudaMalloc( (void **)&retVal->dUext, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth ) );
    cudaTextureObjectHelper(&retVal->tUext, retVal->dUext, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);

    gpuErrchk( cudaMalloc( (void **)&retVal->dVext, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth ) );
    cudaTextureObjectHelper(&retVal->tVext, retVal->dVext, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);

    gpuErrchk( cudaMalloc( (void **)&retVal->dWext, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth ) );
    cudaTextureObjectHelper(&retVal->tWext, retVal->dWext, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);

    gpuErrchk( cudaMalloc( (void **)&retVal->dText, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth ) );
    cudaTextureObjectHelper(&retVal->tText, retVal->dText, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);

    gpuErrchk( cudaMalloc( (void **)&retVal->dQext, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth ) );
    cudaTextureObjectHelper(&retVal->tQext, retVal->dQext, sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);

    gpuErrchk( cudaMalloc( (void **)&retVal->dZ, sizeof(double) * retVal->GridDepth ) );
    gpuErrchk( cudaMalloc( (void **)&retVal->dZZ, sizeof(double) * retVal->GridDepth ) );
    gpuErrchk( cudaMalloc( (void **)&retVal->dPartCount, sizeof(double) * retVal->GridDepth ) );
    gpuErrchk( cudaMalloc( (void **)&retVal->dVPSum, sizeof(double) * retVal->GridDepth * 3 ) );
    gpuErrchk( cudaMalloc( (void **)&retVal->dVPSumSQ, sizeof(double) * retVal->GridDepth * 3 ) );

    gpuErrchk( cudaMemcpy( retVal->dZ, z, sizeof(double) * retVal->GridDepth, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( retVal->dZZ, zz, sizeof(double) * retVal->GridDepth, cudaMemcpyHostToDevice ) );
    #ifdef BUILD_PERFORMANCE_PROFILE
        cudaDeviceSynchronize();
    #endif
#else
    retVal->hUext = (double*) malloc( sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth );
    retVal->hVext = (double*) malloc( sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth );
    retVal->hWext = (double*) malloc( sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth );
    retVal->hText = (double*) malloc( sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth );
    retVal->hQext = (double*) malloc( sizeof(double) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth );
#endif

    retVal->hZ = (double*) malloc( sizeof(double) * retVal->GridDepth );
    retVal->hZZ = (double*) malloc( sizeof(double) * retVal->GridDepth );

    memcpy( retVal->hZ, z, sizeof(double) * retVal->GridDepth );
    memcpy( retVal->hZZ, zz, sizeof(double) * retVal->GridDepth );

    return retVal;
}

extern "C" void ParticleFieldSet( GPU *gpu, double *uext, double *vext, double *wext, double *text, double *qext ) {
#ifdef BUILD_VERIFY_NAN
    std::cout << "Testing for NAN in field:" << std::endl;
    for( int i = 0; i < gpu->GridWidth * gpu->GridHeight * gpu->GridDepth; i++ ){
        if( isnan(uext[i]) ) std::cerr << "UEXT NAN found at index " << i << std::endl;;
        if( isnan(vext[i]) ) std::cerr << "VEXT NAN found at index " << i << std::endl;
        if( isnan(wext[i]) ) std::cerr << "WEXT NAN found at index " << i << std::endl;
        if( isnan(text[i]) ) std::cerr << "TEXT NAN found at index " << i << std::endl;
        if( isnan(qext[i]) ) std::cerr << "QEXT NAN found at index " << i << std::endl;
    }
    std::cout << "\tComplete" << std::endl;
#endif // BUILD_VERIFY_NAN

#ifdef BUILD_CUDA
    gpuErrchk( cudaMemcpy( gpu->dUext, uext, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( gpu->dVext, vext, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( gpu->dWext, wext, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( gpu->dText, text, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( gpu->dQext, qext, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice ) );
    #ifdef BUILD_PERFORMANCE_PROFILE
        cudaDeviceSynchronize();
    #endif
#else
    memcpy( gpu->hUext, uext, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth );
    memcpy( gpu->hVext, vext, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth );
    memcpy( gpu->hWext, wext, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth );
    memcpy( gpu->hText, text, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth );
    memcpy( gpu->hQext, qext, sizeof(double) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth );
#endif
}

extern "C" void ParticleAdd( GPU *gpu, const int position, const Particle *input ){
    assert(position >= 0 && position < gpu->pCount);
    memcpy(&gpu->hParticles[position], input, sizeof(Particle));
}

extern "C" Particle ParticleGet( GPU *gpu, const int position ){
    assert(position >= 0 && position < gpu->pCount);
    return gpu->hParticles[position];
}

extern "C" void ParticleUpload( GPU *gpu ){
#ifdef BUILD_CUDA
    gpuErrchk( cudaMemcpy( gpu->dParticles, gpu->hParticles, sizeof(Particle) * gpu->pCount, cudaMemcpyHostToDevice ) );
#endif
}

extern "C" void ParticleInit( GPU* gpu, const int particles, const Particle* input ){
    if( gpu->pCount != particles ){
        gpu->pCount = particles;
#ifdef BUILD_CUDA
        gpuErrchk( cudaFree( gpu->dParticles ) );
        gpuErrchk( cudaMalloc( (void **)&gpu->dParticles, sizeof(Particle) * particles ) );
#else
        free( gpu->hParticles );
        gpu->hParticles = (Particle*) malloc( sizeof(Particle) * gpu->pCount );
#endif
    }

#ifdef BUILD_CUDA
    gpuErrchk( cudaMemcpy( gpu->dParticles, input, sizeof(Particle) * particles, cudaMemcpyHostToDevice ) );
#else
    memcpy( gpu->hParticles, input, sizeof(Particle) * gpu->pCount);
#endif
}

extern "C" void ParticleGenerate(GPU* gpu, const int processors, const int particles, const int seed, const double temperature, const double xmin, const double xmax, const double ymin, const double ymax, const double zl, const double delta_vis, const double radius, const double qinfp){
    gpu->pCount = particles;

    bool reset = true;
    int currentProcessor = 1;
    const int particles_per_processor = particles / processors;

    gpu->hParticles = (Particle*) malloc( sizeof(Particle) * particles );
    for( size_t i = 0; i < particles; i++ ){
        if( i >= currentProcessor * particles_per_processor) {
            reset = true;
            currentProcessor++;
        }

        double random = 0.0;
        if( reset ) {
            random = rand2(seed, true);
            reset = false;
        }else{
            random = rand2(seed, false);
        }
        const double x = random*(xmax-xmin) + xmin;
        const double y = rand2(seed, false)*(ymax-ymin) + ymin;
        const double z = rand2(seed, false)*(zl-2.0*delta_vis) + delta_vis;

        gpu->hParticles[i].xp[0] = x; gpu->hParticles[i].xp[1] = y; gpu->hParticles[i].xp[2] = z;
        gpu->hParticles[i].vp[0] = 0.0; gpu->hParticles[i].vp[1] = 0.0; gpu->hParticles[i].vp[2] = 0.0;
        gpu->hParticles[i].Tp = temperature;
        gpu->hParticles[i].radius = radius;
        gpu->hParticles[i].uf[0] = 0.0; gpu->hParticles[i].uf[1] = 0.0; gpu->hParticles[i].uf[2] = 0.0;
        gpu->hParticles[i].qinf = qinfp;
        gpu->hParticles[i].xrhs[0] = 0.0; gpu->hParticles[i].xrhs[1] = 0.0; gpu->hParticles[i].xrhs[2] = 0.0;
        gpu->hParticles[i].vrhs[0] = 0.0; gpu->hParticles[i].vrhs[1] = 0.0; gpu->hParticles[i].vrhs[2] = 0.0;
        gpu->hParticles[i].Tprhs_s = 0.0;
        gpu->hParticles[i].Tprhs_L = 0.0;
        gpu->hParticles[i].radrhs = 0.0;
    }

#ifdef BUILD_CUDA
    gpuErrchk( cudaMalloc( (void **)&gpu->dParticles, sizeof(Particle) * particles) );
    gpuErrchk( cudaMemcpy(gpu->dParticles, gpu->hParticles, sizeof(Particle) * particles, cudaMemcpyHostToDevice) );
#endif
}

extern "C" void ParticleInterpolate( GPU *gpu, const double dx, const double dy ) {
#ifdef BUILD_CUDA
    const unsigned int blocks = std::ceil(gpu->pCount / (float)CUDA_BLOCK_THREADS);
    if( gpu->mParameters.LinearInterpolation == 1 ) {
        GPUFieldInterpolateLinear<<< blocks, CUDA_BLOCK_THREADS, ((gpu->GridDepth*4)+4)*sizeof(double) >>> ( gpu->GridWidth, gpu->GridHeight, dx, dy, gpu->GridDepth, gpu->dZ, gpu->dZZ, gpu->tUext, gpu->tVext, gpu->tWext, gpu->tText, gpu->tQext, gpu->pCount, gpu->dParticles);
    }else{
        GPUFieldInterpolate<<< blocks, CUDA_BLOCK_THREADS, gpu->GridDepth*2*sizeof(double) >>> ( gpu->GridWidth, gpu->GridHeight, dx, dy, gpu->GridDepth, gpu->dZ, gpu->dZZ, gpu->tUext, gpu->tVext, gpu->tWext, gpu->tText, gpu->tQext, gpu->pCount, gpu->dParticles);
    }
    gpuErrchk( cudaPeekAtLastError() );
#else
    GPUFieldInterpolate( gpu->GridWidth, gpu->GridHeight, dx, dy, gpu->GridDepth, gpu->hZ, gpu->hZZ, gpu->hUext, gpu->hVext, gpu->hWext, gpu->hText, gpu->hQext, gpu->pCount, gpu->hParticles);
#endif
}

extern "C" void ParticleStep( GPU *gpu, const int it, const int istage, const double dt ) {
#ifdef BUILD_CUDA
    const unsigned int blocks = std::ceil(gpu->pCount / (float)CUDA_BLOCK_THREADS);
    GPUUpdateParticles<<< blocks, CUDA_BLOCK_THREADS >>> (it, istage, dt, gpu->pCount, gpu->dParticles);
    gpuErrchk( cudaPeekAtLastError() );
#else
    GPUUpdateParticles(it, istage, dt, gpu->pCount, gpu->hParticles);
#endif
}

extern "C" void ParticleUpdateNonPeriodic( GPU *gpu ) {
#ifdef BUILD_CUDA
    const unsigned int blocks = std::ceil(gpu->pCount / (float)CUDA_BLOCK_THREADS);
    GPUUpdateNonperiodic<<< blocks, CUDA_BLOCK_THREADS >>> (gpu->FieldWidth, gpu->pCount, gpu->dParticles);
    gpuErrchk( cudaPeekAtLastError() );
#else
    GPUUpdateNonperiodic(gpu->FieldWidth, gpu->pCount, gpu->hParticles);
#endif
}

extern "C" void ParticleUpdatePeriodic( GPU *gpu ) {
#ifdef BUILD_CUDA
    const unsigned int blocks = std::ceil(gpu->pCount / (float)CUDA_BLOCK_THREADS);
    GPUUpdatePeriodic<<< blocks, CUDA_BLOCK_THREADS >>> (gpu->FieldWidth, gpu->FieldHeight, gpu->pCount, gpu->dParticles);
    gpuErrchk( cudaPeekAtLastError() );
#else
    GPUUpdatePeriodic(gpu->FieldWidth, gpu->FieldHeight, gpu->pCount, gpu->hParticles);
#endif
}

extern "C" void ParticleCalculateStatistics( GPU *gpu, const double dx, const double dy ) {
    memset(gpu->hPartCount, 0.0, sizeof(double) * gpu->GridDepth );
    memset(gpu->hVPSum, 0.0, sizeof(double) * gpu->GridDepth * 3);
    memset(gpu->hVPSumSQ, 0.0, sizeof(double) * gpu->GridDepth * 3);

#ifdef BUILD_CUDA
    ParticleDownloadHost(gpu);
#endif

    GPUCalculateStatistics( gpu->GridDepth, gpu->hZ, gpu->hPartCount, gpu->hVPSum, gpu->hVPSumSQ, gpu->pCount, gpu->hParticles);
}


extern "C" void ParticleDownloadHost( GPU *gpu ) {
#ifdef BUILD_CUDA
    gpuErrchk( cudaMemcpy(gpu->hParticles, gpu->dParticles, sizeof(Particle) * gpu->pCount, cudaMemcpyDeviceToHost) );
#endif
}

extern "C" Particle* ParticleDownload( GPU *gpu ) {
#ifdef BUILD_CUDA
    gpuErrchk( cudaMemcpy(gpu->hParticles, gpu->dParticles, sizeof(Particle) * gpu->pCount, cudaMemcpyDeviceToHost) );
#endif
    return gpu->hParticles;
}

void ParticleWrite( GPU* gpu ){
    static int call = 0;
    static char buffer[80];
    sprintf(buffer, "c-particle-%d.dat", call);

    FILE *write_ptr = fopen(buffer,"wb");
    call += 1;

    fwrite(&gpu->pCount, sizeof(unsigned int), 1, write_ptr);
    for( int i = 0; i < gpu->pCount; i++ ){
        fwrite(&gpu->hParticles[i], sizeof(Particle), 1, write_ptr);
    }

    fclose(write_ptr);
}

GPU* ParticleRead(const char * path){
    FILE *data = fopen(path,"rb");

    unsigned int particles = 0;
    fread(&particles, sizeof(unsigned int), 1, data);

    double z[1], zz[1];

    Parameters params;
    GPU *retVal = NewGPU(particles, 0, 0, 0, 0.0, 0.0, 0.0, &z[0], &zz[0], &params );
    for( int i = 0; i < retVal->pCount; i++ ){
        fread(&retVal->hParticles[i], sizeof(Particle), 1, data);
    }

    fclose(data);
    return retVal;
}

void PrintFreeMemory(){
#ifdef BUILD_CUDA
    size_t free_byte, total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
#endif
}

void ParticleFillStatistics(GPU* gpu, double* partCount, double* vSum, double* vSumSQ){
    for( size_t i = 0; i < gpu->GridDepth; i++ ){
        partCount[i] = gpu->hPartCount[i];
        vSum[i] = gpu->hVPSum[i];
        vSumSQ[i] = gpu->hVPSumSQ[i];
    }
}

// Particle Functions
std::ostream& operator<< (std::ostream& stream, const Particle& p) {
    stream << std::fixed << std::setprecision(12);
    stream << "Particle[" << p.procidx << ":" << p.pidx << "]" << std::endl;
    stream << "\t VP: " << p.vp[0] << ", " << p.vp[1] << ", " << p.vp[2] << std::endl;
	stream << "\t XP: " << p.xp[0] << ", " << p.xp[1] << ", " << p.xp[2] << std::endl;
	stream << "\t UF: " << p.uf[0] << ", " << p.uf[1] << ", " << p.uf[2] << std::endl;
	stream << "\t XRHS: " << p.xrhs[0] << ", " << p.xrhs[1] << ", " << p.xrhs[2] << std::endl;
	stream << "\t VRHS: " << p.vrhs[0] << ", " << p.vrhs[1] << ", " << p.vrhs[2] << std::endl;
	stream << "\t Tp: " << p.Tp << std::endl;
	stream << "\t Tprhs_s: " << p.Tprhs_s << std::endl;
	stream << "\t Tprhs_L: " << p.Tprhs_L << std::endl;
	stream << "\t Tf: " << p.Tf << std::endl;
	stream << "\t Radius: " << p.radius << std::endl;
	stream << "\t Radius RHS: " << p.radrhs << std::endl;
	stream << "\t QInf: " << p.qinf << std::endl;
	stream << "\t QStar: " << p.qstar << std::endl;

    return stream;
}

// Helper Functions
const std::vector<double> ReadDoubleArray(const std::string& path){
    std::vector<double> retVal;

    std::ifstream iStream(path.c_str(), std::ifstream::in | std::ifstream::binary);
    if( iStream.fail() ){
        std::cerr << "Unable to open " << path << " to read from.";
        return retVal;
    }

    unsigned int size = 0;
    iStream >> size;

    retVal.resize(size);
    for( unsigned int i = 0; i < size; i++ ){
        iStream >> retVal[i];
    }
    iStream.close();

    return retVal;
}

void WriteDoubleArray(const std::string& path, const std::vector<double>& array){
    std::ofstream oStream(path.c_str(), std::ofstream::out | std::ofstream::binary );
    if( oStream.fail() ){
        std::cerr << "Unable to open " << path << " to write to.";
        return;
    }

    oStream << array.size();
    for( unsigned int i = 0; i < array.size(); i++ ){
        oStream << array[i];
    }
    oStream.close();
}

// Test Helper Functions
void SetParameters(GPU* gpu, const Parameters* params) {
    memcpy( &gpu->mParameters, params, sizeof(Parameters) );

#ifdef BUILD_CUDA
    gpuErrchk( cudaMemcpyToSymbol(cParams, &gpu->mParameters, sizeof(Parameters)) );
#else
    memcpy( &cParams, &gpu->mParameters, sizeof(Parameters) );
#endif
}