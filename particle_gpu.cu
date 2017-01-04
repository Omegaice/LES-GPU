#include "particle_gpu.h"
#include "stdio.h"
#include "assert.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

extern "C" int gpudevices(){
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    return nDevices;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ double FortranIndex(const double* array, const int width, const int height, const int depth, const int ofx, const int ofy, const int ofz, const int x, const int y, const int z) {
    const int xActual = x+ofx-1, yActual = y+ofy-1, zActual = z+ofz-1;
    const int pos = zActual+yActual*(depth-1)+xActual*(width-1)*(depth-1);
    return array[pos];
}

__global__ void GPUFieldInterpolate( const int nx, const int ny, const double dx, const double dy, const int nnz, const double *z, const double *zz, const int offsetX, const int offsetY, const int offsetZ, const double *uext, const double *vext, const double *wext, const double *Text, const double *T2ext, const int pcount, Particle* particles ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= pcount ) return;

    int ijpts[2][6];
    ijpts[0][2] = floor(particles[idx].xp[0]/dx) + 1;
    ijpts[1][2] = floor(particles[idx].xp[1]/dy) + 1;

    ijpts[0][1] = ijpts[0][2]-1;
    ijpts[0][0] = ijpts[0][1]-1;
    ijpts[0][3] = ijpts[0][2]+1;
    ijpts[0][4] = ijpts[0][3]+1;
    ijpts[0][5] = ijpts[0][4]+1;

    ijpts[1][1] = ijpts[1][2]-1;
    ijpts[1][0] = ijpts[1][1]-1;
    ijpts[1][3] = ijpts[1][2]+1;
    ijpts[1][4] = ijpts[1][3]+1;
    ijpts[1][5] = ijpts[1][4]+1;

    int kuvpts[6];
    for( int iz = 0; iz < nnz+1; iz++ ){
        if (zz[iz] > particles[idx].xp[2]){
            kuvpts[2] = iz-1;
            break;
        }
    }

    kuvpts[3] = kuvpts[2]+1;
    kuvpts[4] = kuvpts[3]+1;
    kuvpts[5] = kuvpts[4]+1;
    kuvpts[1] = kuvpts[2]-1;
    kuvpts[0] = kuvpts[1]-1;

    int kwpts[6];
    for( int iz = 0; iz < nnz+1; iz++ ){
        if (z[iz] > particles[idx].xp[2]) {
            kwpts[2] = iz-1;
            break;
        }
    }

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
            double xjval = dxvec[iz]*(ijpts[iz][j]-1);
            double pj = 1.0;
            for( int k = 0; k < 6; k++ ){
                double xkval = dxvec[iz]*(ijpts[iz][k]-1);
                if (j != k) {
                    pj = pj*(particles[idx].xp[iz]-xkval)/(xjval-xkval);
                }
            }
            wt[iz][j] = pj;
        }
    }

    int first, last;
    if (kuvpts[2] == 1) {
        first = 3;
        last = 4;
        kuvpts[0] = 1;
        kuvpts[1] = 1;
    } else if (kuvpts[2] == 0) {
        first = 4;
        last = 5;
        kuvpts[0] = 1;
        kuvpts[1] = 1;
        kuvpts[2] = 1;
    } else if (kuvpts[2] < 0) {
        first = 0;
        last = 0;
    } else if (kuvpts[2] == 2) {
        first = 2;
        last = 5;
    } else if (kuvpts[2] == nnz) {
        first = 2;
        last = 3;
        kuvpts[3] = nnz;
        kuvpts[4] = nnz;
        kuvpts[5] = nnz;
    } else if (kuvpts[2] > nnz) {
        first = 0;
        last = 0;
    } else if (kuvpts[2] == nnz-1) {
        first = 3;
        last = 4;
        kuvpts[4] = nnz;
        kuvpts[5] = nnz;
    } else if (kuvpts[2] == nnz-2) {
        first = 2;
        last = 5;
    } else {
        first = 1;
        last = 6;
    }

    for( int j = first-1; j < last; j++){
        double xjval = zz[kuvpts[j]];
        double pj = 1.0;
        for( int k = first-1; k < last; k++ ){
            double xkval = zz[kuvpts[k]];
            if (j != k) {
                pj = pj*(particles[idx].xp[2]-xkval)/(xjval-xkval);
            }
        }
        wt[2][j] = pj;
    }

    if (kwpts[2] == 0) {
        first = 3;
        last = 4;
    } else if (kwpts[2] < 0) {
        first = 0;
        last = 0;
    } else if (kwpts[2] == 1) {
        first = 2;
        last = 5;
    } else if (kwpts[2] == nnz-1) {
        first = 3;
        last = 4;
    } else if (kwpts[2] > nnz) {
        first = 0;
        last = 0;
    } else if (kwpts[2] == nnz-2) {
        first = 2;
        last = 5;
    } else {
        first = 1;
        last = 6;
    }

    for( int j = first-1; j < last; j++){
        double xjval = z[kwpts[j]];
        double pj = 1.0;
        for( int k = first-1; k < last; k++ ){
            double xkval = z[kwpts[k]];
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
                const int ix = ijpts[0][i];
                const int iy = ijpts[1][j];
                const int izuv = kuvpts[k];
                const int izw = kwpts[k];

                particles[idx].uf[0] = particles[idx].uf[0]+FortranIndex(uext,ny,nnz,nx,offsetZ,offsetY,offsetX,izuv,iy,ix)*wt[0][i]*wt[1][j]*wt[2][k];
                particles[idx].uf[1] = particles[idx].uf[1]+FortranIndex(vext,ny,nnz,nx,offsetZ,offsetY,offsetX,izuv,iy,ix)*wt[0][i]*wt[1][j]*wt[2][k];
                particles[idx].uf[2] = particles[idx].uf[2]+FortranIndex(wext,ny,nnz,nx,offsetZ,offsetY,offsetX,izw,iy,ix)*wt[0][i]*wt[1][j]*wt[3][k];
                particles[idx].Tf = particles[idx].Tf+FortranIndex(Text,ny,nnz,nx,offsetZ,offsetY,offsetX,izuv,iy,ix)*wt[0][i]*wt[1][j]*wt[2][k];
                particles[idx].qinf = particles[idx].qinf+FortranIndex(T2ext,ny,nnz,nx,offsetZ,offsetY,offsetX,izuv,iy,ix)*wt[0][i]*wt[1][j]*wt[2][k];
            }
        }
    }
}

__global__ void GPUUpdateParticles( const int it, const int stage, const double dt, const int pcount, Particle* particles, double* statBuffer ) {
    const double ievap = 1;

	const double Gam = 7.28 * std::pow( 10.0, -2 );
	const double Ion = 2.0;
	const double Os = 1.093;
	const double rhoa = 1.1;
	const double rhow = 1000.0;
	const double nuf  = 1.537e-5;
	const double pi   = 4.0 * std::atan( 1.0 );
	const double pi2  = 2.0 * pi;
	const double Sal = 34.0;
	const double radius_mass = 40.0e-6;
	const double m_s = Sal / 1000.0 * 4.0 / 3.0 * pi * std::pow(radius_mass, 3) * rhow;
    const double Pra = 0.715;
    const double Sc = 0.615;
    const double Mw = 0.018015;
    const double Ru = 8.3144;
    const double Ms = 0.05844;
    const double Cpa = 1006.0;
    const double Cpp = 4179.0;
    const double CpaCpp = Cpa/Cpp;
    const double part_grav = 0.0;

    const double zetas[3] = {0.0, -17.0/60.0, -5.0/12.0};
    const double gama[3]  = {8.0/15.0, 5.0/12.0, 3.0/4.0};
    const double g[3] = {0.0, 0.0, part_grav};

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= pcount ) return;

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
    double diffnorm = std::sqrt( diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] );
    double Rep = 2.0 * particles[idx].radius * diffnorm / nuf;
    statBuffer[idx*4+0] = Rep;

    double Volp = pi2 * 2.0 / 3.0 * ( particles[idx].radius * particles[idx].radius * particles[idx].radius);
    double rhop = ( m_s + Volp * rhow ) / Volp;
    statBuffer[idx*4+1] = Volp*rhop;
    statBuffer[idx*4+2] = Volp*rhow;
    statBuffer[idx*4+3] = Volp;

    double taup_i = 18.0 * rhoa * nuf / rhop / ( (2.0 * particles[idx].radius) * (2.0 * particles[idx].radius) );
    double corrfac = 1.0 + 0.15 * pow( Rep, 0.687 );
    double Nup = 2.0 + 0.6 * pow( Rep, 0.5 ) * pow( Pra, 1.0 / 3.0 );
    double Shp = 2.0 + 0.6 * pow( Rep, 0.5 ) * pow( Sc, 1.0 / 3.0 );

    double TfC = particles[idx].Tf - 273.15;
    double einf = 610.94 * exp( 17.6257 * TfC / ( TfC + 243.04 ) );
    double Lv = ( 25.0 - 0.02274 * 26.0 ) * 100000;
    double Eff_C = 2.0 * Mw * Gam / ( Ru * rhow * particles[idx].radius * particles[idx].Tp );
    double Eff_S = Ion * Os * m_s * Mw / Ms / ( Volp * rhop - m_s );
    double estar = einf * exp( Mw * Lv / Ru * ( 1.0 / particles[idx].Tf - 1.0 / particles[idx].Tp ) + Eff_C - Eff_S );
    particles[idx].qstar = Mw / Ru * estar / particles[idx].Tp / rhoa;

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

    if( ievap == 1 ) {
        particles[idx].radrhs = Shp / 9.0 / Sc * rhop / rhow * particles[idx].radius * taup_i * ( particles[idx].qinf - particles[idx].qstar );
    } else {
        particles[idx].radrhs = 0.0;
    }

    particles[idx].Tprhs_s = -Nup / 3.0 / Pra * CpaCpp * rhop / rhow * taup_i * ( particles[idx].Tp - particles[idx].Tf );
    particles[idx].Tprhs_L = 3.0 * Lv / Cpp / particles[idx].radius * particles[idx].radrhs;

    for( int j = 0; j < 3; j++ ) {
        particles[idx].xp[j] = xtmp[j] + dt * gama[istage] * particles[idx].xrhs[j];
        particles[idx].vp[j] = vtmp[j] + dt * gama[istage] * particles[idx].vrhs[j];
    }
    particles[idx].Tp = Tptmp + dt * gama[istage] * particles[idx].Tprhs_s;
    particles[idx].Tp = particles[idx].Tp + dt * gama[istage] * particles[idx].Tprhs_L;
    particles[idx].radius = radiustmp + dt * gama[istage] * particles[idx].radrhs;
}

__global__ void GPUUpdateNonperiodic( const double grid_width, const double delta_vis, const int pcount, Particle* particles ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= pcount ) return;

    const double top = grid_width - delta_vis;
    const double bot = 0.0 + delta_vis;

    if( particles[idx].xp[2] > top ){
        particles[idx].xp[2] = top - (particles[idx].xp[2]-top);
        particles[idx].vp[2] = -particles[idx].vp[2];
    }else if( particles[idx].xp[2] < bot ){
        particles[idx].xp[2] = bot + (bot-particles[idx].xp[2]);
        particles[idx].vp[2] = -particles[idx].vp[2];
    }
}

__global__ void GPUUpdatePeriodic( const double grid_width, const double grid_height, const int pcount, Particle* particles ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= pcount ) return;

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
}

__global__ void GPUCalculateStatistics( const int nx, const int ny, const int nnz, const double dx, const double dy, const double *z, const double *zz, double *partcount_t, const int pcount, Particle* particles ) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= nnz ) return;

    partcount_t[idx] = 0.0;
    for( int i = 0; i < pcount; i++ ){
        int kpt = -1;
        for( int iz = 0; iz < nnz+1; iz++ ){
            if (z[iz] > particles[i].xp[2]){
                kpt = iz-1;
                break;
            }
        }
        if( kpt == idx ) partcount_t[idx] += 1.0;
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

extern "C" GPU* NewGPU(const int particles, const int width, const int height, const int depth, const int zsize, const double fWidth, const double fHeight, const double fDepth, const double fVis) {
    GPU* retVal = (GPU*) malloc( sizeof(GPU) );

    // Particle Data
    retVal->pCount = particles;
    retVal->hParticles = (Particle*) malloc( sizeof(Particle) * particles );
    gpuErrchk( cudaMalloc( (void **)&retVal->dParticles, sizeof(Particle) * retVal->pCount ) );

    // Field Data
    retVal->FieldWidth = fWidth;
    retVal->FieldHeight = fHeight;
    retVal->FieldDepth = fDepth;
    retVal->FieldVis = fVis;

    // Grid Data
    retVal->GridWidth = width;
    retVal->GridWidthHalo = width + 6;
    retVal->GridHeight = height;
    retVal->GridHeightHalo = height + 6;
    retVal->GridDepth = depth;
    retVal->GridDepthHalo = depth + 2;
    retVal->ZSize = zsize;

    gpuErrchk( cudaMalloc( (void **)&retVal->dUext, sizeof(double) * retVal->GridWidthHalo * retVal->GridHeightHalo * retVal->GridDepthHalo ) );
    gpuErrchk( cudaMalloc( (void **)&retVal->dVext, sizeof(double) * retVal->GridWidthHalo * retVal->GridHeightHalo * retVal->GridDepthHalo ) );
    gpuErrchk( cudaMalloc( (void **)&retVal->dWext, sizeof(double) * retVal->GridWidthHalo * retVal->GridHeightHalo * retVal->GridDepthHalo ) );
    gpuErrchk( cudaMalloc( (void **)&retVal->dText, sizeof(double) * retVal->GridWidthHalo * retVal->GridHeightHalo * retVal->GridDepthHalo ) );
    gpuErrchk( cudaMalloc( (void **)&retVal->dQext, sizeof(double) * retVal->GridWidthHalo * retVal->GridHeightHalo * retVal->GridDepthHalo ) );

    gpuErrchk( cudaMalloc( (void **)&retVal->dZ, sizeof(double) * retVal->ZSize ) );
    gpuErrchk( cudaMalloc( (void **)&retVal->dZZ, sizeof(double) * retVal->ZSize ) );

    // Statistics
    retVal->hStat = (double*) malloc( sizeof(double) * retVal->pCount * 4 );
    gpuErrchk( cudaMalloc( (void **)&retVal->dStat, sizeof(double) * retVal->pCount * 4 ) );

    retVal->hPartCount = (double*) malloc( sizeof(double) * retVal->GridDepthHalo );
    gpuErrchk( cudaMalloc( (void **)&retVal->dPartCount, sizeof(double) * retVal->GridDepthHalo ) );

    return retVal;
}

extern "C" void ParticleFieldSet( GPU *gpu, double *uext, double *vext, double *wext, double *text, double *qext, double *z, double *zz ) {
    gpuErrchk( cudaMemcpy( gpu->dUext, uext, sizeof(double) * gpu->GridWidthHalo * gpu->GridHeightHalo * gpu->GridDepthHalo, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( gpu->dVext, vext, sizeof(double) * gpu->GridWidthHalo * gpu->GridHeightHalo * gpu->GridDepthHalo, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( gpu->dWext, wext, sizeof(double) * gpu->GridWidthHalo * gpu->GridHeightHalo * gpu->GridDepthHalo, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( gpu->dText, text, sizeof(double) * gpu->GridWidthHalo * gpu->GridHeightHalo * gpu->GridDepthHalo, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( gpu->dQext, qext, sizeof(double) * gpu->GridWidthHalo * gpu->GridHeightHalo * gpu->GridDepthHalo, cudaMemcpyHostToDevice ) );

    gpuErrchk( cudaMemcpy( gpu->dZ, z, sizeof(double) * gpu->ZSize, cudaMemcpyHostToDevice ) );
    gpuErrchk( cudaMemcpy( gpu->dZZ, zz, sizeof(double) * gpu->ZSize, cudaMemcpyHostToDevice ) );
}

extern "C" void ParticleAdd( GPU *gpu, const int position, const Particle *input ){
    assert(position >= 0 && position < gpu->pCount);
    assert(input->uf[0] < 20 && input->uf[1] < 20 && input->uf[2] < 20 );
    memcpy(&gpu->hParticles[position], input, sizeof(Particle));
}

extern "C" Particle ParticleGet( GPU *gpu, const int position ){
    assert(position >= 0 && position < gpu->pCount);
    return gpu->hParticles[position];
}

extern "C" void ParticleUpload( GPU *gpu ){
    gpuErrchk( cudaMemcpy( gpu->dParticles, gpu->hParticles, sizeof(Particle) * gpu->pCount, cudaMemcpyHostToDevice ) );
}

extern "C" void ParticleInit( GPU* gpu, const int particles, const Particle* input ){
    gpu->pCount = particles;
    gpuErrchk( cudaMalloc( (void **)&gpu->dParticles, sizeof(Particle) * particles ) );
    gpuErrchk( cudaMemcpy( gpu->dParticles, input, sizeof(Particle) * particles, cudaMemcpyHostToDevice ) );
}

extern "C" void ParticleGenerate(GPU* gpu, const int processors, const int particles, const int seed, const double temperature, const double xmin, const double xmax, const double ymin, const double ymax, const double zl, const double delta_vis, const double radius, const double qinfp){
    gpu->pCount = particles;
    gpuErrchk( cudaMalloc( (void **)&gpu->dParticles, sizeof(Particle) * particles) );

    bool reset = true;
    int currentProcessor = 1;
    const int particles_per_processor = particles / processors;

    Particle *hParticles = (Particle*) malloc( sizeof(Particle) * particles );
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

        hParticles[i].xp[0] = x; hParticles[i].xp[1] = y; hParticles[i].xp[2] = z;
        hParticles[i].vp[0] = 0.0; hParticles[i].vp[1] = 0.0; hParticles[i].vp[2] = 0.0;
        hParticles[i].Tp = temperature;
        hParticles[i].radius = radius;
        hParticles[i].uf[0] = 0.0; hParticles[i].uf[1] = 0.0; hParticles[i].uf[2] = 0.0;
        hParticles[i].qinf = qinfp;
        hParticles[i].xrhs[0] = 0.0; hParticles[i].xrhs[1] = 0.0; hParticles[i].xrhs[2] = 0.0;
        hParticles[i].vrhs[0] = 0.0; hParticles[i].vrhs[1] = 0.0; hParticles[i].vrhs[2] = 0.0;
        hParticles[i].Tprhs_s = 0.0;
        hParticles[i].Tprhs_L = 0.0;
        hParticles[i].radrhs = 0.0;
    }

    gpuErrchk( cudaMemcpy(gpu->dParticles, hParticles, sizeof(Particle) * particles, cudaMemcpyHostToDevice) );
    free(hParticles);
}

extern "C" void ParticleInterpolate( GPU *gpu, const double dx, const double dy, const int nnz, const int offsetX, const int offsetY, const int offsetZ ) {
    GPUFieldInterpolate<<< (gpu->pCount / 32) + 1, 32 >>> ( gpu->GridWidthHalo, gpu->GridHeightHalo, dx, dy, gpu->GridDepthHalo, gpu->dZ, gpu->dZZ, 1-offsetX, 1-offsetY, 1-offsetZ, gpu->dUext, gpu->dVext, gpu->dWext, gpu->dText, gpu->dQext, gpu->pCount, gpu->dParticles);
    gpuErrchk( cudaPeekAtLastError() );
}

extern "C" void ParticleStep( GPU *gpu, const int it, const int istage, const double dt ) {
    GPUUpdateParticles<<< (gpu->pCount / 32) + 1, 32 >>> (it, istage, dt, gpu->pCount, gpu->dParticles, gpu->dStat);

    // Get Statistics
    printf("Particles: %d\n", gpu->pCount);
    gpuErrchk( cudaMemcpy( gpu->hStat, gpu->dStat, sizeof(double) * gpu->pCount, cudaMemcpyDeviceToHost ) );

    gpu->RepAverage = 0.0;
    gpu->Phip = 0.0; gpu->Phiw = 0.0; gpu->Phiv = 0.0;
    for( size_t i = 0; i < gpu->pCount; i++ ){
        gpu->RepAverage += gpu->hStat[i*4+0];
        gpu->Phip += gpu->hStat[i*4+1];
        gpu->Phiw += gpu->hStat[i*4+2];
        gpu->Phiv += gpu->hStat[i*4+3];
    }
    gpu->RepAverage /= gpu->pCount;

    gpu->Phip = gpu->Phip/gpu->FieldWidth/gpu->FieldHeight/gpu->FieldDepth/1.1;
    gpu->Phiw = gpu->Phiw/gpu->FieldWidth/gpu->FieldHeight/gpu->FieldDepth/1.1;
    gpu->Phiv = gpu->Phiv/gpu->FieldWidth/gpu->FieldHeight/gpu->FieldDepth;

    // Check for errors
    gpuErrchk( cudaPeekAtLastError() );
}

extern "C" void ParticleUpdateNonPeriodic( GPU *gpu ) {
    GPUUpdateNonperiodic<<< (gpu->pCount / 32) + 1, 32 >>> (gpu->FieldWidth, gpu->FieldVis, gpu->pCount, gpu->dParticles);
    gpuErrchk( cudaPeekAtLastError() );
}

extern "C" void ParticleUpdatePeriodic( GPU *gpu ) {
    GPUUpdatePeriodic<<< (gpu->pCount / 32) + 1, 32 >>> (gpu->FieldWidth, gpu->FieldHeight, gpu->pCount, gpu->dParticles);
    gpuErrchk( cudaPeekAtLastError() );
}

extern "C" void ParticleCalculateStatistics( GPU *gpu, const double dx, const double dy, const int nnz, const int nny, const int nnx, double *partcount_t, double* dzw ) {
    GPUCalculateStatistics<<< (gpu->GridDepthHalo / 32) + 1, 32 >>> ( gpu->GridWidth, gpu->GridHeight, gpu->GridDepthHalo, dx, dy, gpu->dZ, gpu->dZZ, gpu->dPartCount, gpu->pCount, gpu->dParticles);
    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk( cudaMemcpy(gpu->hPartCount, gpu->dPartCount, sizeof(double) * gpu->GridDepthHalo, cudaMemcpyDeviceToHost) );
    for( int i = 0; i < gpu->GridDepthHalo; i++ ){
        printf("Z[%d] Count: %f\n", i, gpu->hPartCount[i]);
    }
    printf("\n");
}

extern "C" void ParticleDownloadHost( GPU *gpu ) {
    gpuErrchk( cudaMemcpy(gpu->hParticles, gpu->dParticles, sizeof(Particle) * gpu->pCount, cudaMemcpyDeviceToHost) );
}

extern "C" Particle* ParticleDownload( GPU *gpu ) {
    Particle *result = (Particle*) malloc( sizeof(Particle) * gpu->pCount);
    gpuErrchk( cudaMemcpy(result, gpu->dParticles, sizeof(Particle) * gpu->pCount, cudaMemcpyDeviceToHost) );
    return result;
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

GPU* ParticleRead(char * path){
    FILE *data = fopen(path,"rb");
    GPU *retVal = (GPU*) malloc( sizeof(GPU) );

    fread(&retVal->pCount, sizeof(unsigned int), 1, data);
    retVal->hParticles = (Particle*) malloc( sizeof(Particle) * retVal->pCount );
    for( int i = 0; i < retVal->pCount; i++ ){
        fread(&retVal->hParticles[i], sizeof(Particle), 1, data);
    }

    fclose(data);
    return retVal;
}

void PrintFreeMemory(){
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
}