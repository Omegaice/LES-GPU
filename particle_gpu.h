#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <ostream>
#include <string>
#include <vector>

#ifdef BUILD_CUDA
#include <cuda_runtime.h>
#endif

struct Particle {
	int pidx, procidx;
	double vp[3], xp[3], uf[3], xrhs[3], vrhs[3];
	double Tp, Tprhs_s, Tprhs_L, Tf, radius, radrhs, qinf, qstar;

	friend std::ostream& operator<< (std::ostream& stream, const Particle& p);
};

struct Parameters {
	int Evaporation, LinearInterpolation;

	// Material Properties

	// Air Properties
	double rhoa, nuf, Cpa, Pra, Sc;

	// Particle Properties
	double rhow, part_grav, Cpp, Mw, Ru, Ms, Sal, Gam, Ion, Os;

	// Particle Initial Conditions
	double radius_mass;
};

struct GPU {
	Parameters mParameters;

	unsigned int pCount;
	Particle *hParticles, *dParticles;

	int GridHeight, GridWidth, GridDepth;
	double FieldWidth, FieldHeight, FieldDepth;
	double *hUext, *hVext, *hWext, *hText, *hQext, *hZ, *hZZ;
#ifdef BUILD_CUDA
	double *dUext, *dVext, *dWext, *dText, *dQext, *dZ, *dZZ;
	cudaTextureObject_t tUext, tVext, tWext, tText, tQext;
#endif

	// Statistics
	double *hPartCount, *hVPSum, *hVPSumSQ;
#ifdef BUILD_CUDA
	double *dPartCount, *dVPSum, *dVPSumSQ;
#endif
};

extern "C" double rand2(int idum, bool reset = false);
extern "C" GPU* NewGPU(const int particles, const int height, const int width, const int depth, const double fWidth, const double fHeight, const double fDepth, double* z, double* zz, const Parameters* params);
extern "C" void ParticleFieldSet( GPU *gpu, double *uext, double *vext, double *wext, double *text, double *qext );
extern "C" void ParticleAdd( GPU *gpu, const int position, const Particle* input );
extern "C" Particle ParticleGet( GPU *gpu, const int position );
extern "C" void ParticleUpload( GPU *gpu );
extern "C" void ParticleInit( GPU* gpu, const int particles, const Particle* input );
extern "C" void ParticleGenerate( GPU* gpu, const int processors, const int particles, const int seed, const double temperature, const double xmin, const double xmax, const double ymin, const double ymax, const double zl, const double delta_vis, const double radius, const double qinfp );
extern "C" void ParticleInterpolate( GPU *gpu, const double dx, const double dy );
extern "C" void ParticleStep( GPU* gpu, const int it, const int istage, const double dt );
extern "C" void ParticleUpdateNonPeriodic( GPU *gpu );
extern "C" void ParticleUpdatePeriodic( GPU *gpu );
extern "C" void ParticleCalculateStatistics( GPU *gpu, const double dx, const double dy );
extern "C" void ParticleDownloadHost( GPU *gpu );
extern "C" Particle* ParticleDownload( GPU* gpu );

extern "C" void ParticleWrite( GPU* gpu );
extern "C" GPU* ParticleRead(const char *path);

extern "C" void PrintFreeMemory();

// Fortran Data Access
extern "C" void ParticleFillStatistics(GPU* gpu, double* partCount, double* vSum, double* vSumSQ);

// Helper Functions
const std::vector<double> ReadDoubleArray(const std::string& path);
void WriteDoubleArray(const std::string& path, const std::vector<double>& array);

// Test Functions
extern "C" int* ParticleFindXYNeighbours(const double dx, const double dy, const Particle* particle);

// Test Helper Functions
void SetParameters(GPU* gpu, const Parameters* params);

#endif // PARTICLE_H_