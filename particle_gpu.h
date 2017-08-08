#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <ostream>
#include <string>
#include <vector>

#ifdef BUILD_FIELD_DOUBLE
typedef double fieldSize;
#else
typedef float fieldSize;
#endif

struct Particle {
	int pidx, procidx;
	float vp[3], xp[3], uf[3], xrhs[3], vrhs[3];
	float Tp, Tprhs_s, Tprhs_L, Tf, radius, radrhs, qinf, qstar;

	friend std::ostream &operator<<(std::ostream &stream, const Particle &p);
};

struct Parameters {
	int Evaporation, LinearInterpolation;

	// Material Properties

	// Air Properties
	float rhoa, nuf, Cpa, Pra, Sc;

	// Particle Properties
	float rhow, part_grav, Cpp, Mw, Ru, Ms, Sal, Gam, Ion, Os;

	// Particle Initial Conditions
	float radius_mass;
};

struct Device {
#ifdef BUILD_CUDA
	cudaStream_t Stream;
#endif
	int ParticleCount, ParticleOffset;

	Particle *Particles;
	fieldSize *Uext, *Vext, *Wext, *Text, *Qext;
	float *Z, *ZZ;
};

struct GPU {
	Parameters mParameters;

	unsigned int pCount;
	Particle *hParticles, *dParticles;

	int GridHeight, GridWidth, GridDepth;
	double FieldWidth, FieldHeight, FieldDepth;

	fieldSize *hUext, *hVext, *hWext, *hText, *hQext;
	float *hZ, *hZZ;

	// Statistics
	double *hPartCount, *hVPSum, *hVPSumSQ, *hRPSum, *hTPSum, *hTFSum, *hQFSum, *hQSTARSum, radmean;

	// GPU Memory
	Device *mDevices;
	unsigned int cDevice, DeviceCount;
};

extern "C" void rand2_seed(int seed);
extern "C" double rand2();
extern "C" GPU *NewGPU(const int particles, const int height, const int width, const int depth, const double fWidth, const double fHeight, const double fDepth, double *z, double *zz, const Parameters *params);
extern "C" void ParticleAdd(GPU *gpu, const int position, const Particle *input);
extern "C" Particle ParticleGet(GPU *gpu, const int position);
extern "C" void ParticleUpload(GPU *gpu);
extern "C" void ParticleInit(GPU *gpu, const int particles, const Particle *input);
extern "C" void ParticleGenerate(GPU *gpu, const int processors, const int ncpus, const int seed, const double temperature, const double radius, const double qinfp);
extern "C" void ParticleInterpolate(GPU *gpu, const double dx, const double dy);
extern "C" void ParticleStep(GPU *gpu, const int it, const int istage, const double dt);
extern "C" void ParticleUpdateNonPeriodic(GPU *gpu);
extern "C" void ParticleUpdatePeriodic(GPU *gpu);
extern "C" void ParticleCalculateStatistics(GPU *gpu, const double dx, const double dy);
extern "C" void ParticleDownload(GPU *gpu);
extern "C" void ParticleUpdate(GPU *gpu, const int it, const int istage, const double dt, const double dx, const double dy);
extern "C" void ParticleFieldSet(GPU *gpu, fieldSize *uext, fieldSize *vext, fieldSize *wext, fieldSize *text, fieldSize *qext);

extern "C" void ParticleWrite(GPU *gpu);
extern "C" GPU *ParticleRead(const char *path);

extern "C" void PrintFreeMemory();

// Fortran Data Access
extern "C" void ParticleFillStatistics(GPU *gpu, double *partCount, double *vSum, double *vSumSQ, double *rSum, double *tSum, double *tfSum, double *qfSum, double *qstarSum, double *single_stats);

// Helper Functions
const std::vector<double> ReadDoubleArray(const std::string &path);
void WriteDoubleArray(const std::string &path, const std::vector<double> &array);

// Test Functions
extern "C" int *ParticleFindXYNeighbours(const double dx, const double dy, const Particle *particle);

// Test Helper Functions
void SetParameters(GPU *gpu, const Parameters *params);

#endif // PARTICLE_H_
