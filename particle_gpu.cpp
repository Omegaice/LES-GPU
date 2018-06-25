#include "particle_gpu.h"
#include "assert.h"
#include "stdio.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#ifndef BUILD_CUDA
#include "stdlib.h"
#include "string.h"
#endif

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifndef BUILD_CUDA
#define DEVICE
#define GLOBAL
#define SHARED
#define CONSTANT
#else
#define DEVICE __device__
#define GLOBAL __global__
#define CONSTANT __constant__
#define SHARED __shared__

#define gpuErrchk(ans) \
	{ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if(code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}
#endif

extern "C" int gpudevices() {
	int nDevices = 1;
#ifdef BUILD_CUDA
	if(cudaGetDeviceCount(&nDevices) == cudaErrorInsufficientDriver) {
		return 0;
	}
#endif
	return nDevices;
}

CONSTANT Parameters cParams;

DEVICE void GPUFindXYNeighbours(const double dx, const double dy, const Particle *__restrict__ particles, int *__restrict__ neighbours) {
	neighbours[0 * 6 + 2] = floor(particles[0].xp[0] / dx) + 1;
	neighbours[1 * 6 + 2] = floor(particles[0].xp[1] / dy) + 1;

	neighbours[0 * 6 + 1] = neighbours[0 * 6 + 2] - 1;
	neighbours[0 * 6 + 0] = neighbours[0 * 6 + 1] - 1;
	neighbours[0 * 6 + 3] = neighbours[0 * 6 + 2] + 1;
	neighbours[0 * 6 + 4] = neighbours[0 * 6 + 3] + 1;
	neighbours[0 * 6 + 5] = neighbours[0 * 6 + 4] + 1;

	neighbours[1 * 6 + 1] = neighbours[1 * 6 + 2] - 1;
	neighbours[1 * 6 + 0] = neighbours[1 * 6 + 1] - 1;
	neighbours[1 * 6 + 3] = neighbours[1 * 6 + 2] + 1;
	neighbours[1 * 6 + 4] = neighbours[1 * 6 + 3] + 1;
	neighbours[1 * 6 + 5] = neighbours[1 * 6 + 4] + 1;
}

GLOBAL void GGPUFindXYNeighbours(const double dx, const double dy, const Particle *__restrict__ particles, int *__restrict__ neighbours) {
	GPUFindXYNeighbours(dx, dy, particles, neighbours);
}

int *ParticleFindXYNeighbours(const double dx, const double dy, const Particle *particle) {
	int *hResult = (int *)malloc(sizeof(int) * 12);

#ifdef BUILD_CUDA
	int *dResult;
	gpuErrchk(cudaMalloc((void **)&dResult, sizeof(int) * 12));

	Particle *dParticle;
	gpuErrchk(cudaMalloc((void **)&dParticle, sizeof(Particle) * 1));
	gpuErrchk(cudaMemcpy(dParticle, particle, sizeof(Particle) * 1, cudaMemcpyHostToDevice));

	GGPUFindXYNeighbours<<<1, 1>>>(dx, dy, dParticle, dResult);
	gpuErrchk(cudaPeekAtLastError());

	gpuErrchk(cudaMemcpy(hResult, dResult, sizeof(int) * 12, cudaMemcpyDeviceToHost));
#else
	GPUFindXYNeighbours(dx, dy, particle, hResult);
#endif
	return hResult;
}

GLOBAL void GPUFieldInterpolateLinear(const int nx, const int ny, const double dx, const double dy, const int nnz, const double *__restrict__ z, const double *__restrict__ zz, const fieldSize *__restrict__ uext, const fieldSize *__restrict__ vext, const fieldSize *__restrict__ wext, const fieldSize *__restrict__ Text, const fieldSize *__restrict__ T2ext, const int pcount, Particle *__restrict__ particles) {
	int index_start = 0, index_stride = 1;
#ifdef BUILD_CUDA
	index_start = blockIdx.x * blockDim.x + threadIdx.x;
	index_stride = blockDim.x * gridDim.x;
#endif

	for(int idx = index_start; idx < pcount; idx += index_stride) {

// Setup shared memory for Z and ZZ
#ifdef BUILD_CUDA
		extern SHARED double shared[];

		double *dzu = shared, *dzw = &shared[nnz + 1];
		if(threadIdx.x == 0) {
			for(int i = 1; i < nnz; i++) {
				dzu[i] = zz[i] - zz[i - 1];
				dzw[i] = z[i] - z[i - 1];
			}
			dzu[0] = dzu[1];
			dzw[0] = dzw[1];

			dzu[nnz] = dzu[nnz - 1];
			dzw[nnz] = dzw[nnz - 1];
		}
		__syncthreads();
#else
		const double *zShared = z;
		const double *zzShared = zz;

		double dzu[nnz + 1], dzw[nnz + 1];
		for(int i = 1; i < nnz; i++) {
			dzu[i] = zz[i] - zz[i - 1];
			dzw[i] = z[i] - z[i - 1];
		}
		dzu[0] = dzu[1];
		dzw[0] = dzw[1];

		dzu[nnz] = dzu[nnz - 1];
		dzw[nnz] = dzw[nnz - 1];
#endif

		const double xPos = particles[idx].xp[0];
		const double yPos = particles[idx].xp[1];
		const double zPos = particles[idx].xp[2];

		const int ipt = floor(xPos / dx) + 1;
		const int jpt = floor(yPos / dy) + 1;

		int kpt = 0, kwpt = 0;
		for(int j = 0; j < nnz; j++) {
			if(zz[j] < zPos) kpt = j;
			if(z[j] < zPos) kwpt = j;
		}


		double xUF = 0.0, yUF = 0.0, zUF = 0.0;
		double Tf = 0.0, qinf = 0.0;

#pragma unroll
		for(int i = 0; i < 2; i++) {
#pragma unroll
			for(int j = 0; j < 2; j++) {
#pragma unroll
				for(int k = 0; k < 2; k++) {
					const int ix = i + ipt;
					const int iy = i + jpt;
					const int izuv = k + kpt;
					const int izw = k + kwpt;

					const double xv = dx * (i + ipt - 1);
					const double yv = dy * (j + jpt - 1);

					const double wtx = 1.0 - (std::abs(xPos - xv) / dx);
					const double wty = 1.0 - (std::abs(yPos - yv) / dy);
					const double wtz = 1.0 - (std::abs(zPos - zz[izuv]) / dzu[kpt + 1]);
					const double wtzw = 1.0 - (std::abs(zPos - z[izw]) / dzw[kwpt + 1]);
					xUF += uext[(ix + 1) + (iy + 1) * nx + izuv * ny * nx] * wtx * wty * wtz;
					yUF += vext[(ix + 1) + (iy + 1) * nx + izuv * ny * nx] * wtx * wty * wtz;
					zUF += wext[(ix + 1) + (iy + 1) * nx + izw * ny * nx] * wtx * wty * wtzw;
					Tf += Text[(ix + 1) + (iy + 1) * nx + izuv * ny * nx] * wtx * wty * wtz;
					qinf += T2ext[(ix + 1) + (iy + 1) * nx + izuv * ny * nx] * wtx * wty * wtz;

                                        if (kpt == 0){
					xUF = uext[(ix + 1) + (iy + 1) * nx + 1 * ny * nx];
					yUF = vext[(ix + 1) + (iy + 1) * nx + 1 * ny * nx];
					Tf = Text[(ix + 1) + (iy + 1) * nx + 1 * ny * nx];
					qinf = T2ext[(ix + 1) + (iy + 1) * nx + 1 * ny * nx];
                                         }

                                        if (kpt == nnz-2){
					xUF = uext[(ix + 1) + (iy + 1) * nx + (nnz-2) * ny * nx];
					yUF = vext[(ix + 1) + (iy + 1) * nx + (nnz-2) * ny * nx];
					Tf = Text[(ix + 1) + (iy + 1) * nx + (nnz-2) * ny * nx];
					qinf = T2ext[(ix + 1) + (iy + 1) * nx + (nnz-2) * ny * nx];
                                         }
				}
			}
		}

		particles[idx].uf[0] = xUF;
		particles[idx].uf[1] = yUF;
		particles[idx].uf[2] = zUF;
		particles[idx].Tf = Tf;
		particles[idx].qinf = qinf;
	}
}

GLOBAL void GPUFieldInterpolate(const int nx, const int ny, const double dx, const double dy, const int nnz, const double *__restrict__ z, const double *__restrict__ zz, const fieldSize *__restrict__ uext, const fieldSize *__restrict__ vext, const fieldSize *__restrict__ wext, const fieldSize *__restrict__ Text, const fieldSize *__restrict__ T2ext, const int pcount, Particle *__restrict__ particles) {
	int index_start = 0, index_stride = 1;
#ifdef BUILD_CUDA
	index_start = blockIdx.x * blockDim.x + threadIdx.x;
	index_stride = blockDim.x * gridDim.x;
#endif

	for(int idx = index_start; idx < pcount; idx += index_stride) {

#ifdef BUILD_CUDA
		extern SHARED double shared[];

		// Shared memory for Z and ZZ
		double *zShared = shared, *zzShared = &shared[nnz];
		if(threadIdx.x == 0) {
			for(int i = 0; i < nnz; i++) {
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

		int kuvpts[6] = {0, 0, 0, 0, 0, 0};
		for(; kuvpts[2] < nnz; kuvpts[2]++) {
			if(zzShared[kuvpts[2]] > particles[idx].xp[2]) {
				break;
			}
		}
		kuvpts[2] -= 1;

		kuvpts[3] = kuvpts[2] + 1;
		kuvpts[4] = kuvpts[3] + 1;
		kuvpts[5] = kuvpts[4] + 1;
		kuvpts[1] = kuvpts[2] - 1;
		kuvpts[0] = kuvpts[1] - 1;

		int kwpts[6] = {0, 0, 0, 0, 0, 0};
		for(; kwpts[2] < nnz; kwpts[2]++) {
			if(zShared[kwpts[2]] > particles[idx].xp[2]) {
				break;
			}
		}
		kwpts[2] -= 1;

		kwpts[3] = kwpts[2] + 1;
		kwpts[4] = kwpts[3] + 1;
		kwpts[5] = kwpts[4] + 1;
		kwpts[1] = kwpts[2] - 1;
		kwpts[0] = kwpts[1] - 1;

		double wt[4][6] = {
			{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		};

		double dxvec[2] = {dx, dy};
		for(int iz = 0; iz < 2; iz++) {
			for(int j = 0; j < 6; j++) {
				double xjval = dxvec[iz] * (ijpts[iz * 6 + j] - 1);
				double pj = 1.0;
				for(int k = 0; k < 6; k++) {
					double xkval = dxvec[iz] * (ijpts[iz * 6 + k] - 1);
					if(j != k) {
						pj = pj * (particles[idx].xp[iz] - xkval) / (xjval - xkval);
					}
				}
				wt[iz][j] = pj;
			}
		}

		int first, last;
		if(kuvpts[2] == 1) {
			first = 2;
			last = 4;
			kuvpts[0] = 0;
			kuvpts[1] = 0;
		} else if(kuvpts[2] == 0) {
			first = 3;
			last = 5;
			kuvpts[0] = 0;
			kuvpts[1] = 0;
			kuvpts[2] = 0;
		} else if(kuvpts[2] < 0) {
			first = 0;
			last = 0;
			kuvpts[0] = 0;
			kuvpts[1] = 0;
			kuvpts[2] = 0;
		} else if(kuvpts[2] == 2) {
			first = 1;
			last = 5;
		} else if(kuvpts[2] == nnz - 2) {
			first = 1;
			last = 3;
			kuvpts[3] = nnz - 2;
			kuvpts[4] = nnz - 2;
			kuvpts[5] = nnz - 2;
		} else if(kuvpts[2] > nnz - 2) {
			first = 0;
			last = 0;
			kuvpts[3] = nnz - 2;
			kuvpts[4] = nnz - 2;
			kuvpts[5] = nnz - 2;
		} else if(kuvpts[2] == nnz - 3) {
			first = 2;
			last = 4;
			kuvpts[4] = nnz - 2;
			kuvpts[5] = nnz - 2;
		} else if(kuvpts[2] == nnz - 4) {
			first = 1;
			last = 5;
		} else {
			first = 0;
			last = 6;
		}

		for(int j = first; j < last; j++) {
			double xjval = zzShared[kuvpts[j]];
			double pj = 1.0;
			for(int k = first; k < last; k++) {
				double xkval = zzShared[kuvpts[k]];
				if(j != k) {
					pj = pj * (particles[idx].xp[2] - xkval) / (xjval - xkval);
				}
			}
			wt[2][j] = pj;
		}

		if(kwpts[2] == 0) {
			first = 2;
			last = 4;
			kwpts[0] = 0;
			kwpts[1] = 0;
		} else if(kwpts[2] < 0) {
			first = 0;
			last = 0;
			kwpts[0] = 0;
			kwpts[1] = 0;
			kwpts[2] = 0;
		} else if(kwpts[2] == 1) {
			first = 1;
			last = 5;
			kwpts[0] = 0;
		} else if(kwpts[2] >= nnz - 2) {
			first = 0;
			last = 0;
			kwpts[3] = nnz - 2;
			kwpts[4] = nnz - 2;
			kwpts[5] = nnz - 2;
		} else if(kwpts[2] == nnz - 3) {
			first = 2;
			last = 4;
			kwpts[4] = nnz - 2;
			kwpts[5] = nnz - 2;
		} else if(kwpts[2] == nnz - 4) {
			first = 1;
			last = 5;
			kwpts[0] = 0;
		} else {
			first = 0;
			last = 6;
		}

		for(int j = first; j < last; j++) {
			double xjval = zShared[kwpts[j]];
			double pj = 1.0;
			for(int k = first; k < last; k++) {
				double xkval = zShared[kwpts[k]];
				if(j != k) {
					pj = pj * (particles[idx].xp[2] - xkval) / (xjval - xkval);
				}
			}
			wt[3][j] = pj;
		}

		particles[idx].uf[0] = 0.0;
		particles[idx].uf[1] = 0.0;
		particles[idx].uf[2] = 0.0;

		particles[idx].Tf = 0.0;
		particles[idx].qinf = 0.0;
		for(int k = 0; k < 6; k++) {
			for(int j = 0; j < 6; j++) {
				for(int i = 0; i < 6; i++) {
					const int ix = ijpts[0 * 6 + i];
					const int iy = ijpts[1 * 6 + j];
					const int izuv = kuvpts[k];
					const int izw = kwpts[k];
					particles[idx].uf[0] = particles[idx].uf[0] + uext[(ix + 1) + (iy + 1) * nx + izuv * ny * nx] * wt[0][i] * wt[1][j] * wt[2][k];
					particles[idx].uf[1] = particles[idx].uf[1] + vext[(ix + 1) + (iy + 1) * nx + izuv * ny * nx] * wt[0][i] * wt[1][j] * wt[2][k];
					particles[idx].uf[2] = particles[idx].uf[2] + wext[(ix + 1) + (iy + 1) * nx + izw * ny * nx] * wt[0][i] * wt[1][j] * wt[3][k];
					particles[idx].Tf = particles[idx].Tf + Text[(ix + 1) + (iy + 1) * nx + izuv * ny * nx] * wt[0][i] * wt[1][j] * wt[2][k];
					particles[idx].qinf = particles[idx].qinf + T2ext[(ix + 1) + (iy + 1) * nx + izuv * ny * nx] * wt[0][i] * wt[1][j] * wt[2][k];
				}
			}
		}
	}
}

GLOBAL void GPUUpdateParticles(const int it, const int istage, const double dt, const int pcount, Particle *__restrict__ particles) {
	int index_start = 0, index_stride = 1;
#ifdef BUILD_CUDA
	index_start = blockIdx.x * blockDim.x + threadIdx.x;
	index_stride = blockDim.x * gridDim.x;
#endif

	for(int idx = index_start; idx < pcount; idx += index_stride) {

#ifdef BUILD_CUDA
		SHARED double pi, pi2, m_s, CpaCpp, Lv, pPra, pSc, zetas[3], gama[3], g[3], dtZ, dtG;

		if(threadIdx.x == 0) {
			pi = 4.0 * atan(1.0);
			pi2 = 2.0 * pi;
			m_s = cParams.Sal / 1000.0 * 4.0 / 3.0 * pi * pow(cParams.radius_mass, 3) * cParams.rhow;

			CpaCpp = cParams.Cpa / cParams.Cpp;

			zetas[0] = 0.0;
			zetas[1] = -17.0 / 60.0;
			zetas[2] = -5.0 / 12.0;
			gama[0] = 8.0 / 15.0;
			gama[1] = 5.0 / 12.0;
			gama[2] = 3.0 / 4.0;
			g[0] = 0.0;
			g[1] = 0.0;
			g[2] = cParams.part_grav;
			Lv = (25.0 - 0.02274 * 26.0) * 100000;

			pPra = pow(cParams.Pra, 1.0 / 3.0);
			pSc = pow(cParams.Sc, 1.0 / 3.0);

			dtZ = dt * zetas[istage];
			dtG = dt * gama[istage];
		}
		__syncthreads();
#else
		const double pi = 4.0 * atan(1.0);
		const double pi2 = 2.0 * pi;
		const double m_s = cParams.Sal / 1000.0 * 4.0 / 3.0 * pi * pow(cParams.radius_mass, 3) * cParams.rhow;

		const double CpaCpp = cParams.Cpa / cParams.Cpp;

		const double zetas[3] = {0.0, -17.0 / 60.0, -5.0 / 12.0};
		const double gama[3] = {8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0};
		const double g[3] = {0.0, 0.0, cParams.part_grav};
		const double Lv = (25.0 - 0.02274 * 26.0) * 100000;
		const double pPra = pow(cParams.Pra, 1.0 / 3.0);
		const double pSc = pow(cParams.Sc, 1.0 / 3.0);

		const double dtZ = dt * zetas[istage];
		const double dtG = dt * gama[istage];
#endif

		if(it == 1) {
			for(int j = 0; j < 3; j++) {
				particles[idx].vp[j] = particles[idx].uf[j];
			}
			particles[idx].Tp = particles[idx].Tf;
		}

		double diff[3];
		for(int j = 0; j < 3; j++) {
			diff[j] = particles[idx].vp[j] - particles[idx].uf[j];
		}
		double diffnorm = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
		double Rep = 2.0 * particles[idx].radius * diffnorm / cParams.nuf;
		double Volp = pi2 * 2.0 / 3.0 * (particles[idx].radius * particles[idx].radius * particles[idx].radius);
		double rhop = (m_s + Volp * cParams.rhow) / Volp;
		double taup_i = 18.0 * cParams.rhoa * cParams.nuf / rhop / ((2.0 * particles[idx].radius) * (2.0 * particles[idx].radius));

		double corrfac = 1.0 + 0.15 * pow(Rep, 0.687);
		double Nup = 2.0 + 0.6 * pow(Rep, 0.5) * pPra;
		double Shp = 2.0 + 0.6 * pow(Rep, 0.5) * pSc;

		double TfC = particles[idx].Tf - 273.15;
		double einf = 610.94 * exp(17.6257 * TfC / (TfC + 243.04));
		double Eff_C = 2.0 * cParams.Mw * cParams.Gam / (cParams.Ru * cParams.rhow * particles[idx].radius * particles[idx].Tp);
		double Eff_S = cParams.Ion * cParams.Os * m_s * cParams.Mw / cParams.Ms / (Volp * rhop - m_s);
		double estar = einf * exp(cParams.Mw * Lv / cParams.Ru * (1.0 / particles[idx].Tf - 1.0 / particles[idx].Tp) + Eff_C - Eff_S);
		particles[idx].qstar = cParams.Mw / cParams.Ru * estar / particles[idx].Tp / cParams.rhoa;

		double xtmp[3], vtmp[3];
		for(int j = 0; j < 3; j++) {
			xtmp[j] = particles[idx].xp[j] + dtZ * particles[idx].xrhs[j];
			vtmp[j] = particles[idx].vp[j] + dtZ * particles[idx].vrhs[j];
		}

		double Tptmp = particles[idx].Tp + dtZ * particles[idx].Tprhs_s;
		Tptmp += dtZ * particles[idx].Tprhs_L;
		double radiustmp = particles[idx].radius + dtZ * particles[idx].radrhs;

		for(int j = 0; j < 3; j++) {
			particles[idx].xrhs[j] = particles[idx].vp[j];
		}

		for(int j = 0; j < 3; j++) {
			particles[idx].vrhs[j] = corrfac * taup_i * (particles[idx].uf[j] - particles[idx].vp[j]) - g[j];
		}

		particles[idx].radrhs = Shp / 9.0 / cParams.Sc * rhop / cParams.rhow * particles[idx].radius * taup_i * (particles[idx].qinf - particles[idx].qstar) * cParams.Evaporation;
		particles[idx].Tprhs_s = -Nup / 3.0 / cParams.Pra * CpaCpp * rhop / cParams.rhow * taup_i * (particles[idx].Tp - particles[idx].Tf);
		particles[idx].Tprhs_L = 3.0 * Lv / cParams.Cpp / particles[idx].radius * particles[idx].radrhs;

		for(int j = 0; j < 3; j++) {
			particles[idx].xp[j] = xtmp[j] + dtG * particles[idx].xrhs[j];
			particles[idx].vp[j] = vtmp[j] + dtG * particles[idx].vrhs[j];
		}
		particles[idx].Tp = Tptmp + dtG * particles[idx].Tprhs_s;
		particles[idx].Tp += +dtG * particles[idx].Tprhs_L;
		particles[idx].radius = radiustmp + dtG * particles[idx].radrhs;
	}
}

GLOBAL void GPUUpdateNonperiodic(const double grid_width, const int pcount, Particle *__restrict__ particles) {
	int index_start = 0, index_stride = 1;
#ifdef BUILD_CUDA
	index_start = blockIdx.x * blockDim.x + threadIdx.x;
	index_stride = blockDim.x * gridDim.x;
#endif

	for(int idx = index_start; idx < pcount; idx += index_stride) {
		const double radius = particles[idx].radius;
		const double zPos = particles[idx].xp[2];

		const double top = grid_width - radius;
		const double bot = 0.0 + radius;

		if(zPos > top) {
			particles[idx].xp[2] = top - (zPos - top);
			particles[idx].vp[2] = -particles[idx].vp[2];
		} else if(zPos < bot) {
			particles[idx].xp[2] = bot + (bot - zPos);
			particles[idx].vp[2] = -particles[idx].vp[2];
		}
	}
}

GLOBAL void GPUUpdatePeriodic(const double grid_width, const double grid_height, const int pcount, Particle *__restrict__ particles) {
	int index_start = 0, index_stride = 1;
#ifdef BUILD_CUDA
	index_start = blockIdx.x * blockDim.x + threadIdx.x;
	index_stride = blockDim.x * gridDim.x;
#endif

	for(int idx = index_start; idx < pcount; idx += index_stride) {
		const double xPos = particles[idx].xp[0];
		const double yPos = particles[idx].xp[1];

		if(xPos > grid_width) {
			particles[idx].xp[0] -= grid_width;
		} else if(xPos < 0.0) {
			particles[idx].xp[0] = grid_width + xPos;
		}

		if(yPos > grid_height) {
			particles[idx].xp[1] -= grid_height;
		} else if(yPos < 0.0) {
			particles[idx].xp[1] = grid_height + yPos;
		}
	}
}

void GPUCalculateStatistics(const int nnz, const double *__restrict__ z, double *__restrict__ partcount_t, double *__restrict__ vpsum_t, double *__restrict__ vpsqrsum_t, double *__restrict__ rpsum_t, double *__restrict__ tpsum_t, double *__restrict__ tfsum_t, double *__restrict__ qfsum_t, double *__restrict__ qstarsum_t,  const int pcount, Particle *__restrict__ particles, double &radmean , double &radmin, double &radmax) {

        double radsum = 0.0;
        radmin = 1.0;
        radmax = -1.0;
	for(int i = 0; i < pcount; i++) {
		int kpt = 0;
		for(; kpt < nnz; kpt++) {
			if(z[kpt] > particles[i].xp[2]) {
				break;
			}
		}
		kpt -= 1;

		partcount_t[kpt] += 1.0;

		vpsum_t[kpt * 3 + 0] += particles[i].vp[0];
		vpsum_t[kpt * 3 + 1] += particles[i].vp[1];
		vpsum_t[kpt * 3 + 2] += particles[i].vp[2];

		vpsqrsum_t[kpt * 3 + 0] += (particles[i].vp[0] * particles[i].vp[0]);
		vpsqrsum_t[kpt * 3 + 1] += (particles[i].vp[1] * particles[i].vp[1]);
		vpsqrsum_t[kpt * 3 + 2] += (particles[i].vp[2] * particles[i].vp[2]);

                rpsum_t[kpt] += particles[i].radius;
                tpsum_t[kpt] += particles[i].Tp;
                tfsum_t[kpt] += particles[i].Tf;
                qfsum_t[kpt] += particles[i].qinf;
                qstarsum_t[kpt] += particles[i].qstar;

                radsum += particles[i].radius;
                
                if(particles[i].radius > radmax){
                  radmax = particles[i].radius;
                }
                if(particles[i].radius < radmin){
                  radmin = particles[i].radius;
                }

	}

                radmean = radsum/pcount;
}

const int random_NTAB = 32;
static int random_idum = 1080, random_iv[random_NTAB], random_iy = 0, random_idum2 = 123456789;

void rand2_seed(int seed) {
	for(int i = 0; i < random_NTAB; i++) {
		random_iv[i] = 0;
	}
	random_iy = 0;
	random_idum = seed;
	random_idum2 = 123456789;
}

extern "C" double rand2() {
	int k = 0, IM1 = 2147483563, IM2 = 2147483399, IMM1 = IM1 - 1, IA1 = 40014, IA2 = 40692, IQ1 = 53668, IQ2 = 52774, IR1 = 12211, IR2 = 3791, NDIV = 1 + IMM1 / random_NTAB;
	double AM = 1.0 / IM1, EPS = 1.2e-7, RNMX = 1.0 - EPS;

	if(random_idum <= 0) {
		random_idum = MAX(-random_idum, 1);
		random_idum2 = random_idum;
		for(int j = random_NTAB + 8; j > 1; j--) {
			k = random_idum / IQ1;
			random_idum = IA1 * (random_idum - k * IQ1) - k * IR1;
			if(random_idum < 0) {
				random_idum += IM1;
			}
			if(j <= random_NTAB) {
				random_iv[j] = random_idum;
			}
		}
		random_iy = random_iv[0];
	}

	k = random_idum / IQ1;
	random_idum = IA1 * (random_idum - k * IQ1) - k * IR1;
	if(random_idum < 0) {
		random_idum += IM1;
	}
	k = random_idum2 / IQ2;
	random_idum2 = IA2 * (random_idum2 - k * IQ2) - k * IR2;
	if(random_idum2 < 0) {
		random_idum2 += IM2;
	}
	const int j = random_iy / NDIV;
	random_iy = random_iv[j] - random_idum2;
	random_iv[j] = random_idum;
	if(random_iy < 1) {
		random_iy += IMM1;
	}
	return MIN(AM * random_iy, RNMX);
}

void SetDeviceIndex(GPU *gpu, const unsigned int index) {
#ifdef BUILD_CUDA
	if(gpu->cDevice != index) {
		gpu->cDevice = index;
		gpuErrchk(cudaSetDevice(gpu->cDevice));
	}
#endif
}

Device *GetDeviceMemory(GPU *gpu) {
#ifdef BUILD_CUDA
	return &gpu->mDevices[gpu->cDevice];
#else
	return nullptr;
#endif
}

extern "C" GPU *NewGPU(const int particles, const int width, const int height, const int depth, const double fWidth, const double fHeight, const double fDepth, double *z, double *zz, const Parameters *params) {
	GPU *retVal = (GPU *)malloc(sizeof(GPU));

	// Particle Data
	retVal->pCount = particles;
#ifdef BUILD_CUDA
	gpuErrchk(cudaMallocHost((void **)&retVal->hParticles, sizeof(Particle) * particles));
#else
	retVal->hParticles = (Particle *)malloc(sizeof(Particle) * particles);
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
	retVal->hPartCount = (double *)malloc(sizeof(double) * retVal->GridDepth);
	retVal->hVPSum = (double *)malloc(sizeof(double) * retVal->GridDepth * 3);
	retVal->hVPSumSQ = (double *)malloc(sizeof(double) * retVal->GridDepth * 3);
	retVal->hRPSum = (double *)malloc(sizeof(double) * retVal->GridDepth);
	retVal->hTPSum = (double *)malloc(sizeof(double) * retVal->GridDepth);
	retVal->hTFSum = (double *)malloc(sizeof(double) * retVal->GridDepth);
	retVal->hQFSum = (double *)malloc(sizeof(double) * retVal->GridDepth);
	retVal->hQSTARSum = (double *)malloc(sizeof(double) * retVal->GridDepth);

	memset(retVal->hPartCount, 0.0, sizeof(double) * retVal->GridDepth);
	memset(retVal->hVPSum, 0.0, sizeof(double) * retVal->GridDepth * 3);
	memset(retVal->hVPSumSQ, 0.0, sizeof(double) * retVal->GridDepth * 3);
	memset(retVal->hTPSum, 0.0, sizeof(double) * retVal->GridDepth);
	memset(retVal->hRPSum, 0.0, sizeof(double) * retVal->GridDepth);
	memset(retVal->hTFSum, 0.0, sizeof(double) * retVal->GridDepth);
	memset(retVal->hQFSum, 0.0, sizeof(double) * retVal->GridDepth);
	memset(retVal->hQSTARSum, 0.0, sizeof(double) * retVal->GridDepth);

#ifdef BUILD_CUDA
	retVal->mDevices = (Device *)malloc(sizeof(Device) * gpudevices());

	unsigned int offset = 0;
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(retVal, i);
		Device *dev = GetDeviceMemory(retVal);

		dev->ParticleOffset = offset;
		dev->ParticleCount = retVal->pCount / gpudevices();
		if(i == 0) {
			dev->ParticleCount += retVal->pCount % gpudevices();
		}
		offset += dev->ParticleCount;

		gpuErrchk(cudaStreamCreate(&dev->Stream));

		gpuErrchk(cudaMalloc((void **)&dev->Particles, sizeof(Particle) * dev->ParticleCount));

		gpuErrchk(cudaMalloc((void **)&dev->Uext, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));
		gpuErrchk(cudaMallocHost((void **)&retVal->hUext, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));

		gpuErrchk(cudaMalloc((void **)&dev->Vext, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));
		gpuErrchk(cudaMallocHost((void **)&retVal->hVext, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));

		gpuErrchk(cudaMalloc((void **)&dev->Wext, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));
		gpuErrchk(cudaMallocHost((void **)&retVal->hWext, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));

		gpuErrchk(cudaMalloc((void **)&dev->Text, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));
		gpuErrchk(cudaMallocHost((void **)&retVal->hText, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));

		gpuErrchk(cudaMalloc((void **)&dev->Qext, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));
		gpuErrchk(cudaMallocHost((void **)&retVal->hQext, sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth));

		gpuErrchk(cudaMalloc((void **)&dev->Z, sizeof(double) * retVal->GridDepth));
		gpuErrchk(cudaMallocHost((void **)&retVal->hZ, sizeof(double) * retVal->GridDepth));
		memcpy(retVal->hZ, z, sizeof(double) * retVal->GridDepth);

		gpuErrchk(cudaMalloc((void **)&dev->ZZ, sizeof(double) * retVal->GridDepth));
		gpuErrchk(cudaMallocHost((void **)&retVal->hZZ, sizeof(double) * retVal->GridDepth));
		memcpy(retVal->hZZ, zz, sizeof(double) * retVal->GridDepth);

		gpuErrchk(cudaMemcpyAsync(dev->Z, z, sizeof(double) * retVal->GridDepth, cudaMemcpyHostToDevice, dev->Stream));
		gpuErrchk(cudaMemcpyAsync(dev->ZZ, zz, sizeof(double) * retVal->GridDepth, cudaMemcpyHostToDevice, dev->Stream));
	}

#ifdef BUILD_PERFORMANCE_PROFILE
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(retVal, i);
		Device *dev = GetDeviceMemory(retVal);
		cudaStreamSynchronize(dev->Stream);
	}
#endif
#else
	retVal->hUext = (fieldSize *)malloc(sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);
	retVal->hVext = (fieldSize *)malloc(sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);
	retVal->hWext = (fieldSize *)malloc(sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);
	retVal->hText = (fieldSize *)malloc(sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);
	retVal->hQext = (fieldSize *)malloc(sizeof(fieldSize) * retVal->GridWidth * retVal->GridHeight * retVal->GridDepth);

	retVal->hZ = (double *)malloc(sizeof(double) * retVal->GridDepth);
	memcpy(retVal->hZ, z, sizeof(double) * retVal->GridDepth);

	retVal->hZZ = (double *)malloc(sizeof(double) * retVal->GridDepth);
	memcpy(retVal->hZZ, zz, sizeof(double) * retVal->GridDepth);
#endif

	SetParameters(retVal, params);

	return retVal;
}

extern "C" void ParticleFieldSet(GPU *gpu, fieldSize *uext, fieldSize *vext, fieldSize *wext, fieldSize *text, fieldSize *qext) {
#ifdef BUILD_VERIFY_NAN
	std::cout << "Testing for NAN in field:" << std::endl;
	for(int i = 0; i < gpu->GridWidth * gpu->GridHeight * gpu->GridDepth; i++) {
		if(isnan(uext[i])) std::cerr << "UEXT NAN found at index " << i << std::endl;
		if(isnan(vext[i])) std::cerr << "VEXT NAN found at index " << i << std::endl;
		if(isnan(wext[i])) std::cerr << "WEXT NAN found at index " << i << std::endl;
		if(isnan(text[i])) std::cerr << "TEXT NAN found at index " << i << std::endl;
		if(isnan(qext[i])) std::cerr << "QEXT NAN found at index " << i << std::endl;
	}
	std::cout << "\tComplete" << std::endl;
#endif // BUILD_VERIFY_NAN

	memcpy(gpu->hUext, uext, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth);
	memcpy(gpu->hVext, vext, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth);
	memcpy(gpu->hWext, wext, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth);
	memcpy(gpu->hText, text, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth);
	memcpy(gpu->hQext, qext, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth);

#ifdef BUILD_CUDA
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);

		gpuErrchk(cudaMemcpyAsync(dev->Uext, gpu->hUext, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice, dev->Stream));
		gpuErrchk(cudaMemcpyAsync(dev->Vext, gpu->hVext, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice, dev->Stream));
		gpuErrchk(cudaMemcpyAsync(dev->Wext, gpu->hWext, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice, dev->Stream));
		gpuErrchk(cudaMemcpyAsync(dev->Text, gpu->hText, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice, dev->Stream));
		gpuErrchk(cudaMemcpyAsync(dev->Qext, gpu->hQext, sizeof(fieldSize) * gpu->GridWidth * gpu->GridHeight * gpu->GridDepth, cudaMemcpyHostToDevice, dev->Stream));
	}

#ifdef BUILD_PERFORMANCE_PROFILE
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);
		cudaStreamSynchronize(dev->Stream);
	}
#endif
#endif
}

extern "C" void ParticleAdd(GPU *gpu, const int position, const Particle *input) {
	assert(position >= 0 && position < gpu->pCount);
	memcpy(&gpu->hParticles[position], input, sizeof(Particle));
}

extern "C" Particle ParticleGet(GPU *gpu, const int position) {
	assert(position >= 0 && position < gpu->pCount);
	return gpu->hParticles[position];
}

extern "C" void ParticleUpload(GPU *gpu) {
#ifdef BUILD_CUDA
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);
		gpuErrchk(cudaMemcpyAsync(dev->Particles, &gpu->hParticles[dev->ParticleOffset], sizeof(Particle) * dev->ParticleCount, cudaMemcpyHostToDevice, dev->Stream));
	}

#ifdef BUILD_PERFORMANCE_PROFILE
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);
		cudaStreamSynchronize(dev->Stream);
	}
#endif
#endif
}

extern "C" void ParticleGenerate(GPU *gpu, const int processors, const int ncpus, const int seed, const double temperature, const double radius, const double qinfp) {
	const int particles_per_processor = gpu->pCount / processors;
	const int particles_remaining = gpu->pCount % processors;
	const double x_grid_change = gpu->FieldWidth / (double)ncpus, y_grid_change = gpu->FieldHeight / (double)ncpus;

	double xMin = 0.0, xMax = x_grid_change;
	double yMin = 0.0, yMax = y_grid_change;

	memset(gpu->hParticles, 0, sizeof(Particle) * gpu->pCount);

	int offset = 0;
	for(size_t processor = 0; processor < processors; processor++) {
		int particles = particles_per_processor;
		if(processor == 0) particles += particles_remaining;

		if(processor != 0 && processor % ncpus == 0) {
			xMin += x_grid_change;
			xMax += x_grid_change;

			yMin = 0.0;
			yMax = y_grid_change;
		}

		rand2_seed(seed);

		for(size_t i = 0; i < particles; i++) {
			gpu->hParticles[offset].pidx = processor * gpu->pCount + (i + 1);
			gpu->hParticles[offset].xp[0] = rand2() * (xMax - xMin) + xMin;
			gpu->hParticles[offset].xp[1] = rand2() * (yMax - yMin) + yMin;
			gpu->hParticles[offset].xp[2] = rand2() * (gpu->FieldDepth - 2.0 * radius) + radius;
			gpu->hParticles[offset].Tp = temperature;
			gpu->hParticles[offset].radius = radius;
			gpu->hParticles[offset].qinf = qinfp;

			offset++;
		}

		yMin += y_grid_change;
		yMax += y_grid_change;
	}

	ParticleUpload(gpu);
}

extern "C" void ParticleInterpolate(GPU *gpu, const double dx, const double dy) {
#ifdef BUILD_PERFORMANCE_PROFILE
	auto start = std::chrono::steady_clock::now();
#endif

#ifdef BUILD_CUDA
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);

		const unsigned int blocks = std::ceil(gpu->pCount / (double)CUDA_BLOCK_THREADS);
		if(gpu->mParameters.LinearInterpolation == 1) {
			GPUFieldInterpolateLinear<<<blocks, CUDA_BLOCK_THREADS, ((gpu->GridDepth * 2) + 2) * sizeof(double), dev->Stream>>>(gpu->GridWidth, gpu->GridHeight, dx, dy, gpu->GridDepth, dev->Z, dev->ZZ, dev->Uext, dev->Vext, dev->Wext, dev->Text, dev->Qext, dev->ParticleCount, dev->Particles);
		} else {
			GPUFieldInterpolate<<<blocks, CUDA_BLOCK_THREADS, gpu->GridDepth * 2 * sizeof(double), dev->Stream>>>(gpu->GridWidth, gpu->GridHeight, dx, dy, gpu->GridDepth, dev->Z, dev->ZZ, dev->Uext, dev->Vext, dev->Wext, dev->Text, dev->Qext, dev->ParticleCount, dev->Particles);
		}
		gpuErrchk(cudaPeekAtLastError());
	}

#ifdef BUILD_PERFORMANCE_PROFILE
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);
		cudaStreamSynchronize(dev->Stream);
	}
#endif
#else
	if(gpu->mParameters.LinearInterpolation == 1) {
		GPUFieldInterpolateLinear(gpu->GridWidth, gpu->GridHeight, dx, dy, gpu->GridDepth, gpu->hZ, gpu->hZZ, gpu->hUext, gpu->hVext, gpu->hWext, gpu->hText, gpu->hQext, gpu->pCount, gpu->hParticles);
	} else {
		GPUFieldInterpolate(gpu->GridWidth, gpu->GridHeight, dx, dy, gpu->GridDepth, gpu->hZ, gpu->hZZ, gpu->hUext, gpu->hVext, gpu->hWext, gpu->hText, gpu->hQext, gpu->pCount, gpu->hParticles);
	}
#endif

#ifdef BUILD_PERFORMANCE_PROFILE
	auto end = std::chrono::steady_clock::now();
	std::cout << "GPU Interpolate: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
#endif
}

extern "C" void ParticleStep(GPU *gpu, const int it, const int istage, const double dt) {
#ifdef BUILD_PERFORMANCE_PROFILE
	auto start = std::chrono::steady_clock::now();
#endif

#ifdef BUILD_CUDA
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);

		const unsigned int blocks = std::ceil(gpu->pCount / (double)CUDA_BLOCK_THREADS);
		GPUUpdateParticles<<<blocks, CUDA_BLOCK_THREADS, 0, dev->Stream>>>(it, istage - 1, dt, dev->ParticleCount, dev->Particles);
		gpuErrchk(cudaPeekAtLastError());
	}

#ifdef BUILD_PERFORMANCE_PROFILE
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);
		cudaStreamSynchronize(dev->Stream);
	}
#endif
#else
	GPUUpdateParticles(it, istage - 1, dt, gpu->pCount, gpu->hParticles);
#endif

#ifdef BUILD_PERFORMANCE_PROFILE
	auto end = std::chrono::steady_clock::now();
	std::cout << "GPU Step: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
#endif
}

extern "C" void ParticleUpdateNonPeriodic(GPU *gpu) {
#ifdef BUILD_PERFORMANCE_PROFILE
	auto start = std::chrono::steady_clock::now();
#endif

#ifdef BUILD_CUDA
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);

		const unsigned int blocks = std::ceil(gpu->pCount / (double)CUDA_BLOCK_THREADS);
		GPUUpdateNonperiodic<<<blocks, CUDA_BLOCK_THREADS, 0, dev->Stream>>>(gpu->FieldDepth, dev->ParticleCount, dev->Particles);
		gpuErrchk(cudaPeekAtLastError());
	}

#ifdef BUILD_PERFORMANCE_PROFILE
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);
		cudaStreamSynchronize(dev->Stream);
	}
#endif
#else
	GPUUpdateNonperiodic(gpu->FieldDepth, gpu->pCount, gpu->hParticles);
#endif

#ifdef BUILD_PERFORMANCE_PROFILE
	auto end = std::chrono::steady_clock::now();
	std::cout << "GPU NonPeriodic: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
#endif
}

extern "C" void ParticleUpdatePeriodic(GPU *gpu) {
#ifdef BUILD_PERFORMANCE_PROFILE
	auto start = std::chrono::steady_clock::now();
#endif

#ifdef BUILD_CUDA
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);

		const unsigned int blocks = std::ceil(gpu->pCount / (double)CUDA_BLOCK_THREADS);
		GPUUpdatePeriodic<<<blocks, CUDA_BLOCK_THREADS, 0, dev->Stream>>>(gpu->FieldWidth, gpu->FieldHeight, dev->ParticleCount, dev->Particles);
		gpuErrchk(cudaPeekAtLastError());
	}

#ifdef BUILD_PERFORMANCE_PROFILE
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);
		cudaStreamSynchronize(dev->Stream);
	}
#endif
#else
	GPUUpdatePeriodic(gpu->FieldWidth, gpu->FieldHeight, gpu->pCount, gpu->hParticles);
#endif

#ifdef BUILD_PERFORMANCE_PROFILE
	auto end = std::chrono::steady_clock::now();
	std::cout << "GPU Periodic: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
#endif
}

extern "C" void ParticleCalculateStatistics(GPU *gpu, const double dx, const double dy) {
#ifdef BUILD_PERFORMANCE_PROFILE
	auto start = std::chrono::steady_clock::now();
#endif
	memset(gpu->hPartCount, 0.0, sizeof(double) * gpu->GridDepth);
	memset(gpu->hVPSum, 0.0, sizeof(double) * gpu->GridDepth * 3);
	memset(gpu->hVPSumSQ, 0.0, sizeof(double) * gpu->GridDepth * 3);
	memset(gpu->hRPSum, 0.0, sizeof(double) * gpu->GridDepth);
	memset(gpu->hTPSum, 0.0, sizeof(double) * gpu->GridDepth);
	memset(gpu->hTFSum, 0.0, sizeof(double) * gpu->GridDepth);
	memset(gpu->hQFSum, 0.0, sizeof(double) * gpu->GridDepth);
	memset(gpu->hQSTARSum, 0.0, sizeof(double) * gpu->GridDepth);

#ifdef BUILD_CUDA
	ParticleDownload(gpu);
#endif
	GPUCalculateStatistics(gpu->GridDepth, gpu->hZ, gpu->hPartCount, gpu->hVPSum, gpu->hVPSumSQ, gpu->hRPSum, gpu->hTPSum, gpu->hTFSum, gpu-> hQFSum, gpu-> hQSTARSum, gpu->pCount, gpu->hParticles, gpu->radmean, gpu->radmin, gpu-> radmax);

#ifdef BUILD_PERFORMANCE_PROFILE
#ifdef BUILD_CUDA
	cudaDeviceSynchronize();
#endif
	auto end = std::chrono::steady_clock::now();
	std::cout << "GPU Statistics: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;
#endif
}

extern "C" void ParticleDownload(GPU *gpu) {
#ifdef BUILD_CUDA
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		Device *dev = GetDeviceMemory(gpu);
		gpuErrchk(cudaMemcpy(&gpu->hParticles[dev->ParticleOffset], dev->Particles, sizeof(Particle) * dev->ParticleCount, cudaMemcpyDeviceToHost));
	}
#endif
}

void ParticleWrite(GPU *gpu) {
	static int call = 0;
	static char buffer[80];
	sprintf(buffer, "c-particle-%d.dat", call);

	FILE *write_ptr = fopen(buffer, "wb");
	call += 1;

	fwrite(&gpu->pCount, sizeof(unsigned int), 1, write_ptr);
	for(int i = 0; i < gpu->pCount; i++) {
		fwrite(&gpu->hParticles[i], sizeof(Particle), 1, write_ptr);
	}

	fclose(write_ptr);
}

GPU *ParticleRead(const char *path) {
	FILE *data = fopen(path, "rb");

	unsigned int particles = 0;
	fread(&particles, sizeof(unsigned int), 1, data);

	double z[1], zz[1];

	Parameters params;
	GPU *retVal = NewGPU(particles, 0, 0, 0, 0.0, 0.0, 0.0, &z[0], &zz[0], &params);
	for(int i = 0; i < retVal->pCount; i++) {
		fread(&retVal->hParticles[i], sizeof(Particle), 1, data);
	}

	fclose(data);
	return retVal;
}

void PrintFreeMemory() {
#ifdef BUILD_CUDA
	size_t free_byte, total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
	if(cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
#endif
}

void ParticleFillStatistics(GPU *gpu, double *partCount, double *vSum, double *vSumSQ, double *rSum, double *tSum, double *tfSum, double *qfSum, double *qstarSum, double *single_stats) {
	for(size_t i = 0; i < gpu->GridDepth; i++) {
		partCount[i] = gpu->hPartCount[i];

		vSum[i * 3 + 0] = gpu->hVPSum[i * 3 + 0];
		vSum[i * 3 + 1] = gpu->hVPSum[i * 3 + 1];
		vSum[i * 3 + 2] = gpu->hVPSum[i * 3 + 2];

		vSumSQ[i * 3 + 0] = gpu->hVPSumSQ[i * 3 + 0];
		vSumSQ[i * 3 + 1] = gpu->hVPSumSQ[i * 3 + 1];
		vSumSQ[i * 3 + 2] = gpu->hVPSumSQ[i * 3 + 2];

                tSum[i] = gpu->hTPSum[i];
                rSum[i] = gpu->hRPSum[i];

                tfSum[i] = gpu->hTFSum[i];
                qfSum[i] = gpu->hQFSum[i];
                qstarSum[i] = gpu->hQSTARSum[i];
 	}
                single_stats[0] = gpu->radmean;
                single_stats[1] = gpu->radmin;
                single_stats[2] = gpu->radmax;
}

// Particle Functions
std::ostream &operator<<(std::ostream &stream, const Particle &p) {
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
const std::vector<double> ReadDoubleArray(const std::string &path) {
	std::vector<double> retVal;

	std::ifstream iStream(path.c_str(), std::ifstream::in | std::ifstream::binary);
	if(iStream.fail()) {
		std::cerr << "Unable to open " << path << " to read from.";
		return retVal;
	}

	unsigned int size = 0;
	iStream >> size;

	retVal.resize(size);
	for(unsigned int i = 0; i < size; i++) {
		iStream >> retVal[i];
	}
	iStream.close();

	return retVal;
}

void WriteDoubleArray(const std::string &path, const std::vector<double> &array) {
	std::ofstream oStream(path.c_str(), std::ofstream::out | std::ofstream::binary);
	if(oStream.fail()) {
		std::cerr << "Unable to open " << path << " to write to.";
		return;
	}

	oStream << array.size();
	for(unsigned int i = 0; i < array.size(); i++) {
		oStream << array[i];
	}
	oStream.close();
}

// Test Helper Functions
void SetParameters(GPU *gpu, const Parameters *params) {
	memcpy(&gpu->mParameters, params, sizeof(Parameters));

#ifdef BUILD_CUDA
	for(size_t i = 0; i < gpudevices(); i++) {
		SetDeviceIndex(gpu, i);
		gpuErrchk(cudaMemcpyToSymbol(cParams, &gpu->mParameters, sizeof(Parameters)));
	}
#else
	memcpy(&cParams, &gpu->mParameters, sizeof(Parameters));
#endif
}
