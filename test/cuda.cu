#include "gtest/gtest.h"
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>

#include "utility.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void GPUUpdateParticles( const int it, const int istage, const double dt, const int pcount, Particle* particles ) {
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
    const double Ru = 8.3144621;
    const double Ms = 0.05844;
    const double Cpa = 1006.0;
    const double Cpp = 4179.0;
    const double CpaCpp = Cpa/Cpp;
    const double part_grav = 0.0;

    const double zetas[3] = {0.0, -17.0/60.0, -5.0/12.0};
    const double gama[3]  = {8.0/15.0, 5.0/12.0, 3.0/4.0};
    const double g[3] = {0.0, 0.0, part_grav};

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx > pcount ) return;

    if( it < 1 ) {
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
    double Volp = pi2 * 2.0 / 3.0 * ( particles[idx].radius * particles[idx].radius * particles[idx].radius);
    double rhop = ( m_s + Volp * rhow ) / Volp;
    double taup_i = 18.0 * rhoa * nuf / rhop / ( (2.0 * particles[idx].radius) * (2.0 * particles[idx].radius) );

    double corrfac = 1.0 + pow( 0.15 * Rep, 0.687 );
    double Nup = 2.0 + pow( 0.6 * Rep, 0.5 ) * pow( Pra, 1.0 / 3.0 );
    double Shp = 2.0 + pow( 0.6 * Rep, 0.5 ) * pow( Sc, 1.0 / 3.0 );

    double TfC = particles[idx].Tf - 273.15;
    double einf = 610.94 * ( 17.6257 * TfC / ( TfC + 243.04 ) );
    double Lv = ( 25.0 - 0.02274 * 26.0 ) * 100000;
    double Eff_C = 2.0 * Mw * Gam / ( Ru * rhow * particles[idx].radius * particles[idx].Tp );
    double Eff_S = Ion * Os * m_s * Mw / Ms / ( Volp * rhop - m_s );
    double estar = einf * ( Mw * Lv / Ru * ( 1.0 / particles[idx].Tf - 1.0 / particles[idx].Tp ) + Eff_C - Eff_S );
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

TEST( ParticleCUDA, ParticleUpdate ) {
    std::vector<Particle> input = ReadParticles( "../test/data/particle_input.dat" );
	ASSERT_EQ( input.size(), 10 );

	std::vector<Particle> expected = ReadParticles( "../test/data/particle_expected.dat" );
	ASSERT_EQ( expected.size(), 10 );

    Particle *dParticles;
	gpuErrchk( cudaMalloc( (void **)&dParticles, sizeof(Particle) * 10) );
    gpuErrchk( cudaMemcpy(dParticles, &input[0], sizeof(Particle) * 10, cudaMemcpyHostToDevice) );

    GPUUpdateParticles<<< 1, 12 >>> (500, 2, 3.556534376545218E-4, 10, dParticles);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    Particle *result = new Particle[10];
    gpuErrchk( cudaMemcpy(result, dParticles, sizeof(Particle) * 10, cudaMemcpyDeviceToHost) );

    for( int i = 0; i < 10; i++ ) {
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].vp[j], result[i].vp[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].xp[j], result[i].xp[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].uf[j], result[i].uf[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].xrhs[j], result[i].xrhs[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].vrhs[j], result[i].vrhs[j] ) << "I: " << i << " J: " << j;
		}

		ASSERT_FLOAT_EQ( expected[i].Tp, result[i].Tp ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].Tprhs_s, result[i].Tprhs_s ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].Tprhs_L, result[i].Tprhs_L ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].Tf, result[i].Tf ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].radius, result[i].radius ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].radrhs, result[i].radrhs ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].qinf, result[i].qinf ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].qstar, result[i].qstar ) << "I: " << i;
	}
}
