#include "gtest/gtest.h"
#include <cmath>

#include "utility.h"

void UpdateParticles( const int it, const int istage, const double dt, std::vector<Particle> &particles ) {
	const double uext = 1.0;
	const double vext = 0.0;
	const double wext = 0.0;
	const double Text = 295.0;
	const double T2ext = 0.0164;

	const double partcount_t = 0.0;
	const double vpsum_t = 0.0;
	const double upwp_t = 0.0;
	const double vpsqrsum_t = 0.0;
	const double Tpsum_t = 0.0;
	const double Tfsum_t = 0.0;
	const double qfsum_t = 0.0;
	const double radsum_t = 0.0;
	const double rad2sum_t = 0.0;
	const double mpsum_t = 0.0;
	const double mwsum_t = 0.0;
	const double Tpsqrsum_t = 0.0;
	const double wpTpsum_t = 0.0;
	double myRep_avg = 0.0;
	double myphip_sum = 0.0;
	double myphiw_sum = 0.0;
	double myphiv_sum = 0.0;
	const double qstarsum_t = 0.0;
	const double ievap = 1;

	const double Gam = 7.28 * std::pow( 10.0, -2 );
	const double Ion = 2.0;
	const double Os = 1.093;
	const double rhoa = 1.1;
	const double rhow = 1000.0;
	const double nuf = 1.537e-5;
	const double pi = 4.0 * std::atan( 1.0 );
	const double pi2 = 2.0 * pi;
	const double Sal = 34.0;
	const double radius_mass = 40.0e-6;
	const double m_s = Sal / 1000.0 * 4.0 / 3.0 * pi * std::pow( radius_mass, 3 ) * rhow;
	const double Pra = 0.715;
	const double Sc = 0.615;
	const double Mw = 0.018015;
	const double Ru = 8.3144621;
	const double Ms = 0.05844;
	const double Cpa = 1006.0;
	const double Cpp = 4179.0;
	const double CpaCpp = Cpa / Cpp;
	const double part_grav = 0.0;

	const double zetas[3] = {0.0, -17.0 / 60.0, -5.0 / 12.0};
	const double gama[3] = {8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0};
	const double g[3] = {0.0, 0.0, part_grav};

	for( int i = 0; i < particles.size(); i++ ) {
		Particle *part = &particles[i];

		if( it < 1 ) {
			for( int j = 0; j < 3; j++ ) {
				part->vp[j] = part->uf[j];
			}
			part->Tp = part->Tf;
		}

		double diff[3];
		for( int j = 0; j < 3; j++ ) {
			diff[j] = part->vp[j] - part->uf[j];
		}

		double diffnorm = std::sqrt( diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] );
		double Rep = 2.0 * part->radius * diffnorm / nuf;
		double Volp = pi2 * 2.0 / 3.0 * ( part->radius * part->radius * part->radius );
		double rhop = ( m_s + Volp * rhow ) / Volp;
		double taup_i = 18.0 * rhoa * nuf / rhop / ( ( 2.0 * part->radius ) * ( 2.0 * part->radius ) );

		double corrfac = 1.0 + std::pow( 0.15 * Rep, 0.687 );

		double Nup = 2.0 + std::pow( 0.6 * Rep, 0.5 ) * std::pow( Pra, 1.0 / 3.0 );
		double Shp = 2.0 + std::pow( 0.6 * Rep, 0.5 ) * std::pow( Sc, 1.0 / 3.0 );

		double TfC = part->Tf - 273.15;
		double einf = 610.94 * ( 17.6257 * TfC / ( TfC + 243.04 ) );
		double TpC = part->Tp - 273.15;
		double Lv = ( 25.0 - 0.02274 * 26.0 ) * 100000;
		double Eff_C = 2.0 * Mw * Gam / ( Ru * rhow * part->radius * part->Tp );
		double Eff_S = Ion * Os * m_s * Mw / Ms / ( Volp * rhop - m_s );
		double estar = einf * ( Mw * Lv / Ru * ( 1.0 / part->Tf - 1.0 / part->Tp ) + Eff_C - Eff_S );
		part->qstar = Mw / Ru * estar / part->Tp / rhoa;

		double xtmp[3], vtmp[3];
		for( int j = 0; j < 3; j++ ) {
			xtmp[j] = part->xp[j] + dt * zetas[istage] * part->xrhs[j];
			vtmp[j] = part->vp[j] + dt * zetas[istage] * part->vrhs[j];
		}

		double Tptmp = part->Tp + dt * zetas[istage] * part->Tprhs_s;
		Tptmp = Tptmp + dt * zetas[istage] * part->Tprhs_L;
		double radiustmp = part->radius + dt * zetas[istage] * part->radrhs;

		for( int j = 0; j < 3; j++ ) {
			part->xrhs[j] = part->vp[j];
		}

		for( int j = 0; j < 3; j++ ) {
			part->vrhs[j] = corrfac * taup_i * ( part->uf[j] - part->vp[j] ) - g[j];
		}

		if( ievap == 1 ) {
			part->radrhs = Shp / 9.0 / Sc * rhop / rhow * part->radius * taup_i * ( part->qinf - part->qstar );
		} else {
			part->radrhs = 0.0;
		}

		part->Tprhs_s = -Nup / 3.0 / Pra * CpaCpp * rhop / rhow * taup_i * ( part->Tp - part->Tf );
		part->Tprhs_L = 3.0 * Lv / Cpp / part->radius * part->radrhs;

		for( int j = 0; j < 3; j++ ) {
			part->xp[j] = xtmp[j] + dt * gama[istage] * part->xrhs[j];
			part->vp[j] = vtmp[j] + dt * gama[istage] * part->vrhs[j];
		}

		part->Tp = Tptmp + dt * gama[istage] * part->Tprhs_s;
		part->Tp = part->Tp + dt * gama[istage] * part->Tprhs_L;
		part->radius = radiustmp + dt * gama[istage] * part->radrhs;
	}
}

TEST( ParticleReference, ParticleUpdate ) {
	std::vector<Particle> input = ReadParticles( "../test/data/particle_input.dat" );
	ASSERT_EQ( input.size(), 10 );

	std::vector<Particle> expected = ReadParticles( "../test/data/particle_expected.dat" );
	ASSERT_EQ( expected.size(), 10 );

	UpdateParticles( 500, 2, 3.556534376545218E-4, input );

	for( int i = 0; i < 10; i++ ) {
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].vp[j], input[i].vp[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].xp[j], input[i].xp[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].uf[j], input[i].uf[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].xrhs[j], input[i].xrhs[j] ) << "I: " << i << " J: " << j;
		}
		for( int j = 0; j < 3; j++ ) {
			ASSERT_FLOAT_EQ( expected[i].vrhs[j], input[i].vrhs[j] ) << "I: " << i << " J: " << j;
		}

		ASSERT_FLOAT_EQ( expected[i].Tp, input[i].Tp ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].Tprhs_s, input[i].Tprhs_s ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].Tprhs_L, input[i].Tprhs_L ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].Tf, input[i].Tf ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].radius, input[i].radius ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].radrhs, input[i].radrhs ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].qinf, input[i].qinf ) << "I: " << i;
		ASSERT_FLOAT_EQ( expected[i].qstar, input[i].qstar ) << "I: " << i;
	}
}
