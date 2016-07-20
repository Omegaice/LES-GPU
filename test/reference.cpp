#include "gtest/gtest.h"
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>

struct Particle {
	int pidx, procidx;
	double vp[3], xp[3], uf[3], xrhs[3], vrhs[3];
	double Tp, Tprhs_s, Tprhs_L, Tf, radius, radrhs, qinf, qstar;
};

std::vector<Particle *> ReadParticles( std::string path ) {
	std::vector<Particle *> retVal;

	std::ifstream in( path.c_str(), std::ios::in | std::ios::binary );
    if( in.good() ){
        int count = 0;
        in.read ((char*)&count, sizeof(int));

        for( int i = 0; i < count; i++ ){
            Particle *current = new Particle();

            in.read ((char*)&current->pidx, sizeof(unsigned int));
            in.read ((char*)&current->procidx, sizeof(unsigned int));

            in.read ((char*)&current->vp[0], sizeof(double));
            in.read ((char*)&current->vp[1], sizeof(double));
            in.read ((char*)&current->vp[2], sizeof(double));

            in.read ((char*)&current->xp[0], sizeof(double));
            in.read ((char*)&current->xp[1], sizeof(double));
            in.read ((char*)&current->xp[2], sizeof(double));

            in.read ((char*)&current->uf[0], sizeof(double));
            in.read ((char*)&current->uf[1], sizeof(double));
            in.read ((char*)&current->uf[2], sizeof(double));

            in.read ((char*)&current->xrhs[0], sizeof(double));
            in.read ((char*)&current->xrhs[1], sizeof(double));
            in.read ((char*)&current->xrhs[2], sizeof(double));

            in.read ((char*)&current->vrhs[0], sizeof(double));
            in.read ((char*)&current->vrhs[1], sizeof(double));
            in.read ((char*)&current->vrhs[2], sizeof(double));

            in.read ((char*)&current->Tp, sizeof(double));
            in.read ((char*)&current->Tprhs_s, sizeof(double));
            in.read ((char*)&current->Tprhs_L, sizeof(double));
            in.read ((char*)&current->Tf, sizeof(double));
            in.read ((char*)&current->radius, sizeof(double));
            in.read ((char*)&current->radrhs, sizeof(double));
            in.read ((char*)&current->qinf, sizeof(double));
            in.read ((char*)&current->qstar, sizeof(double));

            int dummy = 0;
            in.read ((char*)&dummy, sizeof(int));
            in.read ((char*)&dummy, sizeof(int));
            in.read ((char*)&dummy, sizeof(int));
            in.read ((char*)&dummy, sizeof(int));

            retVal.push_back(current);
        }
    }

	return retVal;
}

void UpdateParticles( const int it, const int istage, const double dt, std::vector<Particle *> &particles ) {
	auto uext = 1.0;
	auto vext = 0.0;
	auto wext = 0.0;
	auto Text = 295.0;
	auto T2ext = 0.0164;

	auto partcount_t = 0.0;
	auto vpsum_t = 0.0;
	auto upwp_t = 0.0;
	auto vpsqrsum_t = 0.0;
	auto Tpsum_t = 0.0;
	auto Tfsum_t = 0.0;
	auto qfsum_t = 0.0;
	auto radsum_t = 0.0;
	auto rad2sum_t = 0.0;
	auto mpsum_t = 0.0;
	auto mwsum_t = 0.0;
	auto Tpsqrsum_t = 0.0;
	auto wpTpsum_t = 0.0;
	auto myRep_avg = 0.0;
	auto myphip_sum = 0.0;
	auto myphiw_sum = 0.0;
	auto myphiv_sum = 0.0;
	auto qstarsum_t = 0.0;
    auto ievap = 1;

	auto Gam = 7.28 * std::pow( 10.0, -2 );
	auto Ion = 2.0;
	auto Os = 1.093;
	auto rhoa = 1.1;
	auto rhow = 1000.0;
	auto nuf  = 1.537e-5;
	auto pi   = 4.0 * std::atan( 1.0 );
	auto pi2  = 2.0 * pi;
	auto Sal = 34.0;
	auto radius_mass = 40.0e-6;
	auto m_s = Sal / 1000.0 * 4.0 / 3.0 * pi * std::pow(radius_mass, 3) * rhow;
    auto Pra = 0.715;
    auto Sc = 0.615;
    auto Mw = 0.018015;
    auto Ru = 8.3144621;
    auto Ms = 0.05844;
    auto Cpa = 1006.0;
    auto Cpp = 4179.0;
    auto CpaCpp = Cpa/Cpp;
    auto part_grav = 0.0;

    double zetas[3] = {0.0, -17.0/60.0, -5.0/12.0};
    double gama[3]  = {8.0/15.0, 5.0/12.0, 3.0/4.0};
    double g[3] = {0.0, 0.0, part_grav};

	for( int i = 0; i < particles.size(); i++ ) {
		Particle *part = particles[i];

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
		auto diffnorm = std::sqrt( diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] );
		auto Rep = 2.0 * part->radius * diffnorm / nuf;
		auto Volp = pi2 * 2.0 / 3.0 * std::pow( part->radius, 3 );
		auto rhop = ( m_s + Volp * rhow ) / Volp;
		auto taup_i = 18.0 * rhoa * nuf / rhop / std::pow( 2.0 * part->radius, 2 );

		myRep_avg = myRep_avg + Rep;
		auto corrfac = 1.0 + std::pow( 0.15 * Rep, 0.687 );
		myphip_sum = myphip_sum + Volp * rhop;
		myphiw_sum = myphiw_sum + Volp * rhow;
		myphiv_sum = myphiv_sum + Volp;

		auto Nup = 2.0 + std::pow( 0.6 * Rep, 0.5 ) * std::pow( Pra, 1.0 / 3.0 );
		auto Shp = 2.0 + std::pow( 0.6 * Rep, 0.5 ) * std::pow( Sc, 1.0 / 3.0 );

		auto TfC = part->Tf - 273.15;
		auto einf = 610.94 * std::exp( 17.6257 * TfC / ( TfC + 243.04 ) );
		auto TpC = part->Tp - 273.15;
		auto Lv = ( 25.0 - 0.02274 * 26.0 ) * std::pow( 10.0, 5 );
		auto Eff_C = 2.0 * Mw * Gam / ( Ru * rhow * part->radius * part->Tp );
		auto Eff_S = Ion * Os * m_s * Mw / Ms / ( Volp * rhop - m_s );
		auto estar = einf * exp( Mw * Lv / Ru * ( 1.0 / part->Tf - 1.0 / part->Tp ) + Eff_C - Eff_S );
		part->qstar = Mw / Ru * estar / part->Tp / rhoa;

        double xtmp[3], vtmp[3];
		for( int j = 0; j < 3; j++ ) {
            xtmp[j] = part->xp[j] + dt * zetas[istage] * part->xrhs[j];
            vtmp[j] = part->vp[j] + dt * zetas[istage] * part->vrhs[j];
		}
		auto Tptmp = part->Tp + dt * zetas[istage] * part->Tprhs_s;
		Tptmp = Tptmp + dt * zetas[istage] * part->Tprhs_L;
		auto radiustmp = part->radius + dt * zetas[istage] * part->radrhs;

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

TEST( ParticleReference, ParseParticles ) {
	std::vector<Particle *> particle = ReadParticles( "../test/data/particle_input.dat" );
	EXPECT_EQ( particle.size(), 10 );
}

TEST( ParticleReference, ParticleUpdate ) {
	std::vector<Particle *> input = ReadParticles( "../test/data/particle_input.dat" );
	EXPECT_EQ( input.size(), 10 );

	std::vector<Particle *> expected = ReadParticles( "../test/data/particle_expected.dat" );
	EXPECT_EQ( expected.size(), 10 );

	UpdateParticles(500, 3, 3.556534376545218E-004, input);

    for( int i = 0; i < 10; i++ ){
        EXPECT_EQ( expected[i]->pidx, input[i]->pidx ) << "I: " << i;
        EXPECT_EQ( expected[i]->procidx, input[i]->procidx ) << "I: " << i;

        for( int j = 0; j < 3; j++ ) ASSERT_FLOAT_EQ( expected[i]->vp[j], input[i]->vp[j] ) << "I: " << i << " J: " << j;
        for( int j = 0; j < 3; j++ ) ASSERT_FLOAT_EQ( expected[i]->xp[j], input[i]->xp[j] ) << "I: " << i << " J: " << j;
        for( int j = 0; j < 3; j++ ) ASSERT_FLOAT_EQ( expected[i]->uf[j], input[i]->uf[j] ) << "I: " << i << " J: " << j;
        for( int j = 0; j < 3; j++ ) ASSERT_FLOAT_EQ( expected[i]->xrhs[j], input[i]->xrhs[j] ) << "I: " << i << " J: " << j;
        for( int j = 0; j < 3; j++ ) ASSERT_FLOAT_EQ( expected[i]->vrhs[j], input[i]->vrhs[j] ) << "I: " << i << " J: " << j;

        ASSERT_FLOAT_EQ( expected[i]->Tp, input[i]->Tp ) << "I: " << i;
        ASSERT_FLOAT_EQ( expected[i]->Tprhs_s, input[i]->Tprhs_s ) << "I: " << i;
        ASSERT_FLOAT_EQ( expected[i]->Tprhs_L, input[i]->Tprhs_L ) << "I: " << i;
        ASSERT_FLOAT_EQ( expected[i]->Tf, input[i]->Tf ) << "I: " << i;
        ASSERT_FLOAT_EQ( expected[i]->radius, input[i]->radius ) << "I: " << i;
        ASSERT_FLOAT_EQ( expected[i]->radrhs, input[i]->radrhs ) << "I: " << i;
        ASSERT_FLOAT_EQ( expected[i]->qinf, input[i]->qinf ) << "I: " << i;
        ASSERT_FLOAT_EQ( expected[i]->qstar, input[i]->qstar ) << "I: " << i;
    }
}
