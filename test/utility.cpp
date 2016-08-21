#include "utility.h"
#include "gtest/gtest.h"

#include <string>
#include <sstream>
#include <fstream>

std::vector<Particle> ReadParticles( std::string path ) {
	std::vector<Particle> retVal;

	std::ifstream in( path.c_str(), std::ios::in | std::ios::binary );
	if( in.good() ) {
		int count = 0;
		in.read( ( char * )&count, sizeof( int ) );

		for( int i = 0; i < count; i++ ) {
			Particle current;

			in.read( ( char * )&current.pidx, sizeof( unsigned int ) );
			in.read( ( char * )&current.procidx, sizeof( unsigned int ) );

			in.read( ( char * )&current.vp[0], sizeof( double ) );
			in.read( ( char * )&current.vp[1], sizeof( double ) );
			in.read( ( char * )&current.vp[2], sizeof( double ) );

			in.read( ( char * )&current.xp[0], sizeof( double ) );
			in.read( ( char * )&current.xp[1], sizeof( double ) );
			in.read( ( char * )&current.xp[2], sizeof( double ) );

			in.read( ( char * )&current.uf[0], sizeof( double ) );
			in.read( ( char * )&current.uf[1], sizeof( double ) );
			in.read( ( char * )&current.uf[2], sizeof( double ) );

			in.read( ( char * )&current.xrhs[0], sizeof( double ) );
			in.read( ( char * )&current.xrhs[1], sizeof( double ) );
			in.read( ( char * )&current.xrhs[2], sizeof( double ) );

			in.read( ( char * )&current.vrhs[0], sizeof( double ) );
			in.read( ( char * )&current.vrhs[1], sizeof( double ) );
			in.read( ( char * )&current.vrhs[2], sizeof( double ) );

			in.read( ( char * )&current.Tp, sizeof( double ) );
			in.read( ( char * )&current.Tprhs_s, sizeof( double ) );
			in.read( ( char * )&current.Tprhs_L, sizeof( double ) );
			in.read( ( char * )&current.Tf, sizeof( double ) );
			in.read( ( char * )&current.radius, sizeof( double ) );
			in.read( ( char * )&current.radrhs, sizeof( double ) );
			in.read( ( char * )&current.qinf, sizeof( double ) );
			in.read( ( char * )&current.qstar, sizeof( double ) );

			int dummy = 0;
			in.read( ( char * )&dummy, sizeof( int ) );
			in.read( ( char * )&dummy, sizeof( int ) );
			in.read( ( char * )&dummy, sizeof( int ) );
			in.read( ( char * )&dummy, sizeof( int ) );

			retVal.push_back( current );
		}
	}

	return retVal;
}

TEST( ParticleUtility, ParseParticles ) {
	std::vector<Particle> particle = ReadParticles( "../test/data/particle_input.dat" );
	EXPECT_EQ( particle.size(), 10 );
}