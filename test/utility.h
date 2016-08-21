#ifndef TEST_UTILITY_H_
#define TEST_UTILITY_H_

#include <string>
#include <vector>

struct Particle {
	int pidx, procidx;
	double vp[3], xp[3], uf[3], xrhs[3], vrhs[3];
	double Tp, Tprhs_s, Tprhs_L, Tf, radius, radrhs, qinf, qstar;
};

std::vector<Particle> ReadParticles( std::string path );

#endif // TEST_UTILITY_H_