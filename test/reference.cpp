#include "gtest/gtest.h"

TEST(ParticleReference, SimplePass) {
    EXPECT_EQ(1, 1);
}

TEST(ParticleReference, SimpleFail) {
    EXPECT_EQ(1, 0);
}
