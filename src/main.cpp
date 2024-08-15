#include <iostream>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <filesystem>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include "../external/HighFive/include/highfive/highfive.hpp"

using real = float;

int main(int argc, char **argv) {
    std::cout << "-- Neural_k library -- "<< std::endl;
    Kokkos::initialize(argc, argv);
    {


    }
    Kokkos::finalize();

    std::cout << "-- Neural_k work completed ;-) --" << std::endl;
    return 0;
}


