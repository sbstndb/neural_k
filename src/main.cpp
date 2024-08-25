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
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp>
#include "../external/HighFive/include/highfive/highfive.hpp"

using real = float;
using View1D = Kokkos::View<real *  >;
using View2D = Kokkos::View<real ** >;
using View3D = Kokkos::View<real ***>;


class Layer  {
public : 
	View2D weights ; 
	View1D biases ; 
	View1D output ; 

	Layer(int input_size, int output_size) {
		weights = View2D("weights", input_size, output_size) ; 
		biases = View1D("biases", output_size) ; 
		output = View1D("output", output_size) , 

		      // init with random
		Kokkos::RandomXorShift64_Pool<> rand_pool(std::time(0)) ; 
		Kokkos::fill_random(weights, rand_pool, -0.3, 0.3);
		Kokkos::fill_random(biases, rand_pool, -0.1, 0.1);
	}

	void forward(const View1D& output){
		KokkosBlas::gemv("N", 1.0, weights, input, 0.0, output); 
		Kokkos::parallel_for("add_biases", output.extend(0), KOKKOS_LAMBDA(int i) {
			output(i) += biases(i) ; 
			});
	}


};







int main(int argc, char **argv) {
    std::cout << "-- Neural Library --" << std::endl;

    Kokkos::initialize(argc, argv);
    {
    }
    Kokkos::finalize();

    std::cout << "-- Brain completed ;-) --" << std::endl;
    return 0;
}


