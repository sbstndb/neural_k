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

	static Kokkos::Random_XorShift64_Pool<> rand_pool ;

	Layer(int input_size, int output_size) {
		weights = View2D("weights", input_size, output_size) ; 
		biases = View1D("biases", output_size) ; 
		output = View1D("output", output_size) , 

		      // init with random
		Kokkos::fill_random(weights, rand_pool, -0.3, 0.3);
		Kokkos::fill_random(biases, rand_pool, -0.1, 0.1);
	}

	void forward(const View1D& input){
		KokkosBlas::gemv("N", 1.0, weights, input, 0.0, output); 
		Kokkos::parallel_for("add_biases_and_sigmoid", output.extent(0), KOKKOS_LAMBDA(int i) {
			output(i) = 1.0 / (1.0 + std::exp(-(output(i) + biases(i)))); 
			});
	}

//	~Layer() ; 
};


class OutputLayer : public Layer {
public : 
	OutputLayer(int input_size, int output_size) : Layer(input_size, output_size) {}

	void forward(const View1D input) {
		Layer::forward(input) ;
		// apply softmax in this example
		real sum = 0.0 ; 
		// softmax pour classement n 
		Kokkos::parallel_reduce("softmax_sum", output.extent(0), KOKKOS_LAMBDA (int i, real& local_sum){
			output(i) = exp(output(i)) ; 
			local_sum += output(i) ; 
		}, sum);
		Kokkos::parallel_for("softmax_normalize", output.extent(0), KOKKOS_LAMBDA (int i){
			output(i) /= sum ; 
		});
	}
};

// attention : il faut utiliser un shared pour le outer non ? 
// A voir comment ca se goupille en cuda...
using innerView = Kokkos::View<Layer> ; 
// here : how to ensure automatic CUDAUVMSpave or HIPUVMSpace handling ????
using outerView = Kokkos::View<innerView*, Kokkos::HostSpace> ; 

class DenseNeuralNetwork {
public:


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


