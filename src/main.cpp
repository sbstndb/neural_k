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

	Kokkos::Random_XorShift64_Pool<> rand_pool ;

	Layer(int input_size, int output_size):
		rand_pool(std::time(0))       
	{


		weights = View2D("weights", output_size, input_size) ; 
		biases = View1D("biases", output_size) ; 
		output = View1D("output", output_size) , 
//
		Kokkos::parallel_for("initialize_weights", output_size * input_size, KOKKOS_LAMBDA(int idx){
			int i = idx / input_size ; 
			int j = idx % input_size ; 
			weights(i,j) = -1.0 ; 
		});

		Kokkos::parallel_for("initialize_other", output_size, KOKKOS_LAMBDA(int i){
			biases(i) = -1.0 ; 
		});

		      // init with random, for now not with kokkos
//		Kokkos::fill_random(weights, rand_pool, -0.0, 0.0);
//		Kokkos::fill_random(biases, rand_pool, -0.0, 0.0);
	}

	void forward(const View1D& input){
        std::cout << "-- Forward in hidden Layer --" << std::endl ;
		KokkosBlas::gemv("N", 1.0, weights, input, 0.0, output); 
		Kokkos::parallel_for("add_biases_and_sigmoid", output.extent(0), KOKKOS_LAMBDA(int i) {
			output(i) = 1.0 / (1.0 + std::exp(-(output(i) + biases(i)))); 
			});
	}

//	~Layer() ; 
};


//Kokkos::Random_XorShift64_Pool<> Layer::rand_pool(std::time(0)) ;


class OutputLayer : public Layer {
public : 
	OutputLayer(int input_size, int output_size) : Layer(input_size, output_size) {}

	void forward(const View1D input) {
        std::cout << "-- Forward in Output Layer --" << std::endl ; 
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
	Layer layer1 ; 
	Layer layer2 ; 
	OutputLayer output_layer ; 

	DenseNeuralNetwork(int n1, int n2, int n3):
		layer1(n1, n2), layer2(n2, n3), output_layer(n3, n3) {}

	View1D forward(const View1D& input){
        std::cout << "-- Forward in Neural Network " << std::endl ;
		layer1.forward(input) ; 
		layer2.forward(layer1.output) ; 
		output_layer.forward(layer2.output) ; 
        std::cout << "-- End     in Neural Network " << std::endl ;
		
		return output_layer.output ; 
	}
};

int main(int argc, char **argv) {
	std::cout << "-- Neural Library --" << std::endl;
	Kokkos::initialize(argc, argv);
	{/////////////////////////
	int n1 = 2 ; 
	int n2 = 4 ; 
	int n3 = 1 ; 
	DenseNeuralNetwork dnn(n1, n2, n3) ; 
	real xor_inputs[4][2] = {
		{0.0,0.0},
		{0.0,1.0},
		{1.0,0.0},
		{1.0,1.0}};
	real xor_outputs[4] = {0.0,1.0,1.0,0.0};
	// entrainement
	// todo
	//
	// inference
	for (int i = 0 ; i < 4 ; i++){
		View1D input("input", n1) ; 
		input(0) = xor_inputs[i][0] ; 
		input(1) = xor_inputs[i][1] ;
		auto output = dnn.forward(input) ;
		std::cout << "Input : " << xor_inputs[i][0] << " -- " << xor_inputs[i][1] << std::endl ; 
		std::cout << "--> Predicted : " << output(0) << std::endl ; 
	}
	}/////////////////////////
	Kokkos::finalize();
	std::cout << "-- Brain completed ;-) --" << std::endl;
	return 0;
}


