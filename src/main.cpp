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
	View1D z ; // before activation
	View1D output ; // similar to activation in my case

	Kokkos::Random_XorShift64_Pool<> rand_pool ;

	Layer(int input_size, int output_size):
		rand_pool(std::time(0))       
	{


		weights = View2D("weights", output_size, input_size) ; 
		biases = View1D("biases", output_size) ; 
		z = View1D("z", output_size) ; 
		output = View1D("output", output_size) , 
//
		Kokkos::fill_random(weights, rand_pool, -0.1, 0.1);
		Kokkos::fill_random(biases, rand_pool, -0.1, 0.1);
	}

	void forward(const View1D& input){
		KokkosBlas::gemv("N", 1.0, weights, input, 0.0, z); 
                Kokkos::parallel_for("add_biases", output.extent(0), KOKKOS_LAMBDA(int i) {
                        z(i)+= biases(i);
                });
		Kokkos::parallel_for("add_sigmoid", output.extent(0), KOKKOS_LAMBDA(int i) {
			output(i) = 1.0 / (1.0 + std::exp(-z(i))); 
		});
	}

	void show(){
		std::cout << "Weights or layer : w(i,j) :" << std::endl; 
		for (int i = 0 ; i < weights.extent(0); i++){
			std::cout << "   "  ; 
			for (int j = 0 ; j < weights.extent(1); j++){
				std::cout << weights(i,j) << " " ; 
			}
			std::cout << std::endl;
		}
                std::cout << "Biases or layer : b(i) :" << std::endl ;
		std::cout << "   " ; 
                for (int i = 0 ; i < biases.extent(0); i++){
                        std::cout << biases(i) << " " ;
                }
		std::cout << std::endl ; 


	}


//	~Layer() ; 
};


//Kokkos::Random_XorShift64_Pool<> Layer::rand_pool(std::time(0)) ;


class OutputLayer : public Layer {
public : 
	OutputLayer(int input_size, int output_size) : Layer(input_size, output_size) {}

	void forward(const View1D input) {
		Layer::forward(input) ;
		// apply softmax in this example
//		real sum = 0.0 ; 
		// softmax pour classement n 
//		Kokkos::parallel_reduce("softmax_sum", output.extent(0), KOKKOS_LAMBDA (int i, real& local_sum){
//			output(i) = exp(output(i)) ; 
//			local_sum += output(i) ; 
//		}, sum);
//		Kokkos::parallel_for("softmax_normalize", output.extent(0), KOKKOS_LAMBDA (int i){
//			output(i) /= sum ; 
//		});
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

	DenseNeuralNetwork(int n1, int n2, int n3, int nsortie):
		layer1(n1, n2), layer2(n2, n3), output_layer(n3, nsortie) {}

	View1D forward(const View1D& input){
		layer1.forward(input) ; 
		layer2.forward(layer1.output) ; 
		output_layer.forward(layer2.output) ; 
		
		return output_layer.output ; 
	}


	void train( const View1D input, const View1D target, real learning_rate){
		auto prediction = forward(input)  ;
		// calcul de gradient
		//
		// output layer
		View1D delta_output("delta_output", output_layer.output.extent(0));
		Kokkos::parallel_for("compute_output_error", prediction.extent(0), KOKKOS_LAMBDA(int i){
				real output_val = output_layer.output(i) ; 
				real z_val = output_layer.z(i) ; 
				delta_output(i) = (output_val - target(i)) * output_val * (1 - output_val) ; 
		});
		Kokkos::parallel_for("update_output_weights", output_layer.weights.extent(0), KOKKOS_LAMBDA (int i){
			for (int j = 0 ; j < output_layer.weights.extent(1); j++){
				output_layer.weights(i,j) -= learning_rate * delta_output(i) * layer2.output(j) ; 
			}
			output_layer.biases(i) -= learning_rate * delta_output(i) ; 
		});

		// layer 2
		View1D delta_layer2("delta_layer2", layer2.output.extent(0));
                Kokkos::parallel_for("compute_layer2_error", delta_layer2.extent(0), KOKKOS_LAMBDA(int i){
                        real layer2_output_val = layer2.output(i) ;
			real sum = 0.0 ;
			for (int j = 0 ; j < output_layer.weights.extent(1); j++){
				sum += delta_output(j) * output_layer.weights(j,i) ; 
			}
			delta_layer2(i) = sum * layer2_output_val * (1.0 - layer2_output_val) ; 
                });
                Kokkos::parallel_for("update_layer2_weights", layer2.weights.extent(0), KOKKOS_LAMBDA (int i){
                        for (int j = 0 ; j < layer2.weights.extent(1); j++){
                                layer2.weights(i,j) -= learning_rate * delta_layer2(i) * layer1.output(j) ;
                        }
                        layer2.biases(i) -= learning_rate * delta_layer2(i) ;
                });

                // layer 1
                View1D delta_layer1("delta_layer1", layer1.output.extent(0));
                Kokkos::parallel_for("compute_layer1_error", delta_layer1.extent(0), KOKKOS_LAMBDA(int i){
                        real layer1_output_val = layer1.output(i) ;
                        real sum = 0.0 ;
                        for (int j = 0 ; j < layer2.weights.extent(1); j++){
                                sum += delta_layer2(j) * layer2.weights(j,i) ;
                        }
                        delta_layer1(i) = sum * layer1_output_val * (1.0 - layer1_output_val) ;
                });
                Kokkos::parallel_for("update_layer1_weights", layer1.weights.extent(0), KOKKOS_LAMBDA (int i){
                        for (int j = 0 ; j < layer1.weights.extent(1); j++){
                                layer1.weights(i,j) -= learning_rate * delta_layer1(i) * input(j) ;
                        }
                        layer1.biases(i) -= learning_rate * delta_layer1(i) ;
                });
	}


	void show(){
		layer1.show() ; 
		layer2.show() ; 
		output_layer.show() ;
	}
};

int main(int argc, char **argv) {
	std::cout << "-- Neural Library --" << std::endl;
	Kokkos::initialize(argc, argv);
	{/////////////////////////
	int n1 = 2 ; // n entree
	int n2 = 4 ; 
	int n3 = 4 ; 
	int nsortie = 1 ; 
	DenseNeuralNetwork dnn(n1, n2, n3, nsortie) ; 
	real xor_inputs[4][2] = {
		{0.0,0.0},
		{0.0,1.0},
		{1.0,0.0},
		{1.0,1.0}};
	real xor_outputs[4] = {0.0,1.0,1.0,0.0};
	// entrainement
	dnn.show() ; 

	int epochs = 10000; 
	real learning_rate = 0.1 ; 
        std::cout << "-- TRAINING --" << std::endl;
	for (int epoch = 0 ; epoch < epochs ; epoch++){
		// sample
		for (int i = 0 ; i < 4 ; i++){
			View1D input("input", n1) ; 
			View1D target("target", nsortie);

			input(0) = xor_inputs[i][0] ; 
			input(1) = xor_inputs[i][1] ;
			target(0) = xor_outputs[i] ; 
			dnn.train(input, target, learning_rate) ; 
		}
	}
        std::cout << "-- TRAINING END --" << std::endl;

	dnn.show() ; 

	// inference
	for (int i = 0 ; i < 4 ; i++){
		View1D input("input", n1) ; 
		input(0) = xor_inputs[i][0] ; 
		input(1) = xor_inputs[i][1] ;
		auto output = dnn.forward(input) ;
		std::cout << "Input : " << xor_inputs[i][0] << " -- " << xor_inputs[i][1] << std::endl ; 
		std::cout << "--> Predicted : " << output(0) << " | Espered --> " << xor_outputs[i] << std::endl ; 
	}
	}/////////////////////////
	Kokkos::finalize();
	std::cout << "-- Brain completed ;-) --" << std::endl;
	return 0;
}


