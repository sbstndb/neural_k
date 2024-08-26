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

real sigmoid(real x) {
	return 1.0 / (1.0 + exp(-x));
}

real d_sigmoid(real x){
	// we can reuse the sigmoid output but not for now ! 
	real s = sigmoid(x) ; 
	return s * (1 - s) ; 
}


class Layer  {
public : 
	enum class LayerActivation {
		SIGMOID,
		TANH,
		RELU,
		LEAKY_RELU,
		LINEAR,
		SOFTPLUS
	};

	void activation(Layer::LayerActivation Activation, View1D input, View1D output){
		if(Activation == LayerActivation::SIGMOID){
	               Kokkos::parallel_for("sigmoid", layer_size, KOKKOS_LAMBDA(int i) {
	                       output(i) = sigmoid(input(i));
	                });
			
		}
		else if(Activation == LayerActivation::LINEAR){
                       Kokkos::parallel_for("linear", layer_size, KOKKOS_LAMBDA(int i) {
                               output(i) = input(i);
                        });
                }
		else if(Activation == LayerActivation::RELU){
                       Kokkos::parallel_for("relu", layer_size, KOKKOS_LAMBDA(int i) {
				if (input(i) > 0.0 ){ output(i) = input(i);}
				else {output(i) = 0.0 ; }
                        });
                }
                else if(Activation == LayerActivation::LEAKY_RELU){
                       Kokkos::parallel_for("relu", layer_size, KOKKOS_LAMBDA(int i) {
                                if (input(i) > 0.0 ){  output(i) = input(i);}
                                else { output(i) = 0.01 * input(i) ;} // TODO --> should add alpha value as argument 
                        });
                }		
                else if(Activation == LayerActivation::TANH){
                       Kokkos::parallel_for("tanh", layer_size, KOKKOS_LAMBDA(int i) {
                               output(i) = tanh(input(i));
                        });
                }
                else if(Activation == LayerActivation::SOFTPLUS){
                       Kokkos::parallel_for("add_sigmoid", layer_size, KOKKOS_LAMBDA(int i) {
                               output(i) = log(1.0 + exp(input(i)));
                        });
                }
	}

        void d_activation(Layer::LayerActivation Activation, View1D input, View1D output){
                if(Activation == LayerActivation::SIGMOID){
                       Kokkos::parallel_for("sigmoid", layer_size, KOKKOS_LAMBDA(int i) {
                               real s = sigmoid(input(i));
			       output(i) = s * (1.0 - s) ; 
                        });

                }
                else if(Activation == LayerActivation::LINEAR){
                       Kokkos::parallel_for("linear", layer_size, KOKKOS_LAMBDA(int i) {
                               output(i) = 1.0;
                        });
                }
                else if(Activation == LayerActivation::RELU){
                       Kokkos::parallel_for("relu", layer_size, KOKKOS_LAMBDA(int i) {
                                if (input(i) > 0.0 ){ output(i) = 1.0;}
                                else {output(i) = 0.0 ; }
                        });
                }
                else if(Activation == LayerActivation::LEAKY_RELU){
                       Kokkos::parallel_for("relu", layer_size, KOKKOS_LAMBDA(int i) {
                                if (input(i) > 0.0 ){  output(i) = 1.0;}
                                else { output(i) = 0.01  ;} // TODO --> should add alpha value as argument 
                        });
                }
                else if(Activation == LayerActivation::TANH){
                       Kokkos::parallel_for("tanh", layer_size, KOKKOS_LAMBDA(int i) {
                               output(i) = 1.0 - pow(tanh(input(i)), 2);
                        });
                }
                else if(Activation == LayerActivation::SOFTPLUS){
                       Kokkos::parallel_for("add_sigmoid", layer_size, KOKKOS_LAMBDA(int i) {
                               output(i) = 1.0 / (1.0 + exp(-input(i)));
                        });
                }
        }



	int input_size ; 
	int layer_size ; 
	LayerActivation activationType = LayerActivation::SIGMOID;

	View2D weights ; 
	View1D biases ; 
	View1D z ; // before activation
	View1D a ; // similar to activation in my case
	View1D tmp ; 


	Kokkos::Random_XorShift64_Pool<> rand_pool ;

	Layer(int input_size, int layer_size):
			rand_pool(std::time(0)), input_size(input_size), layer_size(layer_size) {
		weights = View2D("weights", layer_size, input_size) ; 
		biases = View1D("biases", layer_size) ; 
		z = View1D("z", layer_size) ; 
		a = View1D("a", layer_size) ;
		tmp = View1D("tmp", layer_size) ; 

// issue : repetitive generation among layers. For now : do not use fill random until we know how to random the seed
//		Kokkos::fill_random(weights, rand_pool, -0.8, 0.8);
//		Kokkos::fill_random(biases, rand_pool, -0.8, 0.8);
//              Kokkos::fill_random(z, rand_pool, -0.0, 0.0);// ??
// ///////////////////////
		std::random_device rd ; 
		std::mt19937 gen(rd()); 
		std::uniform_real_distribution<real> dist(-0.8, 0.8);

		for (int i = 0; i < layer_size; ++i) {
			biases(i) = dist(gen); 
			for (int j = 0; j < input_size; ++j) {
				weights(i, j) = dist(gen); 
			}
		}
	}

	void forward(const View1D& input){
		KokkosBlas::gemv("N", 1.0, weights, input, 0.0, z); 
                Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) {
                        z(i) += biases(i);
                });
		
		activation(activationType, z, a) ; 
	}

	void show(){
		std::cout << "Weights or layer : w(i,j) :" << std::endl; 
		for (int i = 0 ; i < layer_size; i++){
			std::cout << "   "  ; 
			for (int j = 0 ; j < input_size; j++){
				std::cout << weights(i,j) << " " ; 
			}
			std::cout << std::endl;
		}
                std::cout << "Biases or layer : b(i) :" << std::endl ;
		std::cout << "   " ; 
                for (int i = 0 ; i < layer_size; i++){
                        std::cout << biases(i) << " " ;
                }
		std::cout << std::endl ; 
	}

//	~Layer() ; 
};


class OutputLayer : public Layer {
public : 
	OutputLayer(int input_size, int output_size) : Layer(input_size, output_size) {}
	void forward(const View1D input) {
		Layer::forward(input) ;
	}
};


class DenseNeuralNetwork {
public:
	// not a generif way : do this until we generalize the neural network 
	Layer layer1 ; 
	Layer layer2 ; 
	OutputLayer output_layer ; 

	DenseNeuralNetwork(int n1, int n2, int n3, int nsortie):
		layer1(n1, n2), layer2(n2, n3), output_layer(n3, nsortie) {}

	View1D forward(const View1D& input){
		layer1.forward(input) ; 
		layer2.forward(layer1.a) ; 
		output_layer.forward(layer2.a) ; 
		return output_layer.a ; 
	}


	void train( const View1D input, const View1D target, real learning_rate){
		auto prediction = forward(input)  ;
		std::cout << " Prediction : " << prediction(0) << " with input " << input(0) << " " << input(1) << "and target " << target(0) <<  std::endl ; 
		

		View1D delta_output("delta_output", output_layer.layer_size);
		Kokkos::parallel_for("compute_output_error", prediction.extent(0), KOKKOS_LAMBDA(int i){
				delta_output(i) = (output_layer.a(i) - target(i)) * d_sigmoid(output_layer.z(i)) ; 
		});

		// layer 2
		View1D delta_layer2("delta_layer2", layer2.layer_size);
                Kokkos::parallel_for("compute_layer2_error", layer2.layer_size, KOKKOS_LAMBDA(int i){
                        real layer2_output_val = layer2.a(i) ;
			real sum = 0.0 ;
			for (int j = 0 ; j < output_layer.layer_size; j++){
				sum += delta_output(j) * output_layer.weights(j,i) ; // transpose ? 
			}
			delta_layer2(i) = sum * d_sigmoid(layer2.z(i)) ; 
                });

                // layer 1
                View1D delta_layer1("delta_layer1", layer1.layer_size);
                Kokkos::parallel_for("compute_layer1_error", delta_layer1.extent(0), KOKKOS_LAMBDA(int i){
                        real layer1_output_val = layer1.a(i) ;
                        real sum = 0.0 ;
                        for (int j = 0 ; j < layer2.weights.extent(0); j++){
                                sum += delta_layer2(j) * layer2.weights(j,i) ;// transpose ? 
                        }
                        delta_layer1(i) = sum * layer1_output_val * d_sigmoid(layer1.z(i));
		
                });

                Kokkos::parallel_for("update_output_weights", output_layer.weights.extent(0), KOKKOS_LAMBDA (int i){
                        for (int j = 0 ; j < output_layer.weights.extent(1); j++){
                                output_layer.weights(i,j) -= learning_rate * delta_output(i) * layer2.a(j) ;
                        }
                        output_layer.biases(i) -= learning_rate * delta_output(i) ;
                });
                Kokkos::parallel_for("update_layer2_weights", layer2.weights.extent(0), KOKKOS_LAMBDA (int i){
                        for (int j = 0 ; j < layer2.weights.extent(1); j++){
                                layer2.weights(i,j) -= learning_rate * delta_layer2(i) * layer1.a(j) ;
                        }
                        layer2.biases(i) -= learning_rate * delta_layer2(i) ;
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

	int epochs = 1000; 
	real learning_rate = 0.5 ; 
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


