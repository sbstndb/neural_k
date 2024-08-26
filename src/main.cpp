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
	LayerActivation activationType = LayerActivation::RELU;

	View2D weights ; 
	View1D biases ; 
	View1D z ; // before activation
	View1D a ; // similar to activation in my case
	View1D d ; // derivative of layer
	View1D tmp ; 


	Kokkos::Random_XorShift64_Pool<> rand_pool ;

	Layer(int input_size, int layer_size):
			rand_pool(std::time(0)), input_size(input_size), layer_size(layer_size) {
		weights = View2D("weights", layer_size, input_size) ; 
		biases = View1D("biases", layer_size) ; 
		z = View1D("z", layer_size) ; 
		a = View1D("a", layer_size) ;
		d = View1D("d_layer", layer_size) ; 
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

	void gradient(const auto& next_layer){
                // layer 2
		// for each neurons
		//
		d_activation(activationType, a, tmp) ; 
                Kokkos::parallel_for("compute_layer_error", layer_size, KOKKOS_LAMBDA(int i){
                        real sum = 0.0 ;
			// for each next neurons
                        for (int j = 0 ; j < next_layer.layer_size; j++){
                                sum += next_layer.d(j) * next_layer.weights(j,i) ; // transpose ? 
                        }
                        d(i) = sum * tmp(i) ;
                });	
	}

	void update(const auto& previous_layer, real learning_rate){
                Kokkos::parallel_for("update_layer_weights", layer_size, KOKKOS_LAMBDA (int i){
                        for (int j = 0 ; j < input_size; j++){
                                weights(i,j) -= learning_rate * d(i) * previous_layer.a(j) ;
                        }
                        biases(i) -= learning_rate * d(i) ;
                });

	}

	void backward(const auto& previous_layer, const auto& next_layer, real learning_rate){
		gradient(next_layer);
		update(previous_layer, learning_rate) ; 
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

	View1D target ; 

	OutputLayer(int input_size, int output_size) : Layer(input_size, output_size) {
		target = View1D("target", layer_size) ;
	}
	void forward(const View1D input) {
		Layer::forward(input) ;
	}

        void gradient(){
                // layer 2
                // for each neurons
                //
                d_activation(activationType, a, tmp) ;
                Kokkos::parallel_for("compute_layer_error", layer_size, KOKKOS_LAMBDA(int i){
                        real sum = 0.0 ;
                        // for each next neurons
                        d(i) = (a(i) - target(i)) * tmp(i) ;
                });
        }

        void update(const Layer& previous_layer, real learning_rate){
                Kokkos::parallel_for("update_layer_weights", layer_size, KOKKOS_LAMBDA (int i){
                        for (int j = 0 ; j < input_size; j++){
                                weights(i,j) -= learning_rate * d(i) * previous_layer.a(j) ;
                        }
                        biases(i) -= learning_rate * d(i) ;
                });
        }
       void backward(const Layer& previous_layer, real learning_rate){
                gradient();
                update(previous_layer, learning_rate) ;
        }


};


class InputLayer : public Layer {
public:
        InputLayer(int input_size, int output_size) : Layer(input_size, output_size) {
                a = View1D("a", input_size) ;
        }

	void input(const View1D& input) {
                Kokkos::parallel_for("input", a.extent(0), KOKKOS_LAMBDA(int i){
                        a(i) = input(i) ;
                });
	}
};


class DenseNeuralNetwork {
public:
	// not a generif way : do this until we generalize the neural network 
	InputLayer input_layer ; 
	Layer layer1 ; 
	Layer layer2 ; 
	OutputLayer output_layer ; 

	DenseNeuralNetwork(int n1, int n2, int n3, int nsortie):
		input_layer(n1, n1), layer1(n1, n2), layer2(n2, n3), output_layer(n3, nsortie) {}


        View1D forward(const View1D& input){
                input_layer.input(input) ;
                layer1.forward(input_layer.a) ;
                layer2.forward(layer1.a) ;
                output_layer.forward(layer2.a) ;
                return output_layer.a ;
        }


        View1D forward(){
		forward(input_layer.a) ; 
                return output_layer.a ;
        }
	


	void train( const View1D input, const View1D target, real learning_rate){
		auto prediction = forward(input)  ;
		// update layers ; 
		output_layer.backward(layer2, learning_rate); 
		// previous layer -- next layer -- learning_rate
		layer2.backward(layer1, output_layer, learning_rate) ; 
		layer1.backward(input_layer, layer2, learning_rate) ; 
	}


	void train(real learning_rate){
		train(input_layer.a, output_layer.target, learning_rate);
	}

	void input(real * input, int size){
		for (int i = 0 ; i < size ; i++){
			input_layer.a(i) = input[i] ; 
		}
	}

	void target(real*  target, int size){

		for (int i = 0 ; i < size ; i++){
			output_layer.target(i) = target[i] ; 
		}
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
	int n2 = 2 ; 
	int n3 = 2 ; 
	int nsortie = 1 ; 
	DenseNeuralNetwork dnn(n1, n2, n3, nsortie) ; 
	dnn.output_layer.activationType = Layer::LayerActivation::LINEAR; 
	real xor_inputs[4][2] = {
		{0.0,0.0},
		{0.0,1.0},
		{1.0,0.0},
		{1.0,1.0}};
	real xor_outputs[4][1] = {{0.0},{1.0},{1.0},{0.0}};

	int epochs = 1000; 
	real learning_rate = 0.1 ; 
        std::cout << "-- TRAINING --" << std::endl;
	for (int epoch = 0 ; epoch < epochs ; epoch++){
		for (int i = 0 ; i < 4 ; i++){
			dnn.input(xor_inputs[i], n1) ; 
			dnn.target(xor_outputs[i], nsortie) ; 
			dnn.train(learning_rate) ; 
		}
	}
        std::cout << "-- TRAINING END --" << std::endl;

	dnn.show() ; 

	// inference
	for (int i = 0 ; i < 4 ; i++){
		dnn.input(xor_inputs[i], n1) ; 
		auto output = dnn.forward() ;
		std::cout << "Input : " << dnn.input_layer.a(0) << " -- " << dnn.input_layer.a(1) << std::endl ; 
		std::cout << "--> Predicted : " << dnn.output_layer.a(0) << " | Espered --> " << xor_outputs[i][0] << std::endl ; 
	}
	}/////////////////////////
	Kokkos::finalize();
	std::cout << "-- Brain completed ;-) --" << std::endl;
	return 0;
}


