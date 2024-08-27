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


// TODO : 
// better activation & d_activation functions
// use kokkos kernels more often when needed
// provide better API --> give dnn the training data once. Take care : consume more memory
// use nested view with kokkos
// implement different gradient algorithms
// -- batch gradient descent
// -- stochastic gradient descent
// -- mini batch gradient descent
// -- batch stochastic gradient descent
// -- gradient descent avec momentum
// -- adaptive method like Adam, RMSprop, Adagrad
// I suggest to implement a new class named optimizer with subclasses. Hence, the neural network only contains an optimizer and not a lot of d_x_blabla arrays. DNN should only contains weights, input, output and the optimizer structure !


class Layer;

class Optimizer {
public:

	View1D d_b ; 
	View1D biases ; 
	View2D weights ;
	real learning_rate ; 
	int opt_size, input_size ; 


//        Optimizer(int input_size, int layer_size):
//                        input_size(input_size), opt_size(opt_size) {}
//	Optimizer(View2D& weights, View1D& biases, int input_size, int layer_size):
//			weights(weights), biases(biases), input_size(input_size), opt_size(opt_size)  {}
//	Optimizer(View2D& weights, View1D& biases):
//                       weights(weights), biases(biases), input_size(weights.extent(1)), opt_size(weights.extent(0))  {}
		


	///////////////////////
	enum class OptimizerType{
		// compute d_ for all input training data hence update weigths
		BatchGradientDescent,
		// compute d_ for only one input hence update
		StochasticGradientDescent,
		// todo
		MiniBatchGradientDescent,
		MomentumGradientDescent,
		ADAM
	};
	///////////////////////


	void gradient(const Layer& previous_layer, const Layer& next_layer){
	}

	void update_biases(const View1D& biases, real _learning_rate){
                Kokkos::parallel_for("update_layer_biases", opt_size, KOKKOS_LAMBDA (int i){
                        biases(i) -= _learning_rate * d_b(i) ;
                });
	}
        void update_biases(const View1D& biases){
		update_biases(biases, learning_rate);
	}


        void update_weights(const View2D& weights, real _learning_rate){
                update_weights(weights, _learning_rate);
                // too specify to be implemented here
        }
	void update_weights(const View2D& weights){
		update_weights(weights, learning_rate);
		// too specify to be implemented here
	}

	void update(const View1D& biases, const View2D& weights){
		update_biases(biases);
		update_weights(weights);
	}
	

};

// dont know if i should use enum or subclasses...
//
//

class BatchGradientDescent : public Optimizer {
public:

	View1D biases ; 
	View2D weights ;

	View1D d, d_b_sum, tmp;
	View2D d_w_sum ; 

	int opt_size, input_size; 
	real learning_rate;

        BatchGradientDescent(int input_size, int opt_size):
                        input_size(input_size), opt_size(opt_size)  {
                d = View1D("d_layer", opt_size) ;
                d_w_sum = View2D("d_w_sum_layer", opt_size, input_size) ;
                d_b_sum = View1D("d_b_sum_layer", opt_size) ;
                tmp = View1D("tmp", opt_size) ;
        }

//        BatchGradientDescent(View2D& weigths, View1D& biases, int input_size, int opt_size):
//                        weights(weights), biases(biases), input_size(input_size), opt_size(opt_size)  {
//                d = View1D("d_layer", opt_size) ;
//                d_w_sum = View2D("d_w_sum_layer", opt_size, input_size) ;
//                d_b_sum = View1D("d_b_sum_layer", opt_size) ;
//                tmp = View1D("tmp", opt_size) ;
//	}

	void zero(){
		Kokkos::deep_copy(d, 0.0);
		Kokkos::deep_copy(d_b_sum, 0.0); 
		Kokkos::deep_copy(d_w_sum, 0.0);
	}

        void gradient(const auto& previous_layer, const auto& layer, const auto& next_layer){
		layer.d_activation(layer.activationType, layer.z, layer.tmp) ;
                Kokkos::parallel_for("compute_layer_error", layer.layer_size, KOKKOS_LAMBDA(int i){
                        real sum = 0.0 ;
                        // for each next neurons
                        for (int j = 0 ; j < next_layer.layer_size; j++){
                                sum += next_layer.d(j) * next_layer.weights(j,i) ; // transpose ? 
                        }
                        d(i) = sum * layer.tmp(i) ;
                        d_b_sum(i) += d(i) ;
                        for (int j = 0 ; j < layer.input_size ; j++){
                                d_w_sum(i,j) += d(i) * previous_layer.a(j);
                        }
                });
        }

	//void update_biases(){}; // already defined in base class
				//
        void update_weigths(const auto& weigths, real learning_rate){
                Kokkos::parallel_for("update_layer_weights", opt_size, KOKKOS_LAMBDA (int i){
                        for (int j = 0 ; j < input_size; j++){
                                weights(i,j) -= learning_rate * d_w_sum(i,j)  ;
                        }
                });

        }
	void update(const auto& layer, real learning_rate){
		update_weights(layer.weigths, learning_rate);
		update_biases(layer.biases, learning_rate);
	}
};

class BatchGradientDescentOutput : public BatchGradientDescent{

        using BatchGradientDescent::BatchGradientDescent;


	// specific definition for gradient
        void gradient(const auto& previous_layer, const auto& layer){
                layer.d_activation(layer.activationType, layer.z, layer.tmp) ;
                Kokkos::parallel_for("compute_layer_error", layer.layer_size, KOKKOS_LAMBDA(int i){
                        real sum = 0.0 ;
                        // for each next neurons
                        d(i) = (layer.a(i) - layer.target(i)) * layer.tmp(i) ;
                        d_b_sum(i) += d(i) ;
                        for (int j = 0 ; j < input_size ; j++){
                                d_w_sum(i,j) += d(i) * previous_layer.a(j);
                        }
                });
        }
};

class StochasticGradientDescent : public Optimizer {
public:
};

class MiniBatchGradientDescent : public Optimizer {
public:
};




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
                               real s = sigmoid(input(i)); // should be = to a(i)
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
	LayerActivation activationType = LayerActivation::TANH;
	BatchGradientDescent optimizer ; 


	View2D weights ; 
	View1D biases ; 
	View1D z ; // before activation
	View1D a ; // similar to activation in my case
	View1D d ; // derivative of layer
        View2D d_w_sum ; // sum of d for all samples ! 
	View1D d_b_sum ; // sum of d for all samples ! 
	View1D tmp ; // general purpose temporary vector


	Kokkos::Random_XorShift64_Pool<> rand_pool ;

	Layer(int input_size, int layer_size):
			rand_pool(std::time(0)), input_size(input_size), layer_size(layer_size), optimizer(input_size, layer_size) {
		weights = View2D("weights", layer_size, input_size) ; 
		biases = View1D("biases", layer_size) ; 
		z = View1D("z", layer_size) ; 
		a = View1D("a", layer_size) ;
		d = View1D("d_layer", layer_size) ; 
                d_w_sum = View2D("d_w_sum_layer", layer_size, input_size) ;		
                d_b_sum = View1D("d_b_sum_layer", layer_size) ;		
		tmp = View1D("tmp", layer_size) ; 

// issue : repetitive generation among layers. For now : do not use fill random until we know how to random the seed
//		Kokkos::fill_random(weights, rand_pool, -0.8, 0.8);
//		Kokkos::fill_random(biases, rand_pool, -0.8, 0.8);
//              Kokkos::fill_random(z, rand_pool, -0.0, 0.0);// ??
// ///////////////////////
		std::random_device rd ; 
		std::mt19937 gen(rd()); 
		std::uniform_real_distribution<real> dist(-0.1, 0.1);

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

	void init_d_sum(){
                Kokkos::parallel_for("init_d_sum", layer_size, KOKKOS_LAMBDA(int i) {
			for (int j = 0 ; j < input_size ; j++){
	                        d_w_sum(i,j) = 0.0;
			}
			d_b_sum(i) = 0.0 ; 
                });

	}

	void gradient(const auto& previous_layer, const auto& next_layer){
                // layer 2
		// for each neurons
		//
		d_activation(activationType, z, tmp) ; 
                Kokkos::parallel_for("compute_layer_error", layer_size, KOKKOS_LAMBDA(int i){
                        real sum = 0.0 ;
			// for each next neurons
                        for (int j = 0 ; j < next_layer.layer_size; j++){
                                sum += next_layer.d(j) * next_layer.weights(j,i) ; // transpose ? 
                        }
                        d(i) = sum * tmp(i) ;
			d_b_sum(i) += d(i) ; 
			for (int j = 0 ; j < input_size ; j++){
				d_w_sum(i,j) += d(i) * previous_layer.a(j);
			}
                });	
	}

	void update(real learning_rate){
                Kokkos::parallel_for("update_layer_weights", layer_size, KOKKOS_LAMBDA (int i){
                        for (int j = 0 ; j < input_size; j++){
                                weights(i,j) -= learning_rate * d_w_sum(i,j)  ;
                        }
                        biases(i) -= learning_rate * d_b_sum(i) ;
                });

	}

	void backward(const auto& previous_layer, const auto& next_layer, real learning_rate){
		gradient(previous_layer, next_layer);
		update(learning_rate) ; 
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
        BatchGradientDescentOutput optimizer ;



	OutputLayer(int input_size, int output_size) : Layer(input_size, output_size), optimizer(input_size, output_size) {
		target = View1D("target", layer_size) ;
	}
	void forward(const View1D input) {
		Layer::forward(input) ;
	}

        void gradient(const Layer& previous_layer){
                d_activation(activationType, z, tmp) ;
                Kokkos::parallel_for("compute_layer_error", layer_size, KOKKOS_LAMBDA(int i){
                        real sum = 0.0 ;
                        // for each next neurons
                        d(i) = (a(i) - target(i)) * tmp(i) ;
                        d_b_sum(i) += d(i) ;
                        for (int j = 0 ; j < input_size ; j++){
                                d_w_sum(i,j) += d(i) * previous_layer.a(j);
                        }


                });
        }

        void update(real learning_rate){
                Kokkos::parallel_for("update_layer_weights", layer_size, KOKKOS_LAMBDA (int i){
                        for (int j = 0 ; j < input_size; j++){
                                weights(i,j) -= learning_rate * d_w_sum(i,j) ;
                        }
                        biases(i) -= learning_rate * d_b_sum(i) ;
                });
        }
       void backward(const Layer& previous_layer, real learning_rate){
                gradient(previous_layer);
                update(learning_rate) ;
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
	


	void init_d_sum(){
		layer1.init_d_sum() ; 
		layer2.init_d_sum() ; 
		output_layer.init_d_sum() ; 
	}

	void train( const View1D input, const View1D target, real learning_rate){
		auto prediction = forward(input)  ;
		// update layers ; 
		//output_layer.backward(layer2, learning_rate); 
		// previous layer -- next layer -- learning_rate
		//layer2.backward(layer1, output_layer, learning_rate) ; 
		//layer1.backward(input_layer, layer2, learning_rate) ; 
		//
		output_layer.gradient(layer2); 
		layer2.gradient(layer1, output_layer); 
		layer1.gradient(input_layer, layer2); 
	}


	void train(real learning_rate){
		train(input_layer.a, output_layer.target, learning_rate);
	}

	void update(real learning_rate){
                output_layer.update(learning_rate) ;
                layer2.update(learning_rate);
                layer1.update(learning_rate);		
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

	real cost(){
		real sum = 0.0 ; 
		for (int i = 0 ; i < output_layer.layer_size ; i++){
			real diff = output_layer.target(i) - output_layer.a(i) ; 
			sum += diff * diff ; 
		}
		return 0.5 * sum ;
	}


	void show(){
		layer1.show() ; 
		layer2.show() ; 
		output_layer.show() ;
	}
};


void xor_train(void){
	int n1 = 2 ; // n entree
        int n2 = 3 ;
        int n3 = 2 ;
        int nsortie = 1 ;
        DenseNeuralNetwork dnn(n1, n2, n3, nsortie) ;
        dnn.output_layer.activationType = Layer::LayerActivation::TANH;
        real xor_inputs[5][2] = {
                {0.0,0.0},
                {1.0,1.0},
                {1.0,0.0},
                {0.0,1.0},
		{0.0,0.0}};
        real xor_outputs[5][1] = {{-0.0},{-0.0},{1.0},{1.0},{-0.0}};

        int epochs = 4000;
        real learning_rate = 0.1 ;
        std::cout << "-- TRAINING --" << std::endl;
        for (int epoch = 0 ; epoch < epochs ; epoch++){
		real total_cost = 0.0 ; 
		dnn.init_d_sum() ; 
                for (int i = 0 ; i < 5 ; i++){
                        dnn.input(xor_inputs[i], n1) ;
                        dnn.target(xor_outputs[i], nsortie) ;
                        dnn.train(learning_rate) ;
			total_cost += dnn.cost()/4 ;  
                }
                dnn.update(learning_rate);// update with average 
		std::cout << "Cost : " << total_cost << std::endl ; 
        }
        std::cout << "-- TRAINING END --" << std::endl;
        dnn.show() ;
        for (int i = 0 ; i < 5 ; i++){
                dnn.input(xor_inputs[i], n1) ;
                auto output = dnn.forward() ;
                std::cout << "Input : " << dnn.input_layer.a(0) << " -- " << dnn.input_layer.a(1) << std::endl ;
                std::cout << "--> Predicted : " << dnn.output_layer.a(0) << " | Espered --> " << xor_outputs[i][0] << std::endl ;
        }
}
void p_train(void){
        int n1 = 2 ; // n entree
        int n2 = 3 ;
        int n3 = 2 ;
        int nsortie = 1 ;
        DenseNeuralNetwork dnn(n1, n2, n3, nsortie) ;
        dnn.output_layer.activationType = Layer::LayerActivation::RELU;
        real xor_inputs[10][2] = {
                {1.2,0.7},
                {-0.3,-0.5},
                {3.0,0.1},
                {-0.1,-1.0},
                {-0.0,1.1},
                {2.1,-1.3},
                {3.1,-1.8},
                {1.1,-0.1},
                {1.5,-2.2},
                {-4.0,-1.0},	
	};
        real xor_outputs[10][1] = {{1.0},{-1.0},{1.0},{-1.0}, {-1.0}, {1.0}, {-1.0}, {1.0}, {-1.0}, {-1.0}};
        int epochs = 40100;
        real learning_rate = 0.10;
        std::cout << "-- TRAINING --" << std::endl;
        for (int epoch = 0 ; epoch < epochs ; epoch++){
                real total_cost = 0.0 ;
		dnn.init_d_sum(); 
                for (int i = 0 ; i < 10 ; i++){
                        dnn.input(xor_inputs[i], n1) ;
                        dnn.target(xor_outputs[i], nsortie) ;
                        dnn.train(learning_rate) ;
                        total_cost += dnn.cost()/10 ;
                }
		dnn.update(learning_rate);
                std::cout << "Cost : " << total_cost << std::endl ;
        }
        std::cout << "-- TRAINING END --" << std::endl;

        dnn.show() ;

        // inference
        for (int i = 0 ; i < 10 ; i++){
                dnn.input(xor_inputs[i], n1) ;
                auto output = dnn.forward() ;
                std::cout << "Input : " << dnn.input_layer.a(0) << " -- " << dnn.input_layer.a(1) << std::endl ;
                std::cout << "--> Predicted : " << dnn.output_layer.a(0) << " | Espered --> " << xor_outputs[i][0] << std::endl ;
        }
}



int main(int argc, char **argv) {
        std::cout << "-- Neural Library --" << std::endl;
        Kokkos::initialize(argc, argv);
        {/////////////////////////
		xor_train() ; 
        }/////////////////////////
        Kokkos::finalize();

	std::cout << "-- Brain completed ;-) --" << std::endl;
	return 0;
}


