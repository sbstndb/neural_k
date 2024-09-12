#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <random>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <map>

#include <functional>


using real = float ;
using View1D = Kokkos::View<real *> ;
using View2D = Kokkos::View<real **>;
using View3D = Kokkos::View<real **> ;

using outerView1D = Kokkos::View<View1D*> ;
using outerView2D = Kokkos::View<View2D*> ;
using outerView3D = Kokkos::View<View3D*> ;
// issue --> we want this to be compatible with openmp too.


// classes
class Optimizer;
class Layer;
class Activation;
class Dataset;
class Network;
class BatchHandler;

class Optimizer {
public:
    // d, d_b, d_w and others
};


real act(real x) {
    return x ;
}


class Dataset{
public:
};

class BatchHandler{

};

class Activation {
    enum class LayerActivation {
        SIGMOID,
        TANH,
        RELU
    };
    LayerActivation activationType = LayerActivation::RELU ;
    View1D a ;

    std::function<real(real)> scalar_activation ;
    std::function<real(real)> scalar_d_activation ;



    void activation(const View1D& input , View1D& output) {
        Kokkos::parallel_for("activation", a.extent(0), KOKKOS_LAMBDA(int i) {
            output(i) = scalar_activation(input(i));
        });
    }

    void d_activation(const View1D& input , View1D& output) {
        Kokkos::parallel_for("d_activation", a.extent(0), KOKKOS_LAMBDA(int i) {
            output(i) = scalar_d_activation(input(i));
        });
    }

};


class RELU : public Activation {
    real scalar_activation(real input){
        if (input > 0.0){ return input;}
        else {return 0.0 ;}
    }
    real scalar_d_activation(real input) {
        if (input > 0.0){ return 1.0;}
        else {return 0.0 ;}
    }
};

class SIGMOID : public Activation {
    real scalar_activation(real input){
        return 1.0 / (1.0 + exp(-input));
    }
    real scalar_d_activation(real input) {
        real s = scalar_activation(input) ;
        return s * (1 - s) ;
    }
};

class TANH : public Activation {
    real scalar_activation(real input){
        return tanh(input);
    }
    real scalar_d_activation(real input) {
        return 1.0 - pow(tanh(input), 2);
    }
};


class Layer {
public:
    enum class LayerActivation {
        SIGMOID,
        TANH,
        RELU
    };
    int input_size ;
    int layer_size ;
    Activation activation ;
    Optimizer optimizer ;

    View2D weights ;
    View1D biases ;
    View1D z ;
    View1D tmp ; // in case of
    Kokkos::Random_XorShift64_Pool<> rand_pool ;

    Layer(int _input_size, int _layer_size):
        input_size(_input_size), layer_size(_layer_size){
        weights = View2D("weights", layer_size, input_size) ;
        biases = View1D("biases", layer_size) ;
        z = View1D("z", layer_size) ;
        tmp = View1D("tmp", layer_size) ;
        //activation(layer_size) ; --> TODO
    }

};

class InputLayer : public Layer {
public:
    InputLayer(int _layer_size) :
        Layer(0, _layer_size){
    }
};

class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size):
        Layer(_input_size, _layer_size){
    }
};

class Network{
public:
    // series of layers with arbitrary size
    using outerViewLayer = Kokkos::View<Layer*> ;
    InputLayer input_layer ;
    outerViewLayer hidden_layers ;
    OutputLayer output_layer;
    Dataset dataset ;
    BatchHandler batch_handler;

    std::map<int, int> size ;
    Network(std::map<int, int> _size):
        size(_size),
        input_layer(size[0]),
        output_layer(_size[_size.size()-1], _size[_size.size()])
        {
        int hidden_size = _size.size() - 2 ;
        // allocation of outer structure
        outerViewLayer hidden_layers(Kokkos::view_alloc(std::string("hidden_layers"),
                                                        Kokkos::WithoutInitializing),
                                                        hidden_size);
        for (int i = 0 ; i < hidden_size; i++){
            const std::string label = std::string("hidden ") + std::to_string(i+1);
            hidden_layers[i] = Layer(size[i], size[i+1]) ; // and allocate from layer
            //new(&hidden_layers[i]) Layer(Kokkos::view_alloc("l", Kokkos::WithoutInitializing), size[i+1]);
        }
    }
};


int main(int argc, char ** argv){


    Kokkos::initialize(argc, argv) ;
    {///////////////////////////////


        std::map<int, int> size ;
        size[0] = 2 ;
        size[1] = 4 ;
        size[2] = 4 ;
        size[3] = 1 ;
        Network network(size) ;
        /////////////////////////////////////////////
        // try nested loop
        int numOuter = 4;
        int numInner = 5;
        outerView1D outer(Kokkos::view_alloc(std::string("Outer"), Kokkos::WithoutInitializing), numOuter);
        for (int k = 0; k < numOuter; ++k) {
            const std::string label = std::string("Inner ") + std::to_string(k);
            new(&outer[k]) View1D(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), numInner);
        }

        Kokkos::RangePolicy<int> range(0, numOuter);
        Kokkos::parallel_for("my kernel label", range,
                             KOKKOS_LAMBDA(
        const int i) {
            for (int j = 0; j < numInner; ++j) {
                outer[i][j] = 10.0 * double(i) + double(j);
            }
        });
        Kokkos::fence();
        for (int i = 0; i < numOuter; i++) {
            for (int j = 0; j < numInner; j++) {
                std::cout << outer[i][j] << " ";
            }
            std::cout << std::endl;
        }
        Kokkos::fence();
        // Destroy inner Views, again on host, outside of a parallel region.
        for (int k = 0; k < numOuter; ++k) {
            outer[k].~View1D();
        }
        // You' re better off disposing of outer immediately.
        outer = outerView1D();

    }
    Kokkos::finalize();
}
