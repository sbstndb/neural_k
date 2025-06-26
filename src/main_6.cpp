#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <random>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <numeric> // Pour std::iota (potentiellement)
#include <iomanip> // Pour std::setprecision
#include <typeinfo> // Pour typeid dans Network::show

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_StdAlgorithms.hpp> // Pour parallel_reduce

// --- Typedefs ---
using real = float;
using View1D = Kokkos::View<real*>;
using View2D = Kokkos::View<real**>;
// using View3D = Kokkos::View<real***>; // Pas utilisé ici

// --- Forward Declarations ---
class Optimizer; // Base class
class SGD;       // Concrete optimizer
class Adam;      // Concrete optimizer (to be added)
// class RMSprop;   // Concrete optimizer (future)
class Layer;
class Activation;
class Dataset;
class Network;
class BatchHandler;

// --- Classe Optimizer ---
class Optimizer {
public:
    real learning_rate;

    Optimizer(real lr) : learning_rate(lr) {} // Removed default value to force specification
    virtual ~Optimizer() = default;

    // Method to update layer parameters based on accumulated gradients
    virtual void update(View2D weights, View1D biases,
                        const View2D& accumulated_d_weights, const View1D& accumulated_d_biases,
                        int batch_size) = 0;

    // Allow changing learning rate
    void set_learning_rate(real lr) { learning_rate = lr; }
    real get_learning_rate() const { return learning_rate; }

    // Optional: For displaying info
    virtual std::string get_info() const {
        return "Optimizer(LR=" + std::to_string(learning_rate) + ")";
    }
};

// --- Concrete Optimizer: SGD ---
class SGD : public Optimizer {
public:
    SGD(real lr = 0.1) : Optimizer(lr) {}

    void update(View2D weights, View1D biases,
                const View2D& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override {

        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for SGD update.");
        }
        if (weights.extent(0) == 0 && biases.extent(0) == 0) {
             // Check if both are empty, might happen for input layer placeholder update attempt
             return;
        }
        if (biases.extent(0) == 0 || weights.extent(0) == 0) {
             // Should not happen for standard layers, but safety check
              throw std::runtime_error("SGD update called on layer with missing weights or biases.");
        }


        const real scale = learning_rate / static_cast<real>(batch_size);
        const int layer_size = biases.extent(0);
        const int input_size = weights.extent(1); // Assuming weights is layer_size x input_size

        // Update weights: weights -= learning_rate/batch_size * accumulated_d_weights
        Kokkos::parallel_for("sgd_update_weights",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                // Check bounds (though MDRangePolicy should handle it)
                if (i < weights.extent(0) && j < weights.extent(1)) {
                     weights(i, j) -= scale * accumulated_d_weights(i, j);
                }
        });

        // Update biases: biases -= learning_rate/batch_size * accumulated_d_biases
        Kokkos::parallel_for("sgd_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
             if (i < biases.extent(0)) { // Check bounds
                biases(i) -= scale * accumulated_d_biases(i);
             }
        });
    }

    std::string get_info() const override {
        return "SGD(LR=" + std::to_string(learning_rate) + ")";
    }
};


// --- Concrete Optimizer: Adam ---
class Adam : public Optimizer {
public:
    real beta1;
    real beta2;
    real epsilon;

private:
    // Structure to hold state (m, v, t) for a parameter tensor (weights or biases)
    struct ParameterState {
        View1D m_1d; // For biases
        View1D v_1d; // For biases
        View2D m_2d; // For weights
        View2D v_2d; // For weights
        long long t = 0; // Timestep for this parameter (use long long to avoid overflow)

        // Constructors to initialize appropriately sized views
        ParameterState(int size) : t(0) { // For biases
            m_1d = View1D("adam_m1d", size);
            v_1d = View1D("adam_v1d", size);
            Kokkos::deep_copy(m_1d, 0.0);
            Kokkos::deep_copy(v_1d, 0.0);
        }
        ParameterState(int rows, int cols) : t(0) { // For weights
            m_2d = View2D("adam_m2d", rows, cols);
            v_2d = View2D("adam_v2d", rows, cols);
            Kokkos::deep_copy(m_2d, 0.0);
            Kokkos::deep_copy(v_2d, 0.0);
        }
         // Default constructor needed for map's default behavior (though we use try_emplace)
         ParameterState() = default;
         // Move constructor/assignment for efficiency in map operations if needed
         ParameterState(ParameterState&&) = default;
         ParameterState& operator=(ParameterState&&) = default;
         // Delete copy constructor/assignment as Views are not trivially copyable
         ParameterState(const ParameterState&) = delete;
         ParameterState& operator=(const ParameterState&) = delete;
    };

    // Maps to store state: Use data pointer as key.
    // Assumes the data pointers for layer weights/biases remain stable.
    std::map<real*, ParameterState> state_map_1d; // For biases
    std::map<real*, ParameterState> state_map_2d; // For weights

public:
    Adam(real lr = 0.001, real b1 = 0.9, real b2 = 0.999, real eps = 1e-8)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps) {
        if (lr <= 0 || b1 < 0 || b1 >= 1 || b2 < 0 || b2 >= 1 || eps <= 0) {
            throw std::runtime_error("Invalid Adam hyperparameters.");
        }
    }

    void update(View2D weights, View1D biases,
                const View2D& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override {

        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for Adam update.");
        }
        if (weights.extent(0) == 0 && biases.extent(0) == 0) {
            return; // Nothing to update
        }
        if (biases.extent(0) == 0 || weights.extent(0) == 0) {
              throw std::runtime_error("Adam update called on layer with missing weights or biases.");
        }

        const int layer_size = biases.extent(0);
        const int input_size = weights.extent(1);
        const real scale = 1.0 / static_cast<real>(batch_size); // Scale for averaging gradients

        // --- Update Weights ---
        {
            // Get or create state for weights
            auto it_w = state_map_2d.find(weights.data());
            if (it_w == state_map_2d.end()) {
                // Create state if it doesn't exist
                auto result = state_map_2d.try_emplace(weights.data(), layer_size, input_size);
                 if (!result.second) {
                     throw std::runtime_error("Failed to insert Adam state for weights.");
                 }
                 it_w = result.first; // Get iterator to the newly inserted element
                 //std::cout << "Adam: Initialized state for weights at " << weights.data() << std::endl;
            }
            ParameterState& state_w = it_w->second;
            state_w.t++; // Increment timestep

            // Precompute bias correction terms (can be done on host)
            // Use double for potentially higher precision in power calculation
            const double beta1_pow_t = std::pow(static_cast<double>(beta1), static_cast<double>(state_w.t));
            const double beta2_pow_t = std::pow(static_cast<double>(beta2), static_cast<double>(state_w.t));
            const real bias_correction1 = 1.0f / (1.0f - static_cast<real>(beta1_pow_t));
            const real bias_correction2 = 1.0f / (1.0f - static_cast<real>(beta2_pow_t));


            // Capture necessary variables for the kernel
            real lr = learning_rate;
            real b1 = beta1;
            real b2 = beta2;
            real eps = epsilon;
            View2D m = state_w.m_2d; // Get references to the state views
            View2D v = state_w.v_2d;

            Kokkos::parallel_for("adam_update_weights",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
                KOKKOS_LAMBDA (const int i, const int j) {
                    if (i < weights.extent(0) && j < weights.extent(1)) { // Bounds check
                        // 1. Calculate average gradient for this sample
                        real grad = scale * accumulated_d_weights(i, j);

                        // 2. Update biased moments
                        m(i, j) = b1 * m(i, j) + (1.0f - b1) * grad;
                        v(i, j) = b2 * v(i, j) + (1.0f - b2) * grad * grad;

                        // 3. Compute bias-corrected moments
                        real m_hat = m(i, j) * bias_correction1;
                        real v_hat = v(i, j) * bias_correction2;

                        // 4. Update weights
                        weights(i, j) -= lr * m_hat / (Kokkos::sqrt(v_hat) + eps);
                    }
            });
        } // End scope for weights update

        // --- Update Biases ---
        {
             // Get or create state for biases
             auto it_b = state_map_1d.find(biases.data());
             if (it_b == state_map_1d.end()) {
                 auto result = state_map_1d.try_emplace(biases.data(), layer_size);
                 if (!result.second) {
                     throw std::runtime_error("Failed to insert Adam state for biases.");
                 }
                 it_b = result.first;
                  //std::cout << "Adam: Initialized state for biases at " << biases.data() << std::endl;
             }
             ParameterState& state_b = it_b->second;
             state_b.t++; // Increment timestep

            // Precompute bias correction terms (can be done on host)
            // Use double for potentially higher precision in power calculation
            const double beta1_pow_t = std::pow(static_cast<double>(beta1), static_cast<double>(state_b.t));
            const double beta2_pow_t = std::pow(static_cast<double>(beta2), static_cast<double>(state_b.t));
            const real bias_correction1 = 1.0f / (1.0f - static_cast<real>(beta1_pow_t));
            const real bias_correction2 = 1.0f / (1.0f - static_cast<real>(beta2_pow_t));


            // Capture necessary variables for the kernel
            real lr = learning_rate;
            real b1 = beta1;
            real b2 = beta2;
            real eps = epsilon;
            View1D m = state_b.m_1d;
            View1D v = state_b.v_1d;

            Kokkos::parallel_for("adam_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
                 if (i < biases.extent(0)) { // Bounds check
                    // 1. Calculate average gradient
                    real grad = scale * accumulated_d_biases(i);

                    // 2. Update biased moments
                    m(i) = b1 * m(i) + (1.0f - b1) * grad;
                    v(i) = b2 * v(i) + (1.0f - b2) * grad * grad;

                    // 3. Compute bias-corrected moments
                    real m_hat = m(i) * bias_correction1;
                    real v_hat = v(i) * bias_correction2;

                    // 4. Update biases
                    biases(i) -= lr * m_hat / (Kokkos::sqrt(v_hat) + eps);
                 }
            });
        } // End scope for biases update
    } // End update method

     std::string get_info() const override {
         return "Adam(LR=" + std::to_string(learning_rate) +
                ", beta1=" + std::to_string(beta1) +
                ", beta2=" + std::to_string(beta2) +
                ", epsilon=" + std::to_string(epsilon) + ")";
     }

     // Need virtual destructor because base class has one
     virtual ~Adam() override = default;

}; // End class Adam


// --- Classes Activation (inchangées) ---
class Activation {
public:
    virtual ~Activation() = default;
    virtual void apply(const View1D& input, View1D& output) const = 0;
    virtual void apply_derivative(const View1D& input_z, View1D& output_deriv) const = 0;
};

class LinearActivation : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override {
        // output(i) = input(i) -> Just copy
        Kokkos::deep_copy(output, input);
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        // f'(z) = 1
        const int size = input_z.extent(0);
         Kokkos::parallel_for("linear_deriv", size, KOKKOS_LAMBDA(const int i) {
             if (i < output_deriv.extent(0)) { // Bounds check
                output_deriv(i) = 1.0f;
             }
        });
    }
};

class RELU : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent(0);
        Kokkos::parallel_for("relu_apply", size, KOKKOS_LAMBDA(const int i) {
             if (i < output.extent(0)) // Bounds check
                output(i) = (input(i) > 0.0f) ? input(i) : 0.0f;
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent(0);
        Kokkos::parallel_for("relu_deriv", size, KOKKOS_LAMBDA(const int i) {
             if (i < output_deriv.extent(0)) // Bounds check
                output_deriv(i) = (input_z(i) > 0.0f) ? 1.0f : 0.0f;
        });
    }
};

class SIGMOID : public Activation {
public:
    KOKKOS_INLINE_FUNCTION real scalar_sigmoid(real x) const {
        x = Kokkos::max(-30.0f, Kokkos::min(30.0f, x)); // Clamping for stability
        return 1.0f / (1.0f + Kokkos::exp(-x));
    }
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent(0);
        Kokkos::parallel_for("sigmoid_apply", size, KOKKOS_LAMBDA(const int i) {
             if (i < output.extent(0)) // Bounds check
                output(i) = scalar_sigmoid(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent(0);
        Kokkos::parallel_for("sigmoid_deriv", size, KOKKOS_LAMBDA(const int i) {
            if (i < output_deriv.extent(0)) { // Bounds check
                real s = scalar_sigmoid(input_z(i));
                output_deriv(i) = s * (1.0f - s);
            }
        });
    }
};

class TANH : public Activation {
public:
     KOKKOS_INLINE_FUNCTION real scalar_tanh(real x) const {
        // Kokkos::tanh might handle large values, but clamping can be added if needed
        // x = Kokkos::max(-15.0f, Kokkos::min(15.0f, x));
        return Kokkos::tanh(x);
    }
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent(0);
        Kokkos::parallel_for("tanh_apply", size, KOKKOS_LAMBDA(const int i) {
             if (i < output.extent(0)) // Bounds check
                output(i) = scalar_tanh(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent(0);
        Kokkos::parallel_for("tanh_deriv", size, KOKKOS_LAMBDA(const int i) {
             if (i < output_deriv.extent(0)) { // Bounds check
                real t = scalar_tanh(input_z(i));
                output_deriv(i) = 1.0f - t * t;
             }
        });
    }
};

// Helper pour créer des activations par type (inchangé)
std::unique_ptr<Activation> create_activation(const std::string& type) {
    if (type == "relu") return std::make_unique<RELU>();
    if (type == "sigmoid") return std::make_unique<SIGMOID>();
    if (type == "tanh") return std::make_unique<TANH>();
    if (type == "linear") return std::make_unique<LinearActivation>();
    throw std::runtime_error("Unknown activation type: " + type);
}


// --- Classe Layer (inchangée par rapport à la version précédente avec Optimizer retiré) ---
class Layer {
public:
    int input_size;
    int layer_size;
    std::unique_ptr<Activation> activation;

    // Vues Kokkos pour les paramètres et états
    View2D weights;
    View1D biases;
    View1D z;     // Sortie avant activation
    View1D a;     // Sortie après activation (activation(z))

    // Vues Kokkos pour la rétropropagation (gradients)
    View1D delta;            // Erreur rétropropagée (δ) pour cette couche
    View1D d_biases;         // Gradient instantané des biais (∂Cost/∂b = δ) pour UN sample
    View2D d_weights;        // Gradient instantané des poids (∂Cost/∂W = δ * a_prev^T) pour UN sample
    View1D d_biases_sum;     // Somme des gradients des biais sur le batch
    View2D d_weights_sum;    // Somme des gradients des poids sur le batch

    View1D tmp_deriv;        // Stockage temporaire pour la dérivée de l'activation f'(z)


    // Constructeur pour couche cachée/sortie
    Layer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        input_size(_input_size),
        layer_size(_layer_size),
        activation(std::move(act_func))
    {
        if (input_size <= 0 || layer_size <= 0) {
             throw std::runtime_error("Layer input/output sizes must be positive.");
        }
        weights = View2D("weights", layer_size, input_size);
        biases = View1D("biases", layer_size);
        z = View1D("z", layer_size);
        a = View1D("a", layer_size);
        delta = View1D("delta", layer_size);
        d_biases = View1D("d_biases", layer_size); // Instantané (pour 1 sample)
        d_weights = View2D("d_weights", layer_size, input_size); // Instantané (pour 1 sample)
        d_biases_sum = View1D("d_biases_sum", layer_size); // Accumulé sur batch
        d_weights_sum = View2D("d_weights_sum", layer_size, input_size); // Accumulé sur batch
        tmp_deriv = View1D("tmp_deriv", layer_size);

        // Initialisation Xavier/Glorot (uniforme)
        Kokkos::Random_XorShift64_Pool<> rand_pool(std::time(nullptr) + reinterpret_cast<uintptr_t>(this));
        real limit = std::sqrt(6.0f / (input_size + layer_size));
        Kokkos::fill_random(weights, rand_pool, static_cast<real>(-limit), static_cast<real>(limit));
        Kokkos::deep_copy(biases, 0.0); // Initialisation des biais à 0

        zero_accumulated_gradients(); // Initialise les _sum à 0
        Kokkos::deep_copy(z, 0.0);
        Kokkos::deep_copy(a, 0.0);
        Kokkos::deep_copy(delta, 0.0);
        Kokkos::deep_copy(d_biases, 0.0);
        Kokkos::deep_copy(d_weights, 0.0);
        Kokkos::deep_copy(tmp_deriv, 0.0);
    }

    // Constructeur spécifique pour InputLayer
    Layer(int _layer_size) :
        input_size(0), layer_size(_layer_size), activation(nullptr)
    {
         if (layer_size <= 0) {
             throw std::runtime_error("Input layer size must be positive.");
         }
        a = View1D("input_a", layer_size); // Seule vue nécessaire pour InputLayer
        Kokkos::deep_copy(a, 0.0);
    }

    // --- Méthodes ---
    virtual void forward(const View1D& prev_layer_a) {
        if (input_size == 0) return; // InputLayer or uninitialized

        // Handle case where activation might be null (e.g., conceptual Linear output)
        // Though we added LinearActivation, this makes it safer
        if (!activation) {
             // Treat as linear: z = W * prev_layer_a + b; a = z
             KokkosBlas::gemv("N", 1.0, weights, prev_layer_a, 0.0, z);
             Kokkos::parallel_for("add_biases_linear", layer_size, KOKKOS_LAMBDA(int i) {
                 z(i) += biases(i);
             });
             Kokkos::deep_copy(a, z); // a = z
             return;
        }

        // 1. z = W * prev_layer_a + b
        KokkosBlas::gemv("N", 1.0, weights, prev_layer_a, 0.0, z);
        Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) {
            if (i < z.extent(0) && i < biases.extent(0)) { // Bounds check
                z(i) += biases(i);
            }
        });

        // 2. a = activation(z)
        activation->apply(z, a);
    }

    virtual void compute_gradients(const Layer& next_layer, const View1D& prev_layer_a) {
         if (input_size == 0) return; // InputLayer or uninitialized

         // Calculate f'(z) - Handle null activation case (linear)
        if (activation) {
             activation->apply_derivative(z, tmp_deriv);
        } else {
            // Derivative of linear activation is 1
            Kokkos::deep_copy(tmp_deriv, 1.0f);
        }


        // 2. Calculer delta : δ_l = (W_{l+1}^T * δ_{l+1}) .* f'(z_l)
        View1D delta_prop("delta_prop", layer_size);
        KokkosBlas::gemv("T", 1.0, next_layer.weights, next_layer.delta, 0.0, delta_prop);
         Kokkos::parallel_for("hadamard_delta_deriv", layer_size, KOKKOS_LAMBDA(int i) {
             if (i < delta.extent(0)) { // Bounds check
                delta(i) = delta_prop(i) * tmp_deriv(i);
             }
        });

        // 3. Calculer gradients instantanés pour ce sample (stockés dans d_weights/d_biases)
        Kokkos::deep_copy(d_biases, delta); // d_biases = delta
        Kokkos::parallel_for("compute_d_weights",
             Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
             KOKKOS_LAMBDA(const int i, const int j) {
                 if (i < d_weights.extent(0) && j < d_weights.extent(1)) { // Bounds check
                    d_weights(i, j) = delta(i) * prev_layer_a(j);
                 }
        });


        // 4. Accumuler les gradients (pour le batch)
         Kokkos::parallel_for("accumulate_gradients",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                 if (i < d_weights_sum.extent(0) && j < d_weights_sum.extent(1)) { // Bounds check
                    Kokkos::atomic_add(&d_weights_sum(i, j), d_weights(i, j));
                 }
         });
         Kokkos::parallel_for("accumulate_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
              if (i < d_biases_sum.extent(0)) { // Bounds check
                 Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
              }
         });
    }

    void zero_accumulated_gradients() {
         if (input_size == 0) return;
         Kokkos::deep_copy(d_biases_sum, 0.0);
         Kokkos::deep_copy(d_weights_sum, 0.0);
    }

    void show() const {
        if (input_size == 0) {
            std::cout << "Input Layer (size " << layer_size << ")" << std::endl;
            return;
        }
        std::cout << "Layer (" << input_size << " -> " << layer_size << "):" << std::endl;
        auto h_weights = Kokkos::create_mirror_view(weights);
        auto h_biases = Kokkos::create_mirror_view(biases);
        Kokkos::deep_copy(h_weights, weights);
        Kokkos::deep_copy(h_biases, biases);
        Kokkos::fence();

        std::cout << "  Weights (sample " << weights.extent(0) << "x" << weights.extent(1) << "):" << std::endl;
        for(int i=0; i< std::min((int)weights.extent(0), 5) ; ++i) {
             std::cout << "    [";
             for(int j=0; j< std::min((int)weights.extent(1), 5); ++j) {
                 std::cout << std::fixed << std::setprecision(3) << h_weights(i,j) << " ";
             }
             if (weights.extent(1) > 5) std::cout << "...";
             std::cout << "]" << std::endl;
        }
         if (weights.extent(0) > 5) std::cout << "    ..." << std::endl;

        std::cout << "  Biases (sample " << biases.extent(0) << "): [";
        for(int i=0; i< std::min((int)biases.extent(0), 10); ++i) {
            std::cout << std::fixed << std::setprecision(3) << h_biases(i) << " ";
        }
        if (biases.extent(0) > 10) std::cout << "...";
        std::cout << "]" << std::endl;
    }

    // Gérer déplacement/copie
    Layer(Layer&& other) = default;
    Layer& operator=(Layer&& other) = default;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    virtual ~Layer() = default;
};

// --- Classe InputLayer (inchangée) ---
class InputLayer : public Layer {
public:
    InputLayer(int _layer_size) : Layer(_layer_size) {}

    void set_input(const View1D& input_data) {
         if (input_data.extent(0) != layer_size) {
             throw std::runtime_error("Input data size mismatch for InputLayer: Expected "
                + std::to_string(layer_size) + ", Got " + std::to_string(input_data.extent(0)));
         }
         Kokkos::deep_copy(a, input_data);
     }
};

// --- Classe OutputLayer (modifiée pour gérer activation linéaire via compute_gradients) ---
class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        Layer(_input_size, _layer_size, std::move(act_func)) {}

    // Calcule les gradients pour la couche de sortie
    void compute_gradients(const View1D& target, const View1D& prev_layer_a) {
         // Pas besoin de vérifier !activation ici car Layer::compute_gradients le ferait,
         // mais on le fait spécifiquement pour la partie delta_L = (a_L - y) * f'(z_L)
         if (target.extent(0) != layer_size) {
             throw std::runtime_error("Target size mismatch for OutputLayer gradient calculation.");
         }

        // 1. Calculer f'(z) - Géré dans la boucle delta
        if (activation) {
            activation->apply_derivative(z, tmp_deriv);
        } else {
             // Assume linear if activation is null
            Kokkos::deep_copy(tmp_deriv, 1.0f);
        }

        // 2. Calculer delta : δ_L = (a_L - y) .* f'(z_L)
        Kokkos::parallel_for("compute_output_delta", layer_size, KOKKOS_LAMBDA(int i) {
             if (i < delta.extent(0)) { // Bounds check
                // tmp_deriv contient soit la dérivée réelle soit 1.0f (pour linéaire)
                delta(i) = (a(i) - target(i)) * tmp_deriv(i);
             }
        });

       // 3. Calculer gradients instantanés (identique à Layer)
        Kokkos::deep_copy(d_biases, delta);
        Kokkos::parallel_for("compute_output_d_weights",
             Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
             KOKKOS_LAMBDA(const int i, const int j) {
                 if (i < d_weights.extent(0) && j < d_weights.extent(1)) { // Bounds check
                    d_weights(i, j) = delta(i) * prev_layer_a(j);
                 }
        });

        // 4. Accumuler les gradients (identique à Layer)
        Kokkos::parallel_for("accumulate_output_gradients",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                if (i < d_weights_sum.extent(0) && j < d_weights_sum.extent(1)) { // Bounds check
                    Kokkos::atomic_add(&d_weights_sum(i, j), d_weights(i, j));
                }
        });
         Kokkos::parallel_for("accumulate_output_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             if (i < d_biases_sum.extent(0)) { // Bounds check
                 Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
             }
        });
    }
};


// --- Classes Dataset / BatchHandler (Placeholders inchangés) ---
class Dataset {};
class BatchHandler {};


// --- Classe Network (inchangée par rapport à la version précédente) ---
class Network {
public:
    std::map<int, int> layer_sizes;
    InputLayer input_layer;
    std::vector<Layer> hidden_layers;
    OutputLayer output_layer;
    std::unique_ptr<Optimizer> optimizer; // Gestionnaire de l'optimiseur
    Dataset dataset; // Placeholder
    BatchHandler batch_handler; // Placeholder

    // Constructeur
    Network(const std::map<int, int>& _layer_sizes,
            const std::vector<std::string>& activation_types,
            std::unique_ptr<Optimizer> opt) // Prend un optimizer (obligatoire maintenant)
        : layer_sizes(_layer_sizes),
          input_layer(get_size(0)),
          output_layer(get_size(num_layers() - 2), get_size(num_layers() - 1), create_activation(activation_types.back())),
          optimizer(std::move(opt)) // Prend possession de l'optimizer
    {
        if (layer_sizes.size() < 2) {
            throw std::runtime_error("Network must have at least an input and output layer.");
        }
        if (activation_types.size() != layer_sizes.size() -1) {
             throw std::runtime_error("Number of activation types must match number of hidden + output layers.");
        }
        if (!optimizer) { // Vérification ajoutée
             throw std::runtime_error("Optimizer must be provided to the Network constructor.");
        }

        int num_hidden_layers = layer_sizes.size() - 2;
        hidden_layers.reserve(num_hidden_layers);
        for (int i = 0; i < num_hidden_layers; ++i) {
            int current_layer_idx = i + 1;
            int in_size = get_size(current_layer_idx - 1);
            int out_size = get_size(current_layer_idx);
            std::string act_type = activation_types[i];
            hidden_layers.emplace_back(in_size, out_size, create_activation(act_type));
        }
    }

    // Helper
    int get_size(int layer_index) const { return layer_sizes.at(layer_index); }
    int num_layers() const { return layer_sizes.size(); }

    // Passe avant
    View1D forward(const View1D& input_data) {
        input_layer.set_input(input_data);
        View1D current_a = input_layer.a;
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.forward(current_a);
            current_a = hidden_layer.a;
        }
        output_layer.forward(current_a);
        return output_layer.a;
    }

    // Passe arrière
    void backward(const View1D& target) {
        // Backprop de sortie vers dernière cachée (ou entrée si pas de cachée)
        const View1D& prev_a_output = hidden_layers.empty() ? input_layer.a : hidden_layers.back().a;
        output_layer.compute_gradients(target, prev_a_output);

        // Backprop à travers les couches cachées (de droite à gauche)
        Layer* next_layer_ptr = &output_layer;
        for (int i = hidden_layers.size() - 1; i >= 0; --i) {
            Layer& current_layer = hidden_layers[i];
            const View1D& prev_a = (i == 0) ? input_layer.a : hidden_layers[i - 1].a;
            current_layer.compute_gradients(*next_layer_ptr, prev_a);
            next_layer_ptr = &current_layer;
        }
    }

     // Mise à jour des poids
    void update(int batch_size) {
        if (!optimizer) {
             throw std::runtime_error("Optimizer not set in Network.");
        }
         if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for update.");
        }

        for (Layer& layer : hidden_layers) {
            optimizer->update(layer.weights, layer.biases,
                              layer.d_weights_sum, layer.d_biases_sum,
                              batch_size);
        }
        optimizer->update(output_layer.weights, output_layer.biases,
                          output_layer.d_weights_sum, output_layer.d_biases_sum,
                          batch_size);
    }

    // Remise à zéro des gradients accumulés
    void zero_accumulated_gradients() {
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.zero_accumulated_gradients();
        }
        output_layer.zero_accumulated_gradients();
    }

    // Calcul du coût (MSE)
    real calculate_cost(const View1D& prediction, const View1D& target) {
        int output_size = prediction.extent(0);
        if (target.extent(0) != output_size) {
            throw std::runtime_error("Prediction and target size mismatch for cost calculation.");
        }
        real squared_error_sum = 0.0;
        Kokkos::parallel_reduce("compute_cost", output_size, KOKKOS_LAMBDA (int i, real& lsum) {
            real diff = prediction(i) - target(i);
            lsum += diff * diff;
        }, squared_error_sum);
        Kokkos::fence(); // Assurer que la réduction est terminée
        return 0.5 * squared_error_sum; // MSE = 1/N * 0.5 * sum(...) - N (batch_size) est implicite dans la boucle d'entraînement
    }

    // Gestion de l'optimizer
    void set_optimizer(std::unique_ptr<Optimizer> opt) {
        if (!opt) {
            throw std::runtime_error("Cannot set a null optimizer.");
        }
        optimizer = std::move(opt);
    }
    Optimizer* get_optimizer() const { return optimizer.get(); }
    void set_learning_rate(real lr) {
        if (!optimizer) throw std::runtime_error("Optimizer not set.");
        optimizer->set_learning_rate(lr);
    }
     real get_learning_rate() const {
         if (!optimizer) return 0.0; // Ou throw
        return optimizer->get_learning_rate();
    }

    // Affichage
    void show() const {
         std::cout << "--- Network Structure ---" << std::endl;
         input_layer.show();
         int i = 1;
         for (const auto& layer : hidden_layers) {
              std::cout << "\n--- Hidden Layer " << i++ << " ---" << std::endl;
              layer.show();
         }
         std::cout << "\n--- Output Layer ---" << std::endl;
         output_layer.show();
         std::cout << "\n--- Optimizer Info ---" << std::endl;
         if(optimizer) {
            std::cout << "  " << optimizer->get_info() << std::endl;
         } else {
            std::cout << "  Optimizer: Not set" << std::endl;
         }
         std::cout << "-------------------------" << std::endl;
    }

    // Déplacement/Copie
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    virtual ~Network() = default;
};





// --- Fonction d'entraînement XOR (adaptée pour choisir l'optimizer) ---
void xor_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- XOR Training Example ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    // Structure du réseau
    std::map<int, int> sizes;
    sizes[0] = 2; // Input
    sizes[1] = 10000; // Hidden 1 (Un peu plus de neurones peuvent aider XOR)
    sizes[2] = 10000 ; 
    sizes[3] = 10000 ; 
    sizes[4] = 1; // Output

    // Sigmoid ou Tanh sont souvent utilisés pour XOR avec une seule couche cachée
    std::vector<std::string> activations = {"relu", "relu", "relu", "sigmoid"}; // H1, Out
//    std::vector<std::string> activations = {"tanh", "tanh", "tanh", "sigmoid"}; // H1, Out

    // Paramètres d'entraînement (peuvent nécessiter ajustement selon l'optimizer)
    real learning_rate;
    int epochs;
    int batch_size = 4; // Utiliser toutes les données (Batch GD)

    // Créer l'optimizer choisi
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 1.5; // SGD peut nécessiter un LR plus élevé
        epochs = 600;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else if (optimizer_choice == "adam") {
        learning_rate = 0.001; // Adam a souvent besoin d'un LR plus faible
        epochs = 1500;       // Adam converge souvent plus vite
        optimizer = std::make_unique<Adam>(learning_rate, 0.9, 0.999, 1e-8);
    } else {
        throw std::runtime_error("Unknown optimizer choice: " + optimizer_choice);
    }


    // Créer le réseau
    Network dnn(sizes, activations, std::move(optimizer));

    // Données XOR
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int num_samples = 4;
    const int input_dim = sizes[0];
    const int output_dim = sizes[sizes.size()-1];
    HostView2D h_xor_inputs("h_xor_inputs", num_samples, input_dim);
    HostView2D h_xor_outputs("h_xor_outputs", num_samples, output_dim);
    h_xor_inputs(0, 0) = 0.0; h_xor_inputs(0, 1) = 0.0; h_xor_outputs(0, 0) = 0.0;
    h_xor_inputs(1, 0) = 1.0; h_xor_inputs(1, 1) = 0.0; h_xor_outputs(1, 0) = 1.0;
    h_xor_inputs(2, 0) = 0.0; h_xor_inputs(2, 1) = 1.0; h_xor_outputs(2, 0) = 1.0;
    h_xor_inputs(3, 0) = 1.0; h_xor_inputs(3, 1) = 1.0; h_xor_outputs(3, 0) = 0.0;
    auto xor_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_inputs);
    auto xor_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_outputs);

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << std::endl;


    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        dnn.zero_accumulated_gradients();

        // Itérer sur les samples du batch
        for (int i = 0; i < num_samples; ++i) {
            auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
            auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());

            View1D prediction = dnn.forward(input_subview);
            real sample_cost = dnn.calculate_cost(prediction, target_subview);
            total_epoch_cost += sample_cost;
            dnn.backward(target_subview); // Accumule les gradients
        }

        // Mettre à jour les poids après le batch
        dnn.update(batch_size);

        // Afficher le coût moyen de l'époque
        real avg_cost = total_epoch_cost / num_samples;
        if ((epoch + 1) % (epochs / 100) == 0 || epoch == 0 || epoch == epochs -1) { // Affichage ~10 fois
            std::cout << "Epoch: " << std::setw(6) << epoch + 1
                      << ", Average Cost: " << std::fixed << std::setprecision(8)
                      << avg_cost << std::endl;
        }

        // Stopper si le coût est suffisamment bas
        if (avg_cost < 1e-5) {
             std::cout << "Convergence reached at epoch " << epoch + 1 << std::endl;
             break;
        }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    // Afficher la structure et les poids finaux
    // dnn.show(); // Peut être très verbeux

    // Tester les prédictions
    std::cout << "\n-- PREDICTIONS --" << std::endl;
    View1D prediction_result("prediction_result", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);

    real final_cost = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());

        View1D prediction = dnn.forward(input_subview); // Utilise les poids finaux
        final_cost += dnn.calculate_cost(prediction, target_subview);

        Kokkos::deep_copy(prediction_result, prediction); // Copie device->device (ou device->host si prediction est déjà host)
        Kokkos::deep_copy(h_prediction_result, prediction_result); // Copie device->host
        Kokkos::fence(); // Assurer la fin des copies

        std::cout << "Input: [" << h_xor_inputs(i,0) << "," << h_xor_inputs(i,1) << "] "
                  << "Target: " << h_xor_outputs(i,0) << " "
                  << "Prediction: " << std::fixed << std::setprecision(4) << h_prediction_result(0)
                  << " (Rounded: " << std::round(h_prediction_result(0)) << ")" << std::endl;
    }
     std::cout << "Final Average Cost: " << std::fixed << std::setprecision(8) << final_cost / num_samples << std::endl;

}

// --- Main (inchangé) ---
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    { // Scope for Kokkos objects
        try {
            std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;

            // Lancer l'entraînement avec Adam par défaut
            xor_train("adam");
            std::cout << "\n---------------------------\n" << std::endl;
             // Lancer l'entraînement avec SGD pour comparaison
             xor_train("sgd");

            std::cout << "\n---------------------------\n" << std::endl;
	    sine_train("adam") ; 

            std::cout << "\n---------------------------\n" << std::endl;
	     linear_sep_train("adam") ;	    


        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            Kokkos::finalize(); // Ensure Kokkos is finalized even on error
            return 1;
        } catch (...) {
            std::cerr << "An unknown error occurred." << std::endl;
             Kokkos::finalize();
             return 1;
        }

    } // Kokkos objects go out of scope
    Kokkos::finalize();
    return 0;
}
