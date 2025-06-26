#include <iostream>
#include <limits>
#include <cmath> // Pour sin, M_PI (peut nécessiter -lm ou définition si M_PI manque)
#include <fstream>
#include <filesystem>
#include <random>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <numeric> // Pour std::iota
#include <iomanip> // Pour std::setprecision
#include <typeinfo> // Pour typeid dans Network::show
#include <chrono>  // Pour seed random

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_StdAlgorithms.hpp> // Pour parallel_reduce

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
         // Check added: Ensure dimensions are valid before accessing
        if (biases.extent_int(0) <= 0 || weights.extent_int(0) <= 0 || weights.extent_int(1) <= 0 ) {
            // Layer might be valid but empty (though unlikely for trainable layers)
            if (biases.extent(0) == 0 && weights.extent(0) == 0) return; // Still allow empty update
             throw std::runtime_error("SGD update called on layer with invalid dimensions.");
        }


        const real scale = learning_rate / static_cast<real>(batch_size);
        const int layer_size = biases.extent_int(0); // Use extent_int for safety with potentially large dimensions
        const int input_size = weights.extent_int(1); // Assuming weights is layer_size x input_size

        // Update weights: weights -= learning_rate/batch_size * accumulated_d_weights
        Kokkos::parallel_for("sgd_update_weights",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                // Check bounds (though MDRangePolicy should handle it)
                // if (i < weights.extent(0) && j < weights.extent(1)) { // Redundant with MDRangePolicy
                     weights(i, j) -= scale * accumulated_d_weights(i, j);
                // }
        });

        // Update biases: biases -= learning_rate/batch_size * accumulated_d_biases
        Kokkos::parallel_for("sgd_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
            // if (i < biases.extent(0)) { // Redundant with Kokkos range
                biases(i) -= scale * accumulated_d_biases(i);
            // }
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
        ParameterState(size_t size) : t(0) { // For biases - use size_t
            if (size > std::numeric_limits<int>::max()) {
                 throw std::runtime_error("Adam state bias size exceeds limits.");
            }
            m_1d = View1D("adam_m1d", size);
            v_1d = View1D("adam_v1d", size);
            Kokkos::deep_copy(m_1d, 0.0);
            Kokkos::deep_copy(v_1d, 0.0);
        }
        ParameterState(size_t rows, size_t cols) : t(0) { // For weights - use size_t
             if (rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max()) {
                 throw std::runtime_error("Adam state weight dimensions exceed limits.");
            }
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
         // Check added: Ensure dimensions are valid before accessing
        if (biases.extent_int(0) <= 0 || weights.extent_int(0) <= 0 || weights.extent_int(1) <= 0 ) {
             if (biases.extent(0) == 0 && weights.extent(0) == 0) return; // Allow empty update
             throw std::runtime_error("Adam update called on layer with invalid dimensions.");
        }

        const int layer_size = biases.extent_int(0); // Use extent_int
        const int input_size = weights.extent_int(1); // Use extent_int
        const real scale = 1.0 / static_cast<real>(batch_size); // Scale for averaging gradients

        // --- Update Weights ---
        {
            // Get or create state for weights
            auto it_w = state_map_2d.find(weights.data());
            if (it_w == state_map_2d.end()) {
                // Create state if it doesn't exist
                auto result = state_map_2d.try_emplace(weights.data(), weights.extent(0), weights.extent(1)); // Use extents directly
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
                    // if (i < weights.extent(0) && j < weights.extent(1)) { // Redundant
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
                    // }
            });
        } // End scope for weights update

        // --- Update Biases ---
        {
             // Get or create state for biases
             auto it_b = state_map_1d.find(biases.data());
             if (it_b == state_map_1d.end()) {
                 auto result = state_map_1d.try_emplace(biases.data(), biases.extent(0)); // Use extent directly
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
                 // if (i < biases.extent(0)) { // Redundant
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
                 // }
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
    // input and output views must have the same extent
    virtual void apply(const View1D& input, View1D& output) const = 0;
    // input_z and output_deriv views must have the same extent
    virtual void apply_derivative(const View1D& input_z, View1D& output_deriv) const = 0;
};

class LinearActivation : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override {
        // output(i) = input(i) -> Just copy
        Kokkos::deep_copy(output, input);
    }
    void apply_derivative(const View1D& /*input_z*/, View1D& output_deriv) const override {
        // f'(z) = 1
        // Use deep_copy for potentially better performance on large views
        Kokkos::deep_copy(output_deriv, 1.0f);
    }
};

class RELU : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent_int(0);
        Kokkos::parallel_for("relu_apply", size, KOKKOS_LAMBDA(const int i) {
             // if (i < output.extent(0)) // Redundant
                output(i) = (input(i) > 0.0f) ? input(i) : 0.0f;
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent_int(0);
        Kokkos::parallel_for("relu_deriv", size, KOKKOS_LAMBDA(const int i) {
             // if (i < output_deriv.extent(0)) // Redundant
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
        const int size = input.extent_int(0);
        Kokkos::parallel_for("sigmoid_apply", size, KOKKOS_LAMBDA(const int i) {
             // if (i < output.extent(0)) // Redundant
                output(i) = scalar_sigmoid(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent_int(0);
        Kokkos::parallel_for("sigmoid_deriv", size, KOKKOS_LAMBDA(const int i) {
            // if (i < output_deriv.extent(0)) { // Redundant
                real s = scalar_sigmoid(input_z(i));
                output_deriv(i) = s * (1.0f - s);
            // }
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
        const int size = input.extent_int(0);
        Kokkos::parallel_for("tanh_apply", size, KOKKOS_LAMBDA(const int i) {
             // if (i < output.extent(0)) // Redundant
                output(i) = scalar_tanh(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent_int(0);
        Kokkos::parallel_for("tanh_deriv", size, KOKKOS_LAMBDA(const int i) {
             // if (i < output_deriv.extent(0)) { // Redundant
                real t = scalar_tanh(input_z(i));
                output_deriv(i) = 1.0f - t * t;
             // }
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


// --- Classe Layer (utilisation de extent_int et vérifications) ---
class Layer {
public:
    int input_size; // Can be 0 for InputLayer
    int layer_size; // Number of neurons in this layer
    std::unique_ptr<Activation> activation; // Can be nullptr for InputLayer or implicit linear

    // Kokkos views for parameters and states
    View2D weights; // [layer_size x input_size]
    View1D biases;  // [layer_size]
    View1D z;       // [layer_size] - Output before activation
    View1D a;       // [layer_size] - Output after activation (activation(z))

    // Kokkos views for backpropagation (gradients)
    View1D delta;          // [layer_size] - Propagated error (δ) for this layer
    View1D d_biases;       // [layer_size] - Instantaneous bias gradient (∂Cost/∂b = δ) for ONE sample
    View2D d_weights;      // [layer_size x input_size] - Instantaneous weight gradient (∂Cost/∂W = δ * a_prev^T) for ONE sample
    View1D d_biases_sum;   // [layer_size] - Sum of bias gradients over the batch
    View2D d_weights_sum;  // [layer_size x input_size] - Sum of weight gradients over the batch

    View1D tmp_deriv;      // [layer_size] - Temporary storage for activation derivative f'(z)


    // Constructor for hidden/output layers
    Layer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        input_size(_input_size),
        layer_size(_layer_size),
        activation(std::move(act_func))
    {
        if (input_size < 0 || layer_size <= 0) { // input_size can be 0, but layer_size must be > 0
             throw std::runtime_error("Layer sizes must be non-negative, layer_size > 0.");
        }
        // Only allocate weights/biases/gradients if it's not effectively an input layer structure passed here
        if (input_size > 0) {
            weights = View2D("weights", layer_size, input_size);
            biases = View1D("biases", layer_size);
            z = View1D("z", layer_size);
            delta = View1D("delta", layer_size);
            d_biases = View1D("d_biases", layer_size);
            d_weights = View2D("d_weights", layer_size, input_size);
            d_biases_sum = View1D("d_biases_sum", layer_size);
            d_weights_sum = View2D("d_weights_sum", layer_size, input_size);
            tmp_deriv = View1D("tmp_deriv", layer_size);

            // Initialize Xavier/Glorot (uniform)
            Kokkos::Random_XorShift64_Pool<> rand_pool(std::chrono::high_resolution_clock::now().time_since_epoch().count() + reinterpret_cast<uintptr_t>(this));

            real limit = (input_size + layer_size > 0) ? std::sqrt(6.0f / (input_size + layer_size)) : 1.0f; // Avoid division by zero
            Kokkos::fill_random(weights, rand_pool, static_cast<real>(-limit), static_cast<real>(limit));
            Kokkos::deep_copy(biases, 0.0); // Initialize biases to 0

            // Initialize sums and temporary views
            zero_accumulated_gradients();
            Kokkos::deep_copy(z, 0.0);
            Kokkos::deep_copy(delta, 0.0);
            Kokkos::deep_copy(d_biases, 0.0);
            Kokkos::deep_copy(d_weights, 0.0);
            Kokkos::deep_copy(tmp_deriv, 0.0);
        } else {
             // If input_size is 0, only 'a' is needed conceptually (like InputLayer)
             // Make other views valid but size 0 to avoid null access issues later if logic tries to use them
             weights = View2D("weights_empty", 0, 0);
             biases = View1D("biases_empty", 0);
             z = View1D("z_empty", 0);
             delta = View1D("delta_empty", 0);
             d_biases = View1D("d_biases_empty", 0);
             d_weights = View2D("d_weights_empty", 0, 0);
             d_biases_sum = View1D("d_biases_sum_empty", 0);
             d_weights_sum = View2D("d_weights_sum_empty", 0, 0);
             tmp_deriv = View1D("tmp_deriv_empty", 0);
        }

        // 'a' is always needed (output activation)
        a = View1D("a", layer_size);
        Kokkos::deep_copy(a, 0.0);
    }

    // Constructor specifically for InputLayer (sets input_size to 0)
    Layer(int _layer_size) :
        Layer(0, _layer_size, nullptr) // Delegate to the main constructor
    {
         // 'a' is already allocated and sized correctly by the delegated constructor
    }

    // --- Methods ---
    virtual void forward(const View1D& prev_layer_a) {
        if (input_size == 0) {
             // Should only happen for InputLayer logic which doesn't call forward this way
             // Or potentially if a layer was constructed with input_size=0 mistakenly.
             // We could throw an error, or just return if 'a' is already set externally (like InputLayer)
             // Let's assume 'a' is managed elsewhere for input_size=0 cases.
             return;
         }
         if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("Forward pass: prev_layer_a size (" + std::to_string(prev_layer_a.extent(0))
                                     + ") mismatch with layer input_size (" + std::to_string(input_size) + ")");
         }

        // 1. z = W * prev_layer_a + b
        // Check dimensions before BLAS call for safety
        if (weights.extent(0) != static_cast<size_t>(layer_size) || weights.extent(1) != static_cast<size_t>(input_size) ||
            z.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Dimension mismatch before GEMV in forward pass.");
        }
        KokkosBlas::gemv("N", 1.0, weights, prev_layer_a, 0.0, z);

        // Check bias dimension before adding
        if (biases.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Bias dimension mismatch in forward pass.");
        }
        Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) {
            // if (i < z.extent(0) && i < biases.extent(0)) { // Redundant check
                z(i) += biases(i);
            // }
        });

        // 2. a = activation(z)
        // Check activation output dimension
        if (a.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Activation output 'a' dimension mismatch in forward pass.");
        }
        if (activation) {
            activation->apply(z, a);
        } else {
            // Treat as linear if no activation function is provided
            Kokkos::deep_copy(a, z);
        }
    }

    // Computes gradients for a hidden layer (requires next layer's delta and weights)
    // Resulting gradients (d_weights, d_biases) are instantaneous for the *current sample*
    // These are then accumulated into d_weights_sum, d_biases_sum
    virtual void compute_gradients(const Layer& next_layer, const View1D& prev_layer_a) {
         if (input_size == 0) return; // Cannot compute gradients if no inputs/weights (e.g., InputLayer)

         if (next_layer.weights.extent(1) != static_cast<size_t>(layer_size) || // next W rows must match current layer size
             next_layer.delta.extent(0) != next_layer.weights.extent(0)) {     // next delta size must match next layer size
             throw std::runtime_error("Dimension mismatch between current layer and next layer for gradient computation.");
         }
          if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("Dimension mismatch for prev_layer_a in gradient computation.");
         }
          if (delta.extent(0) != static_cast<size_t>(layer_size) ||
              tmp_deriv.extent(0) != static_cast<size_t>(layer_size)) {
              throw std::runtime_error("Internal dimension mismatch for delta or tmp_deriv.");
          }


        // 1. Calculate f'(z) -> stored in tmp_deriv
        if (activation) {
             activation->apply_derivative(z, tmp_deriv);
        } else {
            // Derivative of linear activation is 1
            Kokkos::deep_copy(tmp_deriv, 1.0f);
        }


        // 2. Calculate delta for this layer: δ_l = (W_{l+1}^T * δ_{l+1}) .* f'(z_l)
        // Temporary view to store the result of W^T * delta_next
        View1D delta_prop("delta_prop", layer_size); // Must match current layer size
        KokkosBlas::gemv("T", 1.0, next_layer.weights, next_layer.delta, 0.0, delta_prop);

        // Element-wise product (Hadamard product)
         Kokkos::parallel_for("hadamard_delta_deriv", layer_size, KOKKOS_LAMBDA(int i) {
             // if (i < delta.extent(0)) { // Redundant
                delta(i) = delta_prop(i) * tmp_deriv(i);
             // }
        });

        // 3. Calculate INSTANTANEOUS gradients for this sample (store in d_weights/d_biases)
        // d_biases = delta (∂Cost/∂b = δ)
        if (d_biases.extent(0) != delta.extent(0)) throw std::runtime_error("d_biases/delta size mismatch.");
        Kokkos::deep_copy(d_biases, delta);

        // d_weights = delta * prev_layer_a^T (outer product) (∂Cost/∂W = δ * a_prev^T)
         if (d_weights.extent(0) != static_cast<size_t>(layer_size) || d_weights.extent(1) != static_cast<size_t>(input_size)) {
              throw std::runtime_error("d_weights dimension mismatch before outer product.");
         }
        Kokkos::parallel_for("compute_d_weights",
             Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
             KOKKOS_LAMBDA(const int i, const int j) {
                 // if (i < d_weights.extent(0) && j < d_weights.extent(1)) { // Redundant
                    d_weights(i, j) = delta(i) * prev_layer_a(j);
                 // }
        });


        // 4. ACCUMULATE these instantaneous gradients into the sum views (atomic updates)
         if (d_weights_sum.extent(0) != d_weights.extent(0) || d_weights_sum.extent(1) != d_weights.extent(1)) {
             throw std::runtime_error("Accumulated d_weights dimension mismatch.");
         }
         Kokkos::parallel_for("accumulate_weight_gradients",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                 // if (i < d_weights_sum.extent(0) && j < d_weights_sum.extent(1)) { // Redundant
                    Kokkos::atomic_add(&d_weights_sum(i, j), d_weights(i, j));
                 // }
         });

          if (d_biases_sum.extent(0) != d_biases.extent(0)) {
             throw std::runtime_error("Accumulated d_biases dimension mismatch.");
         }
         Kokkos::parallel_for("accumulate_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
              // if (i < d_biases_sum.extent(0)) { // Redundant
                 Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
              // }
         });
    }

    // Zero out the gradient sums (typically called at the start of each batch)
    void zero_accumulated_gradients() {
         if (input_size == 0) return; // No gradients to zero if no weights/biases
         if (d_biases_sum.data() != nullptr) Kokkos::deep_copy(d_biases_sum, 0.0);
         if (d_weights_sum.data() != nullptr) Kokkos::deep_copy(d_weights_sum, 0.0);
    }

    // Display layer info (optional, for debugging)
    void show() const {
        if (input_size == 0 && activation == nullptr) { // Signature of InputLayer
            std::cout << "Input Layer (size " << layer_size << ")" << std::endl;
            return;
        }
        std::cout << "Layer (" << input_size << " -> " << layer_size << "):";
        if (activation) {
             // Attempt to get activation type info (might be compiler specific)
             // Use typeid safely
             try {
                 std::string act_name = typeid(*activation).name();
                 // Basic demangling attempt (platform dependent)
                 // Extract base name if possible
                 size_t last_colon = act_name.find_last_of("::");
                 if(last_colon != std::string::npos) act_name = act_name.substr(last_colon+1);
                 std::cout << " Activation: " << act_name;
             } catch (const std::exception& e) {
                 std::cout << " Activation: [Unknown - typeid failed]";
             }
         } else {
             std::cout << " Activation: Linear (Implicit)";
         }
         std::cout << std::endl;


        // Avoid printing huge weight matrices if layers are very large
        const int max_print_rows = 5;
        const int max_print_cols = 5;
        const int max_print_biases = 10;

        if (weights.data() != nullptr && weights.extent(0) > 0 && weights.extent(1) > 0) {
            auto h_weights = Kokkos::create_mirror_view(weights);
            Kokkos::deep_copy(h_weights, weights);
            Kokkos::fence();

            std::cout << "  Weights (" << weights.extent(0) << "x" << weights.extent(1) << ", showing sample):" << std::endl;
            for(int i=0; i< std::min((int)weights.extent(0), max_print_rows) ; ++i) {
                 std::cout << "    [";
                 for(int j=0; j< std::min((int)weights.extent(1), max_print_cols); ++j) {
                     std::cout << std::fixed << std::setprecision(3) << h_weights(i,j) << " ";
                 }
                 if (weights.extent(1) > max_print_cols) std::cout << "...";
                 std::cout << "]" << std::endl;
            }
             if (weights.extent(0) > max_print_rows) std::cout << "    ..." << std::endl;
        } else {
             std::cout << "  Weights: N/A (Input Layer or uninitialized)" << std::endl;
        }

        if (biases.data() != nullptr && biases.extent(0) > 0) {
            auto h_biases = Kokkos::create_mirror_view(biases);
            Kokkos::deep_copy(h_biases, biases);
            Kokkos::fence();

            std::cout << "  Biases (" << biases.extent(0) << ", showing sample): [";
            for(int i=0; i< std::min((int)biases.extent(0), max_print_biases); ++i) {
                std::cout << std::fixed << std::setprecision(3) << h_biases(i) << " ";
            }
            if (biases.extent(0) > max_print_biases) std::cout << "...";
            std::cout << "]" << std::endl;
         } else {
             std::cout << "  Biases: N/A (Input Layer or uninitialized)" << std::endl;
         }
    }

    // Manage move/copy semantics (default move is fine, disable copy)
    Layer(Layer&& other) = default;
    Layer& operator=(Layer&& other) = default;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    virtual ~Layer() = default;
};


// --- Classe InputLayer (utilisation de l'héritage de Layer) ---
class InputLayer : public Layer {
public:
    InputLayer(int _layer_size) : Layer(_layer_size) {} // Calls Layer(0, _layer_size, nullptr) implicitly via delegation

    // Method to load data into the 'a' view of the input layer
    void set_input(const View1D& input_data) {
         if (input_data.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Input data size mismatch for InputLayer: Expected "
                + std::to_string(layer_size) + ", Got " + std::to_string(input_data.extent(0)));
         }
         // Ensure 'a' view is valid before copying
         if (a.data() == nullptr || a.extent(0) != static_cast<size_t>(layer_size)) {
              throw std::runtime_error("InputLayer 'a' view is not correctly initialized or sized.");
         }
         Kokkos::deep_copy(a, input_data);
     }

     // Override forward and compute_gradients to do nothing or throw, as they aren't applicable
     void forward(const View1D& /*prev_layer_a*/) override {
         // Input layer doesn't compute forward pass based on previous layer
         // Its 'a' is set by set_input()
     }

     void compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) override {
          // Input layer doesn't have weights/biases, so no gradients to compute
     }
};


// --- Classe OutputLayer (utilisation de l'héritage de Layer) ---
class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        Layer(_input_size, _layer_size, std::move(act_func)) {}

    // Special gradient computation for the output layer (based on target values)
    // Calculates instantaneous gradients and accumulates them
    void compute_gradients(const View1D& target, const View1D& prev_layer_a) {
        // Use the base class implementation but provide the specific delta calculation for the output layer.
         if (target.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Target size ("+ std::to_string(target.extent(0))
                                     +") mismatch for OutputLayer gradient calculation (Layer size: "
                                     + std::to_string(layer_size) +").");
         }
          if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("prev_layer_a size mismatch in OutputLayer gradient calculation.");
         }
          if (delta.extent(0) != static_cast<size_t>(layer_size) ||
              a.extent(0) != static_cast<size_t>(layer_size) ||
              tmp_deriv.extent(0) != static_cast<size_t>(layer_size)) {
              throw std::runtime_error("Internal dimension mismatch in OutputLayer gradient calculation.");
          }

        // 1. Calculate f'(z_L) -> stored in tmp_deriv
        if (activation) {
            activation->apply_derivative(z, tmp_deriv);
        } else {
             // Assume linear if activation is null
            Kokkos::deep_copy(tmp_deriv, 1.0f);
        }

        // 2. Calculate delta for the output layer: δ_L = (a_L - y) .* f'(z_L)
        //    (Using MSE cost derivative: ∂Cost/∂a_L = a_L - y)
        Kokkos::parallel_for("compute_output_delta", layer_size, KOKKOS_LAMBDA(int i) {
             // if (i < delta.extent(0)) { // Redundant
                // tmp_deriv contains either the real derivative or 1.0f (for linear)
                delta(i) = (a(i) - target(i)) * tmp_deriv(i);
             // }
        });

       // 3. Calculate INSTANTANEOUS gradients (∂Cost/∂b_L = δ_L, ∂Cost/∂W_L = δ_L * a_{L-1}^T)
        if (d_biases.extent(0) != delta.extent(0)) throw std::runtime_error("OutputLayer d_biases/delta size mismatch.");
        Kokkos::deep_copy(d_biases, delta);

        if (d_weights.extent(0) != static_cast<size_t>(layer_size) || d_weights.extent(1) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("OutputLayer d_weights dimension mismatch.");
        }
        Kokkos::parallel_for("compute_output_d_weights",
             Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
             KOKKOS_LAMBDA(const int i, const int j) {
                 // if (i < d_weights.extent(0) && j < d_weights.extent(1)) { // Redundant
                    d_weights(i, j) = delta(i) * prev_layer_a(j);
                 // }
        });

        // 4. ACCUMULATE gradients into sum views (atomic updates)
         if (d_weights_sum.extent(0) != d_weights.extent(0) || d_weights_sum.extent(1) != d_weights.extent(1)) {
             throw std::runtime_error("OutputLayer accumulated d_weights dimension mismatch.");
         }
        Kokkos::parallel_for("accumulate_output_weight_gradients",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                // if (i < d_weights_sum.extent(0) && j < d_weights_sum.extent(1)) { // Redundant
                    Kokkos::atomic_add(&d_weights_sum(i, j), d_weights(i, j));
                // }
        });

         if (d_biases_sum.extent(0) != d_biases.extent(0)) {
             throw std::runtime_error("OutputLayer accumulated d_biases dimension mismatch.");
         }
         Kokkos::parallel_for("accumulate_output_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             // if (i < d_biases_sum.extent(0)) { // Redundant
                 Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
             // }
        });
    }

    // Override compute_gradients(Layer&, View1D&) to prevent calling the hidden layer version accidentally
     void compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) override {
         throw std::logic_error("OutputLayer::compute_gradients should be called with target, not next_layer.");
     }
};


// --- Classes Dataset / BatchHandler (Placeholders inchangés) ---
class Dataset {};
class BatchHandler {};


// --- Classe Network (revue pour utiliser InputLayer/OutputLayer et vérifier pointeurs) ---
class Network {
public:
    std::map<int, int> layer_sizes_map; // Store original sizes map if needed
    InputLayer input_layer;
    std::vector<Layer> hidden_layers; // Uses base Layer class, holds standard hidden layers
    OutputLayer output_layer;
    std::unique_ptr<Optimizer> optimizer; // Optimizer for updating weights
    Dataset dataset; // Placeholder for data management
    BatchHandler batch_handler; // Placeholder for batching logic

    // Constructor
    Network(const std::map<int, int>& _layer_sizes_map,
            const std::vector<std::string>& activation_types,
            std::unique_ptr<Optimizer> opt) // Requires an optimizer
        : layer_sizes_map(_layer_sizes_map),
          input_layer(get_size(0)), // Create InputLayer with size from map index 0
          // Create OutputLayer: input size = size of last hidden (or input if no hidden), output size = map index N-1
          output_layer(get_size(num_layers() - 2), get_size(num_layers() - 1), create_activation(activation_types.back())),
          optimizer(std::move(opt)) // Take ownership of the optimizer
    {
        if (layer_sizes_map.size() < 2) {
            throw std::runtime_error("Network must have at least an input and output layer defined in sizes map.");
        }
        // activation_types list should correspond to hidden layers + output layer
        if (activation_types.size() != layer_sizes_map.size() - 1) {
             throw std::runtime_error("Number of activation types (" + std::to_string(activation_types.size())
                                     + ") must match number of hidden + output layers ("
                                     + std::to_string(layer_sizes_map.size() - 1) + ").");
        }
        if (!optimizer) { // Ensure optimizer is provided
             throw std::runtime_error("Optimizer must be provided to the Network constructor.");
        }

        // Create hidden layers (if any)
        int num_hidden_layers = layer_sizes_map.size() - 2;
        hidden_layers.reserve(num_hidden_layers);
        for (int i = 0; i < num_hidden_layers; ++i) {
            int current_layer_idx_in_map = i + 1; // Hidden layers start from map index 1
            int in_size = get_size(current_layer_idx_in_map - 1); // Input size from previous layer
            int out_size = get_size(current_layer_idx_in_map);   // Output size from current map index
            std::string act_type = activation_types[i];          // Activation for this hidden layer
            // Use emplace_back which constructs the Layer in place
            hidden_layers.emplace_back(in_size, out_size, create_activation(act_type));
        }
    }

    // Helper to get size from the map, handling potential key errors
    int get_size(int layer_index) const {
        try {
            return layer_sizes_map.at(layer_index);
        } catch (const std::out_of_range& oor) {
            throw std::out_of_range("Layer index " + std::to_string(layer_index) + " not found in layer_sizes map.");
        }
     }
    // Total number of layers including input and output
    int num_layers() const { return layer_sizes_map.size(); }

    // Forward pass through the entire network
    View1D forward(const View1D& input_data) {
        input_layer.set_input(input_data); // Load data into input layer's 'a' view

        // Get the activation view of the input layer to start propagation
        const View1D* current_a = &input_layer.a;

        // Propagate through hidden layers
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.forward(*current_a); // Pass previous layer's activation ('a')
            current_a = &hidden_layer.a;      // Update current activation pointer
        }

        // Propagate through the output layer
        output_layer.forward(*current_a); // Pass last hidden layer's (or input layer's) activation

        return output_layer.a; // Return the final activation view
    }

    // Backward pass (backpropagation)
    void backward(const View1D& target) {
        // 1. Compute gradients for the OutputLayer first, using the target
        //    Need the activation ('a') of the layer *before* the output layer
        const View1D& prev_a_output = hidden_layers.empty() ? input_layer.a : hidden_layers.back().a;
        output_layer.compute_gradients(target, prev_a_output);

        // 2. Backpropagate through hidden layers (from right to left)
        //    Keep track of the layer to the right ("next" layer in forward pass terms)
        Layer* next_layer_ptr = &output_layer;
        for (int i = hidden_layers.size() - 1; i >= 0; --i) {
            Layer& current_layer = hidden_layers[i];
            // Need the activation ('a') of the layer *before* the current hidden layer
            const View1D& prev_a = (i == 0) ? input_layer.a : hidden_layers[i - 1].a;

            // Compute gradients for the current hidden layer, using the 'delta' and 'weights'
            // from the layer to its right (next_layer_ptr)
            current_layer.compute_gradients(*next_layer_ptr, prev_a);

            // Update the pointer to the layer to the right for the next iteration
            next_layer_ptr = &current_layer;
        }
         // Note: InputLayer has no gradients to compute, so the loop stops correctly.
    }

     // Update weights and biases using the chosen optimizer
    void update(int batch_size) {
        if (!optimizer) {
             throw std::runtime_error("Optimizer not set in Network during update.");
        }
         if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for update.");
        }

        // Update hidden layers
        for (Layer& layer : hidden_layers) {
            // Ensure the layer has weights/biases before updating
            if (layer.input_size > 0) {
                optimizer->update(layer.weights, layer.biases,
                                  layer.d_weights_sum, layer.d_biases_sum,
                                  batch_size);
            }
        }
        // Update output layer
         if (output_layer.input_size > 0) { // Should always be true unless network is just input->output
            optimizer->update(output_layer.weights, output_layer.biases,
                              output_layer.d_weights_sum, output_layer.d_biases_sum,
                              batch_size);
         }
    }

    // Zero out accumulated gradients in all layers (before starting a new batch)
    void zero_accumulated_gradients() {
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.zero_accumulated_gradients();
        }
        output_layer.zero_accumulated_gradients();
        // InputLayer has no gradients, so no need to call zero on it.
    }

    // Calculate Mean Squared Error (MSE) cost
    real calculate_cost(const View1D& prediction, const View1D& target) {
        int output_size = prediction.extent_int(0);
        if (target.extent_int(0) != output_size) {
            throw std::runtime_error("Prediction ("+ std::to_string(output_size)
                                     +") and target ("+ std::to_string(target.extent(0))
                                     +") size mismatch for cost calculation.");
        }
        if (output_size == 0) return 0.0; // No cost if output size is zero

        real squared_error_sum = 0.0;
        Kokkos::parallel_reduce("compute_cost", output_size, KOKKOS_LAMBDA (int i, real& lsum) {
            real diff = prediction(i) - target(i);
            lsum += diff * diff;
        }, squared_error_sum);
        Kokkos::fence(); // Ensure reduction is complete before returning

        // Standard MSE cost is often defined as 0.5 * (1/N) * sum(...)
        // Here, we return 0.5 * sum(...). The (1/N) scaling (where N is batch size or dataset size)
        // should be handled implicitly by how the total cost is averaged in the training loop.
        return 0.5 * squared_error_sum;
    }

    // --- Optimizer Management ---
    void set_optimizer(std::unique_ptr<Optimizer> opt) {
        if (!opt) {
            throw std::runtime_error("Cannot set a null optimizer.");
        }
        optimizer = std::move(opt);
    }
    Optimizer* get_optimizer() const { return optimizer.get(); }

    void set_learning_rate(real lr) {
        if (!optimizer) throw std::runtime_error("Optimizer not set, cannot set learning rate.");
        optimizer->set_learning_rate(lr);
    }
     real get_learning_rate() const {
         if (!optimizer) return 0.0; // Or throw
        return optimizer->get_learning_rate();
    }

    // --- Display Network Info ---
    void show() const {
         std::cout << "--- Network Structure ---" << std::endl;
         input_layer.show(); // Show InputLayer info
         int i = 1;
         for (const auto& layer : hidden_layers) {
              std::cout << "\n--- Hidden Layer " << i++ << " ---" << std::endl;
              layer.show(); // Show HiddenLayer info
         }
         std::cout << "\n--- Output Layer ---" << std::endl;
         output_layer.show(); // Show OutputLayer info
         std::cout << "\n--- Optimizer Info ---" << std::endl;
         if(optimizer) {
            std::cout << "  " << optimizer->get_info() << std::endl;
         } else {
            std::cout << "  Optimizer: Not set" << std::endl;
         }
         std::cout << "-------------------------" << std::endl;
    }

    // --- Move/Copy Semantics ---
    Network(Network&&) = default;             // Enable move construction
    Network& operator=(Network&&) = default;  // Enable move assignment
    Network(const Network&) = delete;         // Disable copy construction
    Network& operator=(const Network&) = delete; // Disable copy assignment
    virtual ~Network() = default;                // Default virtual destructor
};


// --- Fonction d'entraînement XOR (inchangée) ---
void xor_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- XOR Training Example ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    // Structure du réseau
    std::map<int, int> sizes;
    sizes[0] = 2; // Input
    sizes[1] = 8; // Hidden 1 (Augmenté un peu pour robustesse)
    sizes[2] = 4; // Hidden 2 (Ajout d'une couche pour complexité optionnelle)
    sizes[3] = 1; // Output

    // Activations: ReLU/Tanh pour cachées, Sigmoid pour sortie binaire (0/1)
    std::vector<std::string> activations = {"relu", "relu", "sigmoid"}; // H1, H2, Out

    // Paramètres d'entraînement
    real learning_rate;
    int epochs;
    int batch_size = 4; // Utiliser toutes les données (Batch GD)

    // Créer l'optimizer choisi
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.8; // Ajusté pour SGD avec cette archi
        epochs = 1500;       // Peut nécessiter plus d'époques
        optimizer = std::make_unique<SGD>(learning_rate);
    } else if (optimizer_choice == "adam") {
        learning_rate = 0.01; // Adam peut être plus rapide, LR ajusté
        epochs = 800;
        optimizer = std::make_unique<Adam>(learning_rate); // Utilise défauts b1,b2,eps
    } else {
        throw std::runtime_error("Unknown optimizer choice: " + optimizer_choice);
    }

    // Créer le réseau
    Network dnn(sizes, activations, std::move(optimizer));
    // dnn.show(); // Afficher la structure initiale

    // Données XOR
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int num_samples = 4;
    const int input_dim = sizes.at(0);
    const int output_dim = sizes.at(sizes.size()-1);
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

        // Itérer sur les samples du batch (ici, tout le dataset)
        for (int i = 0; i < num_samples; ++i) {
            // Extrait une ligne (un sample) des données d'entrée et de sortie
            auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
            auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());

            // Passe avant
            View1D prediction = dnn.forward(input_subview);

            // Calculer le coût pour ce sample (optionnel ici, on le fait sur le total)
            // real sample_cost = dnn.calculate_cost(prediction, target_subview);
            // total_epoch_cost += sample_cost; // Accumuler coût avant backprop

            // Passe arrière (calcule et accumule les gradients)
            dnn.backward(target_subview);
        }

        // Mettre à jour les poids APRÈS avoir traité tous les samples du batch
        dnn.update(batch_size);

         // Calculer le coût total de l'époque APRES la mise à jour pour refléter le nouvel état
         // (ou calculer avant la mise à jour si on veut le coût *avant* l'étape d'optimisation)
         // Ici on le recalcule pour voir l'effet de l'update.
         real current_total_cost = 0.0;
         for (int i = 0; i < num_samples; ++i) {
             auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
             auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
             View1D prediction = dnn.forward(input_subview); // Forward pass avec les poids mis à jour
             current_total_cost += dnn.calculate_cost(prediction, target_subview);
         }
         real avg_cost = current_total_cost / num_samples;


        // Afficher le coût moyen de l'époque
        if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs -1) { // Affichage ~20 fois + début/fin
            std::cout << "Epoch: " << std::setw(5) << epoch + 1
                      << ", Avg Cost: " << std::fixed << std::setprecision(8)
                      << avg_cost << std::endl;
        }

        // Condition d'arrêt simple
        if (avg_cost < 1e-4) {
             std::cout << "Convergence likely reached at epoch " << epoch + 1 << std::endl;
             break;
        }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    // Afficher la structure et les poids finaux (peut être verbeux)
    // dnn.show();

    // Tester les prédictions finales
    std::cout << "\n-- FINAL PREDICTIONS --" << std::endl;
    View1D prediction_result("prediction_result", output_dim); // Pour stocker une prédiction à la fois
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result); // Miroir host

    real final_total_cost = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());

        View1D prediction = dnn.forward(input_subview); // Utilise les poids finaux
        final_total_cost += dnn.calculate_cost(prediction, target_subview);

        Kokkos::deep_copy(prediction_result, prediction); // Copie device -> device (ou host si déjà host)
        Kokkos::deep_copy(h_prediction_result, prediction_result); // Copie device -> host
        Kokkos::fence(); // Assurer la fin des copies avant affichage

        std::cout << "Input: [" << h_xor_inputs(i,0) << "," << h_xor_inputs(i,1) << "] "
                  << "Target: " << h_xor_outputs(i,0) << " "
                  << "Prediction: " << std::fixed << std::setprecision(4) << h_prediction_result(0)
                  << " (Rounded: " << std::round(h_prediction_result(0)) << ")" << std::endl;
    }
     std::cout << "Final Average Cost: " << std::fixed << std::setprecision(8) << final_total_cost / num_samples << std::endl;

}


// --- NOUVELLE Fonction d'entraînement: Approximation de Sinus ---
// Utilise un réseau plus large pour illustrer le cas demandé
void sine_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- Sine Function Approximation Training Example ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    // Structure du réseau: 1 input (x), 2 larges couches cachées, 1 output (sin(x))
    std::map<int, int> sizes;
    sizes[0] = 1;    // Input (x value)
    sizes[1] = 64;   // Hidden 1 (Large layer)
    sizes[2] = 64;   // Hidden 2 (Large layer)
    sizes[3] = 1;    // Output (predicted sin(x))

    // Activations: ReLU ou Tanh pour cachées, Linear pour sortie (car sin(x) va de -1 à 1)
    std::vector<std::string> activations = {"relu", "relu", "linear"}; // H1, H2, Out

    // Paramètres d'entraînement
    real learning_rate;
    int epochs;
    int batch_size =  16; // Utilisation de mini-batchs
    int num_samples = 2048*4; // Nombre de points d'entraînement

    // Créer l'optimizer choisi
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.02; // SGD peut nécessiter ajustement fin pour la régression
        epochs = 500;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else if (optimizer_choice == "adam") {
        learning_rate = 0.001; // Adam est souvent bon pour ce genre de tâche
        epochs = 800;          // Peut converger assez vite
        optimizer = std::make_unique<Adam>(learning_rate);
    } else {
        throw std::runtime_error("Unknown optimizer choice: " + optimizer_choice);
    }

    // Créer le réseau
    Network dnn(sizes, activations, std::move(optimizer));
    // dnn.show(); // Optionnel: voir la structure (sera large!)

    // Générer les données d'entraînement: y = sin(x) pour x dans [-pi, pi]
    using HostView1D = Kokkos::View<real*, Kokkos::HostSpace>;
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0);
    const int output_dim = sizes.at(sizes.size()-1);

    HostView2D h_inputs("h_sine_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_sine_outputs", num_samples, output_dim);

    // Générateur de nombres aléatoires pour les x
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<real> distrib(-M_PI, M_PI);

    for(int i=0; i < num_samples; ++i) {
        real x = distrib(gen);
        h_inputs(i, 0) = x;
        h_outputs(i, 0) = std::sin(x);
    }

    // Copier les données vers l'espace d'exécution par défaut (GPU si disponible)
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << ", Samples: " << num_samples << std::endl;

    // Créer les indices pour le brassage (shuffle)
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., num_samples-1

    // Boucle d'entraînement
    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;

        // Mélanger les indices au début de chaque époque
        std::shuffle(indices.begin(), indices.end(), gen);

        // Itérer sur les mini-batchs
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - batch_start);
            if (current_batch_size <= 0) continue;

            dnn.zero_accumulated_gradients(); // Remettre à zéro pour chaque mini-batch

            // Traiter les samples dans le mini-batch courant
            for (int j = 0; j < current_batch_size; ++j) {
                int sample_index = indices[batch_start + j]; // Obtenir l'index mélangé

                // Extraire le sample correspondant
                auto input_subview = Kokkos::subview(train_inputs, sample_index, Kokkos::ALL());
                auto target_subview = Kokkos::subview(train_outputs, sample_index, Kokkos::ALL());

                // Passe avant
                View1D prediction = dnn.forward(input_subview);

                // Calculer le coût pour ce sample et l'accumuler pour l'époque
                total_epoch_cost += dnn.calculate_cost(prediction, target_subview);

                // Passe arrière (calcule et accumule les gradients pour ce sample)
                dnn.backward(target_subview);
            }

            // Mettre à jour les poids APRÈS avoir traité tous les samples du mini-batch
            dnn.update(current_batch_size);
        } // Fin boucle mini-batchs

        // Afficher le coût moyen de l'époque
        real avg_cost = total_epoch_cost / num_samples;
        if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs - 1) { // Affichage ~20 fois
             std::cout << "Epoch: " << std::setw(5) << epoch + 1
                       << ", Avg Cost: " << std::fixed << std::setprecision(8)
                       << avg_cost << std::endl;
        }

         // Condition d'arrêt possible (si la performance stagne ou est suffisante)
         if (avg_cost < 1e-3) { // Critère peut-être plus lâche pour l'approximation de fonction
             std::cout << "Good convergence likely reached at epoch " << epoch + 1 << std::endl;
             // break; // Décommenter pour arrêter tôt
         }

    } // Fin boucle epochs
    std::cout << "-- TRAINING END --" << std::endl;

    // Tester les prédictions finales sur quelques points
    std::cout << "\n-- FINAL PREDICTIONS (Sample) --" << std::endl;
    View1D prediction_result("prediction_result_sine", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);

    real final_total_cost = 0.0;
    int num_test_samples = std::min(num_samples, 10); // Tester sur les 10 premiers points générés (ou moins)
    for (int i = 0; i < num_test_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL()); // Utilise les données originales non mélangées ici
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());

        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);

        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result);
        Kokkos::fence();

         // Récupérer la valeur x correspondante depuis le host mirror des inputs
         real input_x = h_inputs(i, 0);
         real target_y = h_outputs(i, 0);

        std::cout << "Input x: " << std::fixed << std::setprecision(4) << input_x << " "
                  << "Target sin(x): " << std::fixed << std::setprecision(4) << target_y << " "
                  << "Prediction: " << std::fixed << std::setprecision(4) << h_prediction_result(0)
                  << std::endl;
    }
    // Note: le coût final ici est calculé seulement sur les `num_test_samples` points affichés
    std::cout << "Final Average Cost (on first " << num_test_samples << " samples): "
              << std::fixed << std::setprecision(8) << final_total_cost / num_test_samples << std::endl;
}


// --- AUTRE Fonction d'entraînement: Séparation Linéaire Simple ---
// Utilise un réseau très simple, juste pour contraster
void linear_sep_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- Linear Separation Training Example ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    // Structure du réseau: 2 inputs (x,y), PAS de couche cachée, 1 output (classe 0 ou 1)
    // Un seul neurone avec Sigmoid suffit pour la séparation linéaire.
    std::map<int, int> sizes;
    sizes[0] = 2;    // Input (x, y coordinates)
    sizes[1] = 1;    // Output (class label 0 or 1)

    // Activation: Sigmoid pour la sortie binaire
    std::vector<std::string> activations = {"sigmoid"}; // Juste pour la couche de sortie

    // Paramètres d'entraînement
    real learning_rate;
    int epochs;
    int batch_size = 16;
    int num_samples = 2560;

    // Créer l'optimizer choisi
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.1;
        epochs = 100;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else if (optimizer_choice == "adam") {
        learning_rate = 0.01;
        epochs = 150;
        optimizer = std::make_unique<Adam>(learning_rate);
    } else {
        throw std::runtime_error("Unknown optimizer choice: " + optimizer_choice);
    }

    // Créer le réseau
    Network dnn(sizes, activations, std::move(optimizer));
    // dnn.show();

    // Générer les données d'entraînement: Points en 2D séparés par la ligne y = x
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0);
    const int output_dim = sizes.at(sizes.size()-1);

    HostView2D h_inputs("h_linear_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_linear_outputs", num_samples, output_dim);

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + 123); // Seed différent
    std::uniform_real_distribution<real> distrib(-1.0, 1.0); // Points dans [-1, 1] x [-1, 1]
    real margin = 0.1; // Marge autour de la ligne de séparation y=x

    int count_class0 = 0;
    int count_class1 = 0;
    for(int i=0; i < num_samples; ++i) {
        real x = distrib(gen);
        real y = distrib(gen);
        h_inputs(i, 0) = x;
        h_inputs(i, 1) = y;
        // Classe 0 si y < x - margin, Classe 1 si y > x + margin
        // On ignore les points dans la marge pour une séparation plus claire
        if (y < x - margin) {
            h_outputs(i, 0) = 0.0;
            count_class0++;
        } else if (y > x + margin) {
            h_outputs(i, 0) = 1.0;
            count_class1++;
        } else {
            // Point dans la marge, on le regénère pour éviter l'ambiguïté
            i--;
            continue;
        }
    }
    std::cout << "Generated " << count_class0 << " Class 0 samples and " << count_class1 << " Class 1 samples." << std::endl;


    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << ", Samples: " << num_samples << std::endl;

    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        std::shuffle(indices.begin(), indices.end(), gen);

        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - batch_start);
             if (current_batch_size <= 0) continue;
            dnn.zero_accumulated_gradients();

            for (int j = 0; j < current_batch_size; ++j) {
                int sample_index = indices[batch_start + j];
                auto input_subview = Kokkos::subview(train_inputs, sample_index, Kokkos::ALL());
                auto target_subview = Kokkos::subview(train_outputs, sample_index, Kokkos::ALL());
                View1D prediction = dnn.forward(input_subview);
                total_epoch_cost += dnn.calculate_cost(prediction, target_subview);
                dnn.backward(target_subview);
            }
            dnn.update(current_batch_size);
        }

        real avg_cost = total_epoch_cost / num_samples;
        if ((epoch + 1) % (epochs / 10) == 0 || epoch == 0 || epoch == epochs - 1) { // Affichage ~10 fois
             std::cout << "Epoch: " << std::setw(4) << epoch + 1
                       << ", Avg Cost: " << std::fixed << std::setprecision(8)
                       << avg_cost << std::endl;
        }
         if (avg_cost < 1e-3) {
             std::cout << "Convergence likely reached at epoch " << epoch + 1 << std::endl;
             break;
         }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    // Tester les prédictions finales et calculer l'accuracy
    std::cout << "\n-- FINAL PREDICTIONS & ACCURACY --" << std::endl;
    View1D prediction_result("prediction_result_linear", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0;
    int correct_predictions = 0;

    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());

        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);

        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result);
        Kokkos::fence();

        // Comparer la prédiction arrondie à la cible
        real predicted_value = h_prediction_result(0);
        int predicted_class = std::round(predicted_value);
        int target_class = static_cast<int>(h_outputs(i, 0)); // Cible est déjà 0 ou 1

        if (predicted_class == target_class) {
            correct_predictions++;
        }
         // Afficher quelques exemples
         if (i < 10) {
              std::cout << "Input: [" << std::fixed << std::setprecision(2) << h_inputs(i,0) << "," << h_inputs(i,1) << "] "
                        << "Target: " << target_class << " "
                        << "Pred: " << std::fixed << std::setprecision(3) << predicted_value
                        << " (Rounded: " << predicted_class << ")"
                        << (predicted_class == target_class ? "" : " <-- WRONG") << std::endl;
         }
    }
    real accuracy = static_cast<real>(correct_predictions) / num_samples;
    std::cout << "Final Average Cost: " << std::fixed << std::setprecision(8) << final_total_cost / num_samples << std::endl;
    std::cout << "Final Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%" << std::endl;
}


// --- Main ---
// Ajout des appels aux nouvelles fonctions
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    { // Scope for Kokkos objects
        try {
            std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;

            // Lancer l'entraînement XOR avec Adam par défaut
            xor_train("adam");
            std::cout << "\n---------------------------\n" << std::endl;
             // Lancer l'entraînement XOR avec SGD pour comparaison
             // xor_train("sgd"); // Peut être lent ou nécessiter ajustement

             // --- NOUVEAU: Lancer l'entraînement d'approximation de Sinus ---
             std::cout << "\n---------------------------\n" << std::endl;
//             sine_train("adam"); // Utilise un réseau plus large

             // --- NOUVEAU: Lancer l'entraînement de séparation linéaire ---
             std::cout << "\n---------------------------\n" << std::endl;
             linear_sep_train("adam"); // Utilise un réseau très simple


        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
             // Ensure Kokkos is finalized even on error
             // No explicit finalize needed here due to RAII if Kokkos::ScopeGuard is used
             // But if not using ScopeGuard, finalize here:
             // Kokkos::finalize(); // Uncomment if not using Kokkos::ScopeGuard/RAII for finalize
            return 1;
        } catch (...) {
            std::cerr << "An unknown error occurred." << std::endl;
             // Kokkos::finalize(); // Uncomment if necessary
             return 1;
        }

    } // Kokkos objects go out of scope, Kokkos::finalize() called automatically if using RAII ScopeGuard
    Kokkos::finalize(); // Call finalize explicitly if not using RAII Kokkos::ScopeGuard
    return 0;
} 
