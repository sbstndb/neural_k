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
#include <numeric>
#include <iomanip>
#include <typeinfo>
#include <chrono>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosKernels_Handle.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Typedefs ---
using real = float;
using View1D = Kokkos::View<real*>;
// using View2D = Kokkos::View<real**>; // Moins utilisé maintenant pour les poids

// *** TYPEDEFS POUR SPARSE MATRIX ***
using Scalar = real;
using Ordinal = int;
using Offset = size_t;
using Device = Kokkos::DefaultExecutionSpace;
using Layout = Kokkos::LayoutLeft;
using SparseMatrixType = KokkosSparse::CrsMatrix<Scalar, Ordinal, Device, void, Offset>;
using GraphType = typename SparseMatrixType::staticcrsgraph_type;
using ValuesType = typename SparseMatrixType::values_type;
using RowMapType = typename GraphType::row_map_type::non_const_type;
using EntriesType = typename GraphType::entries_type::non_const_type;

class Optimizer;
class SGD;
class Adam;
class Layer;
class Activation;
class Dataset;
class Network;
class BatchHandler;

class Optimizer {
public:
    real learning_rate;

    Optimizer(real lr) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    virtual void update(SparseMatrixType& weights, View1D biases,
                        const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
                        int batch_size) = 0;

    void set_learning_rate(real lr) { learning_rate = lr; }
    real get_learning_rate() const { return learning_rate; }

    virtual std::string get_info() const {
        return "Optimizer(LR=" + std::to_string(learning_rate) + ")";
    }
};

class SGD : public Optimizer {
public:
    SGD(real lr = 0.1) : Optimizer(lr) {}

    void update(SparseMatrixType& weights, View1D biases,
                const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override {

        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for SGD update.");
        }
        if (weights.numRows() == 0 && biases.extent(0) == 0) {
             return;
        }
        if (weights.numRows() > 0 && weights.numRows() != biases.extent(0)) {
            throw std::runtime_error("SGD update: Mismatch between weights rows and biases size.");
        }
        if (biases.extent_int(0) <= 0 && weights.numRows() <= 0) {
            if (biases.extent(0) == 0 && weights.numRows() == 0) return;
             throw std::runtime_error("SGD update called on layer with invalid dimensions.");
        }

        const real scale = learning_rate / static_cast<real>(batch_size);
        const int layer_size = biases.extent_int(0);

        auto w_vals = weights.values;
        auto dw_vals = accumulated_d_weights.values;
        const int nnz = w_vals.extent_int(0);

        if (nnz != dw_vals.extent_int(0)) {
             throw std::runtime_error("SGD update: Mismatch in number of non-zeros between weights and gradients.");
        }

        Kokkos::parallel_for("sgd_update_weights_sparse", nnz, KOKKOS_LAMBDA(const int k) {
            w_vals(k) -= scale * dw_vals(k);
        });

        Kokkos::parallel_for("sgd_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
            biases(i) -= scale * accumulated_d_biases(i);
        });
    }

    std::string get_info() const override {
        return "SGD(LR=" + std::to_string(learning_rate) + ")";
    }
};

class Adam : public Optimizer {
public:
    real beta1;
    real beta2;
    real epsilon;

private:
    // Structure to hold state (m, v, t)
    struct ParameterState {
        View1D m_1d;
        View1D v_1d;
        Kokkos::View<real**> m_2d;
        Kokkos::View<real**> v_2d;
        long long t = 0;

        ParameterState(size_t size) : t(0) {
            if (size > std::numeric_limits<int>::max()) {
                 throw std::runtime_error("Adam state bias size exceeds limits.");
            }
            m_1d = View1D("adam_m1d", size);
            v_1d = View1D("adam_v1d", size);
            Kokkos::deep_copy(m_1d, 0.0);
            Kokkos::deep_copy(v_1d, 0.0);
        }
        // Constructor for weights state (keeps dense moments)
        ParameterState(size_t rows, size_t cols) : t(0) {
             if (rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max()) {
                 throw std::runtime_error("Adam state weight dimensions exceed limits.");
            }
            // Allocate dense views for moments, matching the *logical* dense dimensions
            m_2d = Kokkos::View<real**>("adam_m2d", rows, cols);
            v_2d = Kokkos::View<real**>("adam_v2d", rows, cols);
            Kokkos::deep_copy(m_2d, 0.0);
            Kokkos::deep_copy(v_2d, 0.0);
        }
         ParameterState() = default;
         ParameterState(ParameterState&&) = default;
         ParameterState& operator=(ParameterState&&) = default;
         ParameterState(const ParameterState&) = delete;
         ParameterState& operator=(const ParameterState&) = delete;
    };

    // Maps to store state: Use data pointer of the *values* array for sparse matrices.
    // For biases, use the bias View1D data pointer.
    // Assumes these pointers remain stable.
    std::map<real*, ParameterState> state_map_1d; // For biases
    std::map<real*, ParameterState> state_map_sparse; // For sparse weights (key is weights.values.data())

public:
    Adam(real lr = 0.001, real b1 = 0.9, real b2 = 0.999, real eps = 1e-8)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps) {
        if (lr <= 0 || b1 < 0 || b1 >= 1 || b2 < 0 || b2 >= 1 || eps <= 0) {
            throw std::runtime_error("Invalid Adam hyperparameters.");
        }
    }

    // MODIFIED: Takes SparseMatrixType
    void update(SparseMatrixType& weights, View1D biases,
                const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override {

        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for Adam update.");
        }
        if (weights.numRows() == 0 && biases.extent(0) == 0) {
            return; // Nothing to update
        }
        if (weights.numRows() > 0 && weights.numRows() != biases.extent(0)) {
            throw std::runtime_error("Adam update: Mismatch between weights rows and biases size.");
        }
        if (biases.extent_int(0) <= 0 && weights.numRows() <= 0) {
             if (biases.extent(0) == 0 && weights.numRows() == 0) return; // Allow empty update
             throw std::runtime_error("Adam update called on layer with invalid dimensions.");
        }

        const int layer_size = biases.extent_int(0); // numRows
        const int input_size = weights.numCols();    // numCols
        const real scale = 1.0 / static_cast<real>(batch_size);

        // --- Update Weights (Sparse Weights, Dense Moments) ---
        {
            // Use weights.values.data() as the key for the state map
            real* weight_values_ptr = weights.values.data();
            auto it_w = state_map_sparse.find(weight_values_ptr);
            if (it_w == state_map_sparse.end()) {
                // Create state if it doesn't exist, using logical dense dimensions
                auto result = state_map_sparse.try_emplace(weight_values_ptr, weights.numRows(), weights.numCols());
                 if (!result.second) {
                     throw std::runtime_error("Failed to insert Adam state for sparse weights.");
                 }
                 it_w = result.first;
                 //std::cout << "Adam: Initialized state for weights (dense moments) at " << weight_values_ptr << std::endl;
            }
            ParameterState& state_w = it_w->second;
            state_w.t++; // Increment timestep

            // Precompute bias correction terms
            const double beta1_pow_t = std::pow(static_cast<double>(beta1), static_cast<double>(state_w.t));
            const double beta2_pow_t = std::pow(static_cast<double>(beta2), static_cast<double>(state_w.t));
            const real bias_correction1 = 1.0f / (1.0f - static_cast<real>(beta1_pow_t));
            const real bias_correction2 = 1.0f / (1.0f - static_cast<real>(beta2_pow_t));

            // Capture necessary variables
            real lr = learning_rate;
            real b1 = beta1;
            real b2 = beta2;
            real eps = epsilon;
            auto m = state_w.m_2d; // Dense moment views
            auto v = state_w.v_2d;
            auto w_vals = weights.values;           // Sparse weight values
            auto dw_vals = accumulated_d_weights.values; // Sparse gradient values
            auto graph = weights.graph; // Graph structure (row_map, entries)

            // Iterate row by row over the sparse matrix structure
            Kokkos::parallel_for("adam_update_weights_sparse", layer_size, KOKKOS_LAMBDA (const int i) {
                const auto row_start = graph.row_map(i);
                const auto row_end = graph.row_map(i+1);
                for (auto k = row_start; k < row_end; ++k) {
                    const int j = graph.entries(k); // Get column index

                    // 1. Calculate average gradient for this non-zero entry
                    real grad = scale * dw_vals(k);

                    // 2. Update biased moments (using dense moments at (i, j))
                    m(i, j) = b1 * m(i, j) + (1.0f - b1) * grad;
                    v(i, j) = b2 * v(i, j) + (1.0f - b2) * grad * grad;

                    // 3. Compute bias-corrected moments
                    real m_hat = m(i, j) * bias_correction1;
                    real v_hat = v(i, j) * bias_correction2;

                    // 4. Update sparse weight value
                    w_vals(k) -= lr * m_hat / (Kokkos::sqrt(v_hat) + eps);
                }
            });
        } // End scope for weights update

        // --- Update Biases (Dense - unchanged) ---
        {
             auto it_b = state_map_1d.find(biases.data());
             if (it_b == state_map_1d.end()) {
                 auto result = state_map_1d.try_emplace(biases.data(), biases.extent(0));
                 if (!result.second) {
                     throw std::runtime_error("Failed to insert Adam state for biases.");
                 }
                 it_b = result.first;
             }
             ParameterState& state_b = it_b->second;
             state_b.t++;

            const double beta1_pow_t = std::pow(static_cast<double>(beta1), static_cast<double>(state_b.t));
            const double beta2_pow_t = std::pow(static_cast<double>(beta2), static_cast<double>(state_b.t));
            const real bias_correction1 = 1.0f / (1.0f - static_cast<real>(beta1_pow_t));
            const real bias_correction2 = 1.0f / (1.0f - static_cast<real>(beta2_pow_t));

            real lr = learning_rate;
            real b1 = beta1;
            real b2 = beta2;
            real eps = epsilon;
            auto m = state_b.m_1d;
            auto v = state_b.v_1d;

            Kokkos::parallel_for("adam_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
                real grad = scale * accumulated_d_biases(i);
                m(i) = b1 * m(i) + (1.0f - b1) * grad;
                v(i) = b2 * v(i) + (1.0f - b2) * grad * grad;
                real m_hat = m(i) * bias_correction1;
                real v_hat = v(i) * bias_correction2;
                biases(i) -= lr * m_hat / (Kokkos::sqrt(v_hat) + eps);
            });
        } // End scope for biases update
    } // End update method

     std::string get_info() const override {
         return "Adam(LR=" + std::to_string(learning_rate) +
                ", beta1=" + std::to_string(beta1) +
                ", beta2=" + std::to_string(beta2) +
                ", epsilon=" + std::to_string(epsilon) + ")";
     }

     virtual ~Adam() override = default;
};

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
        Kokkos::deep_copy(output, input);
    }
    void apply_derivative(const View1D& /*input_z*/, View1D& output_deriv) const override {
        Kokkos::deep_copy(output_deriv, 1.0f);
    }
};

class RELU : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent_int(0);
        Kokkos::parallel_for("relu_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = (input(i) > 0.0f) ? input(i) : 0.0f;
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent_int(0);
        Kokkos::parallel_for("relu_deriv", size, KOKKOS_LAMBDA(const int i) {
            output_deriv(i) = (input_z(i) > 0.0f) ? 1.0f : 0.0f;
        });
    }
};

class SIGMOID : public Activation {
public:
    KOKKOS_INLINE_FUNCTION real scalar_sigmoid(real x) const {
        x = Kokkos::max(-30.0f, Kokkos::min(30.0f, x));
        return 1.0f / (1.0f + Kokkos::exp(-x));
    }
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent_int(0);
        Kokkos::parallel_for("sigmoid_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = scalar_sigmoid(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent_int(0);
        Kokkos::parallel_for("sigmoid_deriv", size, KOKKOS_LAMBDA(const int i) {
            real s = scalar_sigmoid(input_z(i));
            output_deriv(i) = s * (1.0f - s);
        });
    }
};

class TANH : public Activation {
public:
     KOKKOS_INLINE_FUNCTION real scalar_tanh(real x) const {
        return Kokkos::tanh(x);
    }
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent_int(0);
        Kokkos::parallel_for("tanh_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = scalar_tanh(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent_int(0);
        Kokkos::parallel_for("tanh_deriv", size, KOKKOS_LAMBDA(const int i) {
            real t = scalar_tanh(input_z(i));
            output_deriv(i) = 1.0f - t * t;
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

// --- Classe Layer (MODIFIED for Sparse Weights) ---
class Layer {
public:
    int input_size; // Can be 0 for InputLayer
    int layer_size; // Number of neurons in this layer (rows in weights matrix)
    std::unique_ptr<Activation> activation;

    // Kokkos views for parameters and states
    // *** MODIFIED: Weights and their gradients are now sparse ***
    SparseMatrixType weights; // [layer_size x input_size] (Sparse)
    View1D biases;            // [layer_size] (Dense)
    View1D z;                 // [layer_size] - Output before activation (Dense)
    View1D a;                 // [layer_size] - Output after activation (Dense)

    // Kokkos views for backpropagation (gradients)
    View1D delta;             // [layer_size] - Propagated error (δ) (Dense)
    View1D d_biases;          // [layer_size] - Instantaneous bias gradient (Dense)
    // *** MODIFIED: Sparse weight gradients ***
    SparseMatrixType d_weights;      // [layer_size x input_size] - Instantaneous weight gradient (Sparse)
    View1D d_biases_sum;      // [layer_size] - Sum of bias gradients (Dense)
    // *** MODIFIED: Sparse accumulated weight gradients ***
    SparseMatrixType d_weights_sum;  // [layer_size x input_size] - Sum of weight gradients (Sparse)

    View1D tmp_deriv;         // [layer_size] - Temporary storage for activation derivative (Dense)


    // Constructor for hidden/output layers (MODIFIED for sparse init)
    Layer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        input_size(_input_size),
        layer_size(_layer_size),
        activation(std::move(act_func))
    {
        if (input_size < 0 || layer_size <= 0) {
             throw std::runtime_error("Layer sizes must be non-negative, layer_size > 0.");
        }

        // Allocate dense views first (biases, z, a, delta, etc.)
        biases = View1D("biases", layer_size);
        z = View1D("z", layer_size);
        delta = View1D("delta", layer_size);
        d_biases = View1D("d_biases", layer_size);
        d_biases_sum = View1D("d_biases_sum", layer_size);
        tmp_deriv = View1D("tmp_deriv", layer_size);

        Kokkos::deep_copy(biases, 0.0); // Initialize biases to 0
        Kokkos::deep_copy(z, 0.0);
        Kokkos::deep_copy(delta, 0.0);
        Kokkos::deep_copy(d_biases, 0.0);
        Kokkos::deep_copy(d_biases_sum, 0.0);
        Kokkos::deep_copy(tmp_deriv, 0.0);

        // 'a' is always needed (output activation)
        a = View1D("a", layer_size);
        Kokkos::deep_copy(a, 0.0);

        // *** Initialize Sparse Matrices (weights, d_weights, d_weights_sum) ***
        // Only if input_size > 0 (otherwise, matrix is empty)
        if (input_size > 0) {
            // For this conversion, we create a *dense graph* structure.
            // All possible connections exist.
            size_t nnz = static_cast<size_t>(layer_size) * input_size;

            // Allocate graph structure (row_map, entries) and values
            RowMapType row_map("row_map", layer_size + 1);
            EntriesType entries("entries", nnz);
            ValuesType w_values("w_values", nnz);
            ValuesType dw_values("dw_values", nnz);
            ValuesType dw_sum_values("dw_sum_values", nnz);

            // --- Create the dense graph structure ---
            Kokkos::parallel_for("create_dense_graph", layer_size, KOKKOS_LAMBDA(const int i) {
                row_map(i) = static_cast<Offset>(i) * input_size; // Start of row i
                // Fill entries for row i
                for (int j = 0; j < input_size; ++j) {
                    Ordinal k = static_cast<Ordinal>(i) * input_size + j; // Linear index
                    entries(k) = j; // Column index for this entry is j
                }
                // Handle last entry for row_map size
                if (i == layer_size - 1) {
                     row_map(layer_size) = nnz; // Total number of non-zeros
                }
            });
            // Ensure graph creation is finished before using it
             Kokkos::fence();

            // --- Initialize Weight Values (Xavier/Glorot) ---
            Kokkos::Random_XorShift64_Pool<> rand_pool(std::chrono::high_resolution_clock::now().time_since_epoch().count() + reinterpret_cast<uintptr_t>(this));
            real limit = (input_size + layer_size > 0) ? std::sqrt(6.0f / (input_size + layer_size)) : 1.0f;
            // Fill the values array directly
            Kokkos::fill_random(w_values, rand_pool, static_cast<real>(-limit), static_cast<real>(limit));

            // --- Initialize Gradient Values to Zero ---
            Kokkos::deep_copy(dw_values, 0.0);
            Kokkos::deep_copy(dw_sum_values, 0.0);

            // --- Create the CrsMatrix objects ---
            // Construct the graph object first
            GraphType graph(entries, row_map);

            weights = SparseMatrixType("weights", input_size, w_values, graph);
            d_weights = SparseMatrixType("d_weights", input_size, dw_values, graph); // Share graph structure initially
            d_weights_sum = SparseMatrixType("d_weights_sum", input_size,  dw_sum_values, graph); // Share graph structure initially

        } else {
            // If input_size is 0, create empty matrices
            // Need valid but empty graph/values views if Kokkos requires non-null pointers
             RowMapType row_map("row_map_empty", layer_size + 1); // Still need row map for numRows
             EntriesType entries("entries_empty", 0);
             ValuesType w_values("w_values_empty", 0);
             ValuesType dw_values("dw_values_empty", 0);
             ValuesType dw_sum_values("dw_sum_values_empty", 0);
             Kokkos::deep_copy(row_map, 0); // Initialize row map to zeros

             GraphType graph(entries, row_map);
             weights = SparseMatrixType("weights_empty",  0,  w_values, graph);
             d_weights = SparseMatrixType("d_weights_empty",  0,  dw_values, graph);
             d_weights_sum = SparseMatrixType("d_weights_sum_empty",  0, dw_sum_values, graph);
        }
    }

    // Constructor specifically for InputLayer (sets input_size to 0)
    Layer(int _layer_size) :
        Layer(0, _layer_size, nullptr) // Delegate to the main constructor
    {
         // 'a' is allocated by delegated constructor. Sparse matrices are empty.
    }

    // --- Methods ---
    virtual void forward(const View1D& prev_layer_a) {
        if (input_size == 0) {
             return; // Input layer logic, 'a' set externally
         }
         if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("Forward pass: prev_layer_a size (" + std::to_string(prev_layer_a.extent(0))
                                     + ") mismatch with layer input_size (" + std::to_string(input_size) + ")");
         }

        // 1. z = W * prev_layer_a + b
        if (weights.numRows() != static_cast<size_t>(layer_size) || weights.numCols() != static_cast<size_t>(input_size) ||
            z.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Dimension mismatch before SpMV in forward pass.");
        }

        // *** MODIFIED: Use Sparse Matrix-Vector Multiply (SpMV) ***
        // z = 1.0 * weights * prev_layer_a + 0.0 * z (overwrite z)
        KokkosSparse::spmv("N", 1.0, weights, prev_layer_a, 0.0, z);

        // Add biases (dense vector operation - unchanged)
        if (biases.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Bias dimension mismatch in forward pass.");
        }
        Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) {
            z(i) += biases(i);
        });

        // 2. a = activation(z) (unchanged)
        if (a.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Activation output 'a' dimension mismatch in forward pass.");
        }
        if (activation) {
            activation->apply(z, a);
        } else {
            Kokkos::deep_copy(a, z);
        }
    }

    // Computes gradients for a hidden layer (MODIFIED for sparse weights)
    virtual void compute_gradients(const Layer& next_layer, const View1D& prev_layer_a) {
         if (input_size == 0) return; // InputLayer has no gradients

         // Check dimensions using next_layer.weights properties
         if (next_layer.weights.numCols() != static_cast<size_t>(layer_size) || // next W cols must match current layer size
             next_layer.delta.extent(0) != next_layer.weights.numRows()) {     // next delta size must match next layer rows
             throw std::runtime_error("Dimension mismatch between current layer and next layer for gradient computation.");
         }
          if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("Dimension mismatch for prev_layer_a in gradient computation.");
         }
          if (delta.extent(0) != static_cast<size_t>(layer_size) ||
              tmp_deriv.extent(0) != static_cast<size_t>(layer_size)) {
              throw std::runtime_error("Internal dimension mismatch for delta or tmp_deriv.");
          }

        // 1. Calculate f'(z) -> stored in tmp_deriv (unchanged)
        if (activation) {
             activation->apply_derivative(z, tmp_deriv);
        } else {
            Kokkos::deep_copy(tmp_deriv, 1.0f);
        }

        // 2. Calculate delta for this layer: δ_l = (W_{l+1}^T * δ_{l+1}) .* f'(z_l)
        // *** MODIFIED: Use SpMV with Transpose ***
        View1D delta_prop("delta_prop", layer_size); // Must match current layer size
        // delta_prop = 1.0 * next_layer.weights^T * next_layer.delta + 0.0 * delta_prop
        KokkosSparse::spmv("T", 1.0, next_layer.weights, next_layer.delta, 0.0, delta_prop);

        // Element-wise product (Hadamard product) (unchanged)
         Kokkos::parallel_for("hadamard_delta_deriv", layer_size, KOKKOS_LAMBDA(int i) {
            delta(i) = delta_prop(i) * tmp_deriv(i);
        });

        // 3. Calculate INSTANTANEOUS gradients (store in d_weights.values, d_biases)
        // d_biases = delta (unchanged)
        if (d_biases.extent(0) != delta.extent(0)) throw std::runtime_error("d_biases/delta size mismatch.");
        Kokkos::deep_copy(d_biases, delta);

        // *** MODIFIED: d_weights = delta * prev_layer_a^T (outer product, store in sparse values) ***
        // We iterate through the non-zero structure defined in d_weights.graph
        auto dw_vals = d_weights.values;
        auto graph = d_weights.graph; // Assumes d_weights has the same graph structure as weights
        if (dw_vals.extent(0) != weights.nnz()) { // Sanity check
             throw std::runtime_error("d_weights nnz mismatch.");
        }
        // Iterate over rows and then non-zeros within each row
        Kokkos::parallel_for("compute_d_weights_sparse", layer_size, KOKKOS_LAMBDA(const int i) {
             const auto row_start = graph.row_map(i);
             const auto row_end = graph.row_map(i+1);
             for (auto k = row_start; k < row_end; ++k) {
                 const int j = graph.entries(k); // Get column index
                 // Check bounds (redundant if graph is correct, but safe)
                 // if (i < delta.extent_int(0) && j < prev_layer_a.extent_int(0)) {
                      dw_vals(k) = delta(i) * prev_layer_a(j);
                 // } else {
                 //     // Should not happen with the dense graph structure
                 // }
             }
         });

        // 4. ACCUMULATE instantaneous gradients into sum views
        // *** MODIFIED: Accumulate into d_weights_sum.values ***
        auto dw_sum_vals = d_weights_sum.values;
        if (dw_sum_vals.extent(0) != dw_vals.extent(0)) {
            throw std::runtime_error("Accumulated d_weights nnz mismatch.");
        }
        const int nnz = dw_vals.extent_int(0);
        Kokkos::parallel_for("accumulate_weight_gradients_sparse", nnz, KOKKOS_LAMBDA (const int k) {
                 Kokkos::atomic_add(&dw_sum_vals(k), dw_vals(k));
         });

         // Accumulate bias gradients (unchanged)
          if (d_biases_sum.extent(0) != d_biases.extent(0)) {
             throw std::runtime_error("Accumulated d_biases dimension mismatch.");
         }
         Kokkos::parallel_for("accumulate_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
                 Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
         });
    }

    // Zero out the gradient sums (MODIFIED for sparse weights)
    void zero_accumulated_gradients() {
         if (input_size == 0 && layer_size == 0) return; // Skip if completely empty layer

         // Zero dense bias gradients
         if (d_biases_sum.data() != nullptr) Kokkos::deep_copy(d_biases_sum, 0.0);

         // *** MODIFIED: Zero only the values array of the sparse weight gradients ***
         if (d_weights_sum.values.data() != nullptr && d_weights_sum.nnz() > 0) {
              Kokkos::deep_copy(d_weights_sum.values, 0.0);
         }
    }

    // Display layer info (MODIFIED for sparse weights)
    void show() const {
        if (input_size == 0 && activation == nullptr) { // Signature of InputLayer
            std::cout << "Input Layer (size " << layer_size << ")" << std::endl;
            return;
        }
        std::cout << "Layer (" << input_size << " -> " << layer_size << "):";
        if (activation) {
             try {
                 std::string act_name = typeid(*activation).name();
                 size_t last_colon = act_name.find_last_of("::");
                 if(last_colon != std::string::npos) act_name = act_name.substr(last_colon+1);
                 std::cout << " Activation: " << act_name;
             } catch (...) { std::cout << " Activation: [Unknown]"; }
         } else {
             std::cout << " Activation: Linear (Implicit)";
         }
         std::cout << std::endl;

        // --- Display Sparse Weights Info ---
        if (weights.values.data() != nullptr && weights.numRows() > 0 && weights.numCols() > 0) {
            std::cout << "  Weights (Sparse): " << weights.numRows() << "x" << weights.numCols()
                      << ", NNZ: " << weights.nnz() << std::endl;

            // Avoid printing huge value arrays
            const int max_print_vals = 10;
            auto h_values = Kokkos::create_mirror_view(weights.values);
            Kokkos::deep_copy(h_values, weights.values);
            // Optionally copy graph structure too if needed for display
            // auto h_rowmap = Kokkos::create_mirror_view(weights.graph.row_map);
            // Kokkos::deep_copy(h_rowmap, weights.graph.row_map);
            // auto h_entries = Kokkos::create_mirror_view(weights.graph.entries);
            // Kokkos::deep_copy(h_entries, weights.graph.entries);
            Kokkos::fence();

            std::cout << "    Values sample: [";
            for(int k=0; k < std::min((int)weights.nnz(), max_print_vals); ++k) {
                std::cout << std::fixed << std::setprecision(3) << h_values(k) << " ";
            }
            if (weights.nnz() > max_print_vals) std::cout << "...";
            std::cout << "]" << std::endl;

        } else {
             std::cout << "  Weights: N/A (Input Layer or uninitialized)" << std::endl;
        }

        // --- Display Biases Info (Unchanged) ---
        if (biases.data() != nullptr && biases.extent(0) > 0) {
            auto h_biases = Kokkos::create_mirror_view(biases);
            Kokkos::deep_copy(h_biases, biases);
            Kokkos::fence();

            const int max_print_biases = 10;
            std::cout << "  Biases (" << biases.extent(0) << ", sample): [";
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
// --- Classe InputLayer (inchangée, hérite de Layer modifié) ---
class InputLayer : public Layer {
public:
    InputLayer(int _layer_size) : Layer(_layer_size) {}

    void set_input(const View1D& input_data) {
         if (input_data.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Input data size mismatch for InputLayer: Expected "
                + std::to_string(layer_size) + ", Got " + std::to_string(input_data.extent(0)));
         }
         if (a.data() == nullptr || a.extent(0) != static_cast<size_t>(layer_size)) {
              throw std::runtime_error("InputLayer 'a' view is not correctly initialized or sized.");
         }
         Kokkos::deep_copy(a, input_data);
     }

     void forward(const View1D& /*prev_layer_a*/) override { }
     void compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) override { }
};


// --- Classe OutputLayer (hérite de Layer modifié) ---
class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        Layer(_input_size, _layer_size, std::move(act_func)) {}

    // Special gradient computation for the output layer (MODIFIED for sparse weights)
    void compute_gradients(const View1D& target, const View1D& prev_layer_a) {
        // Check dimensions
         if (target.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Target size mismatch for OutputLayer.");
         }
         if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("prev_layer_a size mismatch in OutputLayer gradient.");
         }
         if (delta.extent(0) != static_cast<size_t>(layer_size) ||
              a.extent(0) != static_cast<size_t>(layer_size) ||
              tmp_deriv.extent(0) != static_cast<size_t>(layer_size)) {
              throw std::runtime_error("Internal dimension mismatch in OutputLayer gradient.");
          }

        // 1. Calculate f'(z_L) -> stored in tmp_deriv (unchanged)
        if (activation) {
            activation->apply_derivative(z, tmp_deriv);
        } else {
            Kokkos::deep_copy(tmp_deriv, 1.0f);
        }

        // 2. Calculate delta for the output layer: δ_L = (a_L - y) .* f'(z_L) (unchanged)
        Kokkos::parallel_for("compute_output_delta", layer_size, KOKKOS_LAMBDA(int i) {
            delta(i) = (a(i) - target(i)) * tmp_deriv(i);
        });

       // 3. Calculate INSTANTANEOUS gradients (∂Cost/∂b_L = δ_L, ∂Cost/∂W_L = δ_L * a_{L-1}^T)
        // d_biases = delta (unchanged)
        if (d_biases.extent(0) != delta.extent(0)) throw std::runtime_error("OutputLayer d_biases/delta size mismatch.");
        Kokkos::deep_copy(d_biases, delta);

        // *** MODIFIED: d_weights = delta * prev_layer_a^T (outer product, store in sparse values) ***
        auto dw_vals = d_weights.values;
        auto graph = d_weights.graph;
         if (dw_vals.extent(0) != weights.nnz()) {
             throw std::runtime_error("OutputLayer d_weights nnz mismatch.");
         }
         Kokkos::parallel_for("compute_output_d_weights_sparse", layer_size, KOKKOS_LAMBDA(const int i) {
             const auto row_start = graph.row_map(i);
             const auto row_end = graph.row_map(i+1);
             for (auto k = row_start; k < row_end; ++k) {
                 const int j = graph.entries(k);
                 // if (i < delta.extent_int(0) && j < prev_layer_a.extent_int(0)) {
                     dw_vals(k) = delta(i) * prev_layer_a(j);
                 // }
             }
        });

        // 4. ACCUMULATE gradients into sum views
        // *** MODIFIED: Accumulate sparse weight gradients ***
        auto dw_sum_vals = d_weights_sum.values;
        if (dw_sum_vals.extent(0) != dw_vals.extent(0)) {
             throw std::runtime_error("OutputLayer accumulated d_weights nnz mismatch.");
         }
        const int nnz = dw_vals.extent_int(0);
        Kokkos::parallel_for("accumulate_output_weight_gradients_sparse", nnz, KOKKOS_LAMBDA (const int k) {
            Kokkos::atomic_add(&dw_sum_vals(k), dw_vals(k));
        });

        // Accumulate bias gradients (unchanged)
         if (d_biases_sum.extent(0) != d_biases.extent(0)) {
             throw std::runtime_error("OutputLayer accumulated d_biases dimension mismatch.");
         }
         Kokkos::parallel_for("accumulate_output_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
        });
    }

     void compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) override {
         throw std::logic_error("OutputLayer::compute_gradients should be called with target, not next_layer.");
     }
};


// --- Classes Dataset / BatchHandler (Placeholders inchangés) ---
class Dataset {};
class BatchHandler {};

// --- Classe Network (Utilise Layer modifié, logic inchangée) ---
class Network {
public:
    std::map<int, int> layer_sizes_map; // Store original sizes map if needed
    InputLayer input_layer;
    std::vector<Layer> hidden_layers; // Uses base Layer class (now with sparse weights)
    OutputLayer output_layer;         // Uses Layer class (now with sparse weights)
    std::unique_ptr<Optimizer> optimizer;
    Dataset dataset;
    BatchHandler batch_handler;

    // Constructor (logic mostly unchanged, relies on Layer constructors)
    Network(const std::map<int, int>& _layer_sizes_map,
            const std::vector<std::string>& activation_types,
            std::unique_ptr<Optimizer> opt)
        : layer_sizes_map(_layer_sizes_map),
          input_layer(get_size(0)),
          output_layer(get_size(num_layers() - 2), get_size(num_layers() - 1), create_activation(activation_types.back())),
          optimizer(std::move(opt))
    {
        if (layer_sizes_map.size() < 2) {
            throw std::runtime_error("Network must have at least an input and output layer.");
        }
        if (activation_types.size() != layer_sizes_map.size() - 1) {
             throw std::runtime_error("Number of activation types mismatch.");
        }
        if (!optimizer) {
             throw std::runtime_error("Optimizer must be provided.");
        }

        int num_hidden_layers = layer_sizes_map.size() - 2;
        hidden_layers.reserve(num_hidden_layers);
        for (int i = 0; i < num_hidden_layers; ++i) {
            int current_layer_idx_in_map = i + 1;
            int in_size = get_size(current_layer_idx_in_map - 1);
            int out_size = get_size(current_layer_idx_in_map);
            std::string act_type = activation_types[i];
            hidden_layers.emplace_back(in_size, out_size, create_activation(act_type));
        }
    }

    int get_size(int layer_index) const {
        try {
            return layer_sizes_map.at(layer_index);
        } catch (const std::out_of_range& oor) {
            throw std::out_of_range("Layer index " + std::to_string(layer_index) + " not found.");
        }
     }
    int num_layers() const { return layer_sizes_map.size(); }

    // Forward pass (logic unchanged, uses Layer::forward)
    View1D forward(const View1D& input_data) {
        input_layer.set_input(input_data);
        const View1D* current_a = &input_layer.a;
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.forward(*current_a);
            current_a = &hidden_layer.a;
        }
        output_layer.forward(*current_a);
        return output_layer.a;
    }

    // Backward pass (logic unchanged, uses Layer::compute_gradients)
    void backward(const View1D& target) {
        const View1D& prev_a_output = hidden_layers.empty() ? input_layer.a : hidden_layers.back().a;
        output_layer.compute_gradients(target, prev_a_output);

        Layer* next_layer_ptr = &output_layer;
        for (int i = hidden_layers.size() - 1; i >= 0; --i) {
            Layer& current_layer = hidden_layers[i];
            const View1D& prev_a = (i == 0) ? input_layer.a : hidden_layers[i - 1].a;
            current_layer.compute_gradients(*next_layer_ptr, prev_a);
            next_layer_ptr = &current_layer;
        }
    }

     // Update weights and biases (logic unchanged, uses Optimizer::update)
    void update(int batch_size) {
        if (!optimizer) throw std::runtime_error("Optimizer not set.");
        if (batch_size <= 0) throw std::runtime_error("Batch size must be positive.");

        // Update hidden layers (Optimizer::update now handles sparse weights)
        for (Layer& layer : hidden_layers) {
            if (layer.input_size > 0) { // Check if the layer has weights
                optimizer->update(layer.weights, layer.biases,
                                  layer.d_weights_sum, layer.d_biases_sum,
                                  batch_size);
            }
        }
        // Update output layer
         if (output_layer.input_size > 0) {
            optimizer->update(output_layer.weights, output_layer.biases,
                              output_layer.d_weights_sum, output_layer.d_biases_sum,
                              batch_size);
         }
    }

    // Zero accumulated gradients (logic unchanged, uses Layer::zero_accumulated_gradients)
    void zero_accumulated_gradients() {
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.zero_accumulated_gradients();
        }
        output_layer.zero_accumulated_gradients();
    }

    // Calculate Cost (unchanged)
    real calculate_cost(const View1D& prediction, const View1D& target) {
        int output_size = prediction.extent_int(0);
        if (target.extent_int(0) != output_size) {
            throw std::runtime_error("Prediction/target size mismatch for cost.");
        }
        if (output_size == 0) return 0.0;

        real squared_error_sum = 0.0;
        Kokkos::parallel_reduce("compute_cost", output_size, KOKKOS_LAMBDA (int i, real& lsum) {
            real diff = prediction(i) - target(i);
            lsum += diff * diff;
        }, squared_error_sum);
        Kokkos::fence();
        return 0.5 * squared_error_sum;
    }

    // Optimizer Management (unchanged)
    void set_optimizer(std::unique_ptr<Optimizer> opt) {
        if (!opt) throw std::runtime_error("Cannot set a null optimizer.");
        optimizer = std::move(opt);
    }
    Optimizer* get_optimizer() const { return optimizer.get(); }
    void set_learning_rate(real lr) {
        if (!optimizer) throw std::runtime_error("Optimizer not set.");
        optimizer->set_learning_rate(lr);
    }
     real get_learning_rate() const {
         if (!optimizer) return 0.0;
        return optimizer->get_learning_rate();
    }

    // Display Network Info (uses Layer::show modified for sparse)
    void show() const {
         std::cout << "--- Network Structure (Sparse Weights) ---" << std::endl;
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
         std::cout << "------------------------------------------" << std::endl;
    }

    // Move/Copy Semantics (unchanged)
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    virtual ~Network() = default;
};

// --- Fonctions d'entraînement (XOR, Sine, Linear Separation) ---
// *** AUCUNE MODIFICATION NÉCESSAIRE ICI ***
// Ces fonctions utilisent l'interface publique de Network, Layer, Optimizer,
// qui a été maintenue stable (même si l'implémentation interne a changé
// pour utiliser des matrices creuses).

void xor_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- XOR Training Example (Sparse Weights Storage) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    // Structure du réseau
    std::map<int, int> sizes;
    sizes[0] = 2; sizes[1] = 80; sizes[2] = 40; sizes[3] = 1;
    std::vector<std::string> activations = {"relu", "relu", "sigmoid"};

    // Paramètres
    real learning_rate; int epochs; int batch_size = 4;
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.8; epochs = 1500;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.01; epochs = 800;
        optimizer = std::make_unique<Adam>(learning_rate);
    }

    Network dnn(sizes, activations, std::move(optimizer));
    // dnn.show(); // Show structure (will mention sparse)

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
        dnn.zero_accumulated_gradients();
        for (int i = 0; i < num_samples; ++i) {
            auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
            auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
            View1D prediction = dnn.forward(input_subview);
            dnn.backward(target_subview);
        }
        dnn.update(batch_size);

        // Recalculate cost for reporting
        real current_total_cost = 0.0;
        for (int i = 0; i < num_samples; ++i) {
             auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
             auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
             View1D prediction = dnn.forward(input_subview);
             current_total_cost += dnn.calculate_cost(prediction, target_subview);
        }
        real avg_cost = current_total_cost / num_samples;

        if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs -1) {
            std::cout << "Epoch: " << std::setw(5) << epoch + 1
                      << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost << std::endl;
        }
        if (avg_cost < 1e-4) {
             std::cout << "Convergence likely reached at epoch " << epoch + 1 << std::endl; break;
        }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    std::cout << "\n-- FINAL PREDICTIONS --" << std::endl;
    View1D prediction_result("prediction_result", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        std::cout << "Input: [" << h_xor_inputs(i,0) << "," << h_xor_inputs(i,1) << "] "
                  << "Target: " << h_xor_outputs(i,0) << " "
                  << "Prediction: " << std::fixed << std::setprecision(4) << h_prediction_result(0)
                  << " (Rounded: " << std::round(h_prediction_result(0)) << ")" << std::endl;
    }
     std::cout << "Final Average Cost: " << std::fixed << std::setprecision(8) << final_total_cost / num_samples << std::endl;
}


void sine_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- Sine Function Approx Training (Sparse Weights Storage) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    std::map<int, int> sizes;
    sizes[0] = 1; sizes[1] = 32; sizes[2] = 32; sizes[3] = 1;
    std::vector<std::string> activations = {"relu", "relu", "linear"};

    real learning_rate; int epochs; int batch_size = 16; int num_samples = 2048*4;
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.02; epochs = 500;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.001; epochs = 800;
        optimizer = std::make_unique<Adam>(learning_rate);
    }

    Network dnn(sizes, activations, std::move(optimizer));

    using HostView1D = Kokkos::View<real*, Kokkos::HostSpace>;
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0); const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_inputs("h_sine_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_sine_outputs", num_samples, output_dim);
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<real> distrib(-M_PI, M_PI);
    for(int i=0; i < num_samples; ++i) {
        real x = distrib(gen); h_inputs(i, 0) = x; h_outputs(i, 0) = std::sin(x);
    }
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << ", Samples: " << num_samples << std::endl;

    std::vector<int> indices(num_samples); std::iota(indices.begin(), indices.end(), 0);

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
        if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs - 1) {
             std::cout << "Epoch: " << std::setw(5) << epoch + 1 << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost << std::endl;
        }
         if (avg_cost < 1e-3) {
             // std::cout << "Good convergence likely reached at epoch " << epoch + 1 << std::endl;
             // break;
         }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    std::cout << "\n-- FINAL PREDICTIONS (Sample) --" << std::endl;
    View1D prediction_result("prediction_result_sine", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; int num_test_samples = std::min(num_samples, 10);
    for (int i = 0; i < num_test_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        real input_x = h_inputs(i, 0); real target_y = h_outputs(i, 0);
        std::cout << "Input x: " << std::fixed << std::setprecision(4) << input_x << " "
                  << "Target sin(x): " << std::fixed << std::setprecision(4) << target_y << " "
                  << "Prediction: " << std::fixed << std::setprecision(4) << h_prediction_result(0) << std::endl;
    }
    std::cout << "Final Average Cost (on first " << num_test_samples << " samples): " << std::fixed << std::setprecision(8) << final_total_cost / num_test_samples << std::endl;
}


void linear_sep_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- Linear Separation Training (Sparse Weights Storage) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    std::map<int, int> sizes; sizes[0] = 2; sizes[1] = 1;
    std::vector<std::string> activations = {"sigmoid"};

    real learning_rate; int epochs; int batch_size = 16; int num_samples = 2560;
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.1; epochs = 100;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.01; epochs = 150;
        optimizer = std::make_unique<Adam>(learning_rate);
    }

    Network dnn(sizes, activations, std::move(optimizer));

    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0); const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_inputs("h_linear_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_linear_outputs", num_samples, output_dim);
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + 123);
    std::uniform_real_distribution<real> distrib(-1.0, 1.0);
    real margin = 0.1; int count_class0 = 0; int count_class1 = 0;
    for(int i=0; i < num_samples; ++i) {
        real x = distrib(gen); real y = distrib(gen);
        h_inputs(i, 0) = x; h_inputs(i, 1) = y;
        if (y < x - margin) { h_outputs(i, 0) = 0.0; count_class0++; }
        else if (y > x + margin) { h_outputs(i, 0) = 1.0; count_class1++; }
        else { i--; continue; }
    }
    std::cout << "Generated " << count_class0 << " Class 0 and " << count_class1 << " Class 1." << std::endl;
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << ", Samples: " << num_samples << std::endl;

    std::vector<int> indices(num_samples); std::iota(indices.begin(), indices.end(), 0);

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
        if ((epoch + 1) % (epochs / 10) == 0 || epoch == 0 || epoch == epochs - 1) {
             std::cout << "Epoch: " << std::setw(4) << epoch + 1 << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost << std::endl;
        }
         if (avg_cost < 1e-3) {
             std::cout << "Convergence likely reached at epoch " << epoch + 1 << std::endl; break;
         }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    std::cout << "\n-- FINAL PREDICTIONS & ACCURACY --" << std::endl;
    View1D prediction_result("prediction_result_linear", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; int correct_predictions = 0;
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        real predicted_value = h_prediction_result(0); int predicted_class = std::round(predicted_value);
        int target_class = static_cast<int>(h_outputs(i, 0));
        if (predicted_class == target_class) correct_predictions++;
         if (i < 10) {
              std::cout << "Input: [" << std::fixed << std::setprecision(2) << h_inputs(i,0) << "," << h_inputs(i,1) << "] "
                        << "Target: " << target_class << " Pred: " << std::fixed << std::setprecision(3) << predicted_value
                        << " (Rounded: " << predicted_class << ")" << (predicted_class == target_class ? "" : " <-- WRONG") << std::endl;
         }
    }
    real accuracy = static_cast<real>(correct_predictions) / num_samples;
    std::cout << "Final Average Cost: " << std::fixed << std::setprecision(8) << final_total_cost / num_samples << std::endl;
    std::cout << "Final Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%" << std::endl;
}

// --- Main (inchangé) ---
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        try {
            std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
            std::cout << "*** NOTE: Using KokkosSparse::CrsMatrix for weights storage (initially dense structure). ***" << std::endl;

            xor_train("adam");
            std::cout << "\n---------------------------\n" << std::endl;
            // xor_train("sgd"); // Optionnel

            std::cout << "\n---------------------------\n" << std::endl;
            sine_train("adam");

            std::cout << "\n---------------------------\n" << std::endl;
            linear_sep_train("adam");


        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            Kokkos::finalize(); // Finalize explicitly on error if not using RAII finalize
            return 1;
        } catch (...) {
            std::cerr << "An unknown error occurred." << std::endl;
            Kokkos::finalize(); // Finalize explicitly on error if not using RAII finalize
             return 1;
        }
    } // Scope ensures Kokkos objects are destroyed before finalize if using RAII finalize
    Kokkos::finalize();
    return 0;
}









