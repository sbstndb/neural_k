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
#include <numeric>
#include <iomanip>
#include <typeinfo>
#include <chrono>
#include <algorithm> // Pour std::sort, std::min, std::max

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_StdAlgorithms.hpp> // Pour parallel_reduce, sort

#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosKernels_Handle.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Typedefs ---
using real = float;
using View1D = Kokkos::View<real*>;
using BoolView1D = Kokkos::View<bool*>; // Pour le masque d'élagage

using Scalar = real;
using Ordinal = int;
using Offset = size_t;
using Device = Kokkos::DefaultExecutionSpace;
using Layout = Kokkos::LayoutLeft; // Gardons LayoutLeft pour la cohérence conceptuelle
using SparseMatrixType = KokkosSparse::CrsMatrix<Scalar, Ordinal, Device, void, Offset>;
using GraphType = typename SparseMatrixType::staticcrsgraph_type;
using ValuesType = typename SparseMatrixType::values_type;
using RowMapType = typename GraphType::row_map_type::non_const_type;
using EntriesType = typename GraphType::entries_type::non_const_type;

// --- Forward Declarations ---
class Optimizer;
class SGD;
class Adam;
class Layer;
class Activation;
class Dataset;
class Network;
class BatchHandler;

// --- Helper: Cubic Pruning Schedule ---
// Fonction pour calculer la sparsité cible à une époque donnée
// selon un calendrier cubique (doux au début et à la fin)
KOKKOS_INLINE_FUNCTION
real cubic_pruning_schedule(int current_epoch, int start_epoch, int end_epoch, real initial_sparsity, real final_sparsity) {
    if (current_epoch < start_epoch || end_epoch <= start_epoch) {
        return initial_sparsity;
    }
    if (current_epoch >= end_epoch) {
        return final_sparsity;
    }
    // Normaliser le temps entre 0 et 1
    real time_normalized = static_cast<real>(current_epoch - start_epoch) / static_cast<real>(end_epoch - start_epoch);
    // Appliquer la fonction cubique f(t) = 3t^2 - 2t^3 (interpole entre 0 et 1 avec dérivées nulles aux extrêmes)
    real sparsity_progress = 3.0f * time_normalized * time_normalized - 2.0f * time_normalized * time_normalized * time_normalized;
    // Calculer la sparsité actuelle
    real current_target_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * sparsity_progress;
    return current_target_sparsity;
}


// --- Classe Optimizer ---
class Optimizer {
public:
    real learning_rate;

    Optimizer(real lr) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    // MODIFIED: Takes SparseMatrixType and the pruning mask
    virtual void update(SparseMatrixType& weights, View1D biases, // biases remain dense
                        const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
                        const BoolView1D& weight_mask, // Masque d'élagage
                        int batch_size) = 0;

    void set_learning_rate(real lr) { learning_rate = lr; }
    real get_learning_rate() const { return learning_rate; }

    virtual std::string get_info() const {
        return "Optimizer(LR=" + std::to_string(learning_rate) + ")";
    }
};

// --- Concrete Optimizer: SGD ---
class SGD : public Optimizer {
public:
    SGD(real lr = 0.1) : Optimizer(lr) {}

    // MODIFIED: Takes SparseMatrixType and mask
    void update(SparseMatrixType& weights, View1D biases,
                const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
                const BoolView1D& weight_mask, // Added mask
                int batch_size) override {

        if (batch_size <= 0) throw std::runtime_error("Batch size must be positive.");
        if (weights.numRows() == 0 && biases.extent(0) == 0) return;
        if (weights.numRows() > 0 && weights.numRows() != biases.extent(0)) {
            throw std::runtime_error("SGD update: Mismatch weights rows vs biases size.");
        }
        if (biases.extent_int(0) <= 0 && weights.numRows() <= 0) return;

        const real scale = learning_rate / static_cast<real>(batch_size);
        const int layer_size = biases.extent_int(0);

        // --- Update Weights (Sparse & Masked) ---
        auto w_vals = weights.values;
        auto dw_vals = accumulated_d_weights.values;
        const int nnz = w_vals.extent_int(0);

        if (nnz != dw_vals.extent_int(0) || (nnz > 0 && nnz != weight_mask.extent_int(0))) {
             throw std::runtime_error("SGD update: Mismatch in nnz or mask size.");
        }

        Kokkos::parallel_for("sgd_update_weights_sparse_masked", nnz, KOKKOS_LAMBDA(const int k) {
            // Only update if the weight is active (not pruned)
            if (weight_mask(k)) {
                w_vals(k) -= scale * dw_vals(k);
            }
            // Ensure pruned weights remain zero ( belt-and-suspenders, prune should handle this)
            // else {
            //     w_vals(k) = 0.0;
            // }
        });

        // --- Update Biases (Dense - unchanged) ---
        Kokkos::parallel_for("sgd_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
            biases(i) -= scale * accumulated_d_biases(i);
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
    // State structure (using dense moments for now)
    struct ParameterState {
        View1D m_1d; // For biases
        View1D v_1d; // For biases
        Kokkos::View<real**> m_2d; // Dense moments for weights
        Kokkos::View<real**> v_2d; // Dense moments for weights
        long long t = 0;

        ParameterState(size_t size) : t(0) { // Bias state
            m_1d = View1D("adam_m1d", size); v_1d = View1D("adam_v1d", size);
            Kokkos::deep_copy(m_1d, 0.0); Kokkos::deep_copy(v_1d, 0.0);
            // Initialize empty dense views to avoid nullptrs if accessed accidentally
            m_2d = Kokkos::View<real**>("adam_m2d_bias_empty", 0, 0);
            v_2d = Kokkos::View<real**>("adam_v2d_bias_empty", 0, 0);
        }
        ParameterState(size_t rows, size_t cols) : t(0) { // Weight state (dense moments)
             m_2d = Kokkos::View<real**>("adam_m2d", rows, cols);
             v_2d = Kokkos::View<real**>("adam_v2d", rows, cols);
             Kokkos::deep_copy(m_2d, 0.0); Kokkos::deep_copy(v_2d, 0.0);
             // Initialize empty 1d views
             m_1d = View1D("adam_m1d_weight_empty", 0); v_1d = View1D("adam_v1d_weight_empty", 0);
        }
         ParameterState() = default;
         ParameterState(ParameterState&&) = default;
         ParameterState& operator=(ParameterState&&) = default;
         ParameterState(const ParameterState&) = delete;
         ParameterState& operator=(const ParameterState&) = delete;
    };

    std::map<real*, ParameterState> state_map_1d;     // For biases (key = biases.data())
    std::map<real*, ParameterState> state_map_sparse; // For sparse weights (key = weights.values.data())

public:
    Adam(real lr = 0.001, real b1 = 0.9, real b2 = 0.999, real eps = 1e-8)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps) {
        if (lr <= 0 || b1 < 0 || b1 >= 1 || b2 < 0 || b2 >= 1 || eps <= 0) {
            throw std::runtime_error("Invalid Adam hyperparameters.");
        }
    }

    // MODIFIED: Takes SparseMatrixType and mask
    void update(SparseMatrixType& weights, View1D biases,
                const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
                const BoolView1D& weight_mask, // Added mask
                int batch_size) override {

        if (batch_size <= 0) throw std::runtime_error("Batch size must be positive.");
        if (weights.numRows() == 0 && biases.extent(0) == 0) return;
        if (weights.numRows() > 0 && weights.numRows() != biases.extent(0)) {
             throw std::runtime_error("Adam update: Mismatch weights rows vs biases size.");
        }
        if (biases.extent_int(0) <= 0 && weights.numRows() <= 0) return;

        const int layer_size = biases.extent_int(0); // numRows
        const int input_size = weights.numCols();    // numCols
        const real scale = 1.0 / static_cast<real>(batch_size);

        // --- Update Weights (Sparse Weights, Dense Moments, Masked) ---
        {
            real* weight_values_ptr = weights.values.data();
            auto it_w = state_map_sparse.find(weight_values_ptr);
            if (it_w == state_map_sparse.end()) {
                auto result = state_map_sparse.try_emplace(weight_values_ptr, weights.numRows(), weights.numCols());
                 if (!result.second) throw std::runtime_error("Failed to insert Adam state for sparse weights.");
                 it_w = result.first;
            }
            ParameterState& state_w = it_w->second;
            state_w.t++;

            const double beta1_pow_t = std::pow(static_cast<double>(beta1), static_cast<double>(state_w.t));
            const double beta2_pow_t = std::pow(static_cast<double>(beta2), static_cast<double>(state_w.t));
            const real bias_correction1 = 1.0f / (1.0f - static_cast<real>(beta1_pow_t));
            const real bias_correction2 = 1.0f / (1.0f - static_cast<real>(beta2_pow_t));

            real lr = learning_rate; real b1 = beta1; real b2 = beta2; real eps = epsilon;
            auto m = state_w.m_2d; auto v = state_w.v_2d; // Dense moments
            auto w_vals = weights.values; auto dw_vals = accumulated_d_weights.values;
            auto graph = weights.graph;
            const int nnz = w_vals.extent_int(0);

             if (nnz != dw_vals.extent_int(0) || (nnz > 0 && nnz != weight_mask.extent_int(0))) {
                 throw std::runtime_error("Adam update: Mismatch in nnz or mask size for weights.");
            }

            Kokkos::parallel_for("adam_update_weights_sparse_masked", layer_size, KOKKOS_LAMBDA (const int i) {
                const auto row_start = graph.row_map(i);
                const auto row_end = graph.row_map(i+1);
                for (auto k = row_start; k < row_end; ++k) {
                    const int j = graph.entries(k);

                    // Check the mask
                    if (weight_mask(k)) {
                        // Weight is active, perform Adam update
                        real grad = scale * dw_vals(k);
                        m(i, j) = b1 * m(i, j) + (1.0f - b1) * grad;
                        v(i, j) = b2 * v(i, j) + (1.0f - b2) * grad * grad;
                        real m_hat = m(i, j) * bias_correction1;
                        real v_hat = v(i, j) * bias_correction2;
                        w_vals(k) -= lr * m_hat / (Kokkos::sqrt(v_hat) + eps);
                    } else {
                        // Weight is pruned: ensure weight remains 0 and reset moments
                        // w_vals(k) = 0.0; // Should be already done by prune, but safer
                        m(i, j) = 0.0f;
                        v(i, j) = 0.0f;
                    }
                }
            });
        } // End scope weights

        // --- Update Biases (Dense - unchanged) ---
        {
             auto it_b = state_map_1d.find(biases.data());
             if (it_b == state_map_1d.end()) {
                 auto result = state_map_1d.try_emplace(biases.data(), biases.extent(0));
                 if (!result.second) throw std::runtime_error("Failed to insert Adam state for biases.");
                 it_b = result.first;
             }
             ParameterState& state_b = it_b->second;
             state_b.t++;

            const double beta1_pow_t = std::pow(static_cast<double>(beta1), static_cast<double>(state_b.t));
            const double beta2_pow_t = std::pow(static_cast<double>(beta2), static_cast<double>(state_b.t));
            const real bias_correction1 = 1.0f / (1.0f - static_cast<real>(beta1_pow_t));
            const real bias_correction2 = 1.0f / (1.0f - static_cast<real>(beta2_pow_t));

            real lr = learning_rate; real b1 = beta1; real b2 = beta2; real eps = epsilon;
            auto m = state_b.m_1d; auto v = state_b.v_1d;

            Kokkos::parallel_for("adam_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
                real grad = scale * accumulated_d_biases(i);
                m(i) = b1 * m(i) + (1.0f - b1) * grad;
                v(i) = b2 * v(i) + (1.0f - b2) * grad * grad;
                real m_hat = m(i) * bias_correction1;
                real v_hat = v(i) * bias_correction2;
                biases(i) -= lr * m_hat / (Kokkos::sqrt(v_hat) + eps);
            });
        } // End scope biases
    }

     std::string get_info() const override {
         return "Adam(LR=" + std::to_string(learning_rate) +
                ", b1=" + std::to_string(beta1) + ", b2=" + std::to_string(beta2) + ")";
     }
     virtual ~Adam() override = default;
};

// --- Classes Activation (inchangées) ---
// ... (RELU, SIGMOID, TANH, LinearActivation, create_activation) ...
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

std::unique_ptr<Activation> create_activation(const std::string& type) {
    if (type == "relu") return std::make_unique<RELU>();
    if (type == "sigmoid") return std::make_unique<SIGMOID>();
    if (type == "tanh") return std::make_unique<TANH>();
    if (type == "linear") return std::make_unique<LinearActivation>();
    throw std::runtime_error("Unknown activation type: " + type);
}


// --- Classe Layer (MODIFIED for Sparse Weights & Pruning) ---
class Layer {
public:
    int input_size;
    int layer_size;
    std::unique_ptr<Activation> activation;

    // Sparse weights and dense biases
    SparseMatrixType weights;
    View1D biases;
    View1D z;
    View1D a;

    // Gradients (Sparse for weights, Dense for biases)
    View1D delta;
    View1D d_biases;
    SparseMatrixType d_weights; // Instantaneous (masked)
    View1D d_biases_sum;
    SparseMatrixType d_weights_sum; // Accumulated (masked)

    View1D tmp_deriv;

    // *** NEW: Pruning Mask ***
    BoolView1D weight_mask; // Same size as weights.values

    // Structure to store magnitude and original index for sorting
    struct MagnitudeIndex {
        real magnitude;
        int index; // Index into the values/mask array

        // Comparaison pour le tri (ordre croissant de magnitude)
        KOKKOS_INLINE_FUNCTION bool operator<(const MagnitudeIndex& other) const {
            return magnitude < other.magnitude;
        }
    };
     using MagView = Kokkos::View<MagnitudeIndex*>;


    // Constructor (MODIFIED for mask init)
    Layer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        input_size(_input_size),
        layer_size(_layer_size),
        activation(std::move(act_func))
    {
        if (input_size < 0 || layer_size <= 0) throw std::runtime_error("Invalid layer sizes.");

        // Allocate dense views
        biases = View1D("biases", layer_size); z = View1D("z", layer_size);
        a = View1D("a", layer_size); delta = View1D("delta", layer_size);
        d_biases = View1D("d_biases", layer_size);
        d_biases_sum = View1D("d_biases_sum", layer_size);
        tmp_deriv = View1D("tmp_deriv", layer_size);
        Kokkos::deep_copy(biases, 0.0); Kokkos::deep_copy(z, 0.0); Kokkos::deep_copy(a, 0.0);
        Kokkos::deep_copy(delta, 0.0); Kokkos::deep_copy(d_biases, 0.0);
        Kokkos::deep_copy(d_biases_sum, 0.0); Kokkos::deep_copy(tmp_deriv, 0.0);

        // Initialize Sparse Matrices (dense structure initially)
        if (input_size > 0) {
            size_t nnz = static_cast<size_t>(layer_size) * input_size;
            RowMapType row_map("row_map", layer_size + 1);
            EntriesType entries("entries", nnz);
            ValuesType w_values("w_values", nnz);
            ValuesType dw_values("dw_values", nnz);
            ValuesType dw_sum_values("dw_sum_values", nnz);

            // Create dense graph
            Kokkos::parallel_for("create_dense_graph", layer_size, KOKKOS_LAMBDA(const int i) {
                row_map(i) = static_cast<Offset>(i) * input_size;
                for (int j = 0; j < input_size; ++j) {
                    entries(static_cast<Offset>(i) * input_size + j) = j;
                }
                if (i == layer_size - 1) row_map(layer_size) = nnz;
            });
            Kokkos::fence();

            // Initialize weights (Xavier)
	    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + reinterpret_cast<uintptr_t>(this);
            Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
            real limit = std::sqrt(6.0f / (input_size + layer_size));
            Kokkos::fill_random(w_values, rand_pool, static_cast<real>(-limit), static_cast<real>(limit));
            Kokkos::deep_copy(dw_values, 0.0); Kokkos::deep_copy(dw_sum_values, 0.0);

            GraphType graph(entries, row_map);
            weights       = SparseMatrixType("weights",       input_size, w_values,      graph);
            d_weights     = SparseMatrixType("d_weights",     input_size, dw_values,     graph);
            d_weights_sum = SparseMatrixType("d_weights_sum", input_size, dw_sum_values, graph);

            // *** NEW: Initialize Mask ***
            weight_mask = BoolView1D("weight_mask", nnz);
            Kokkos::deep_copy(weight_mask, true); // Start with all weights active

        } else { // Empty matrices/mask if input_size == 0
            RowMapType row_map("row_map_empty", layer_size + 1);
            EntriesType entries("entries_empty", 0);
            ValuesType w_values("w_values_empty", 0);
            ValuesType dw_values("dw_values_empty", 0);
            ValuesType dw_sum_values("dw_sum_values_empty", 0);
            Kokkos::deep_copy(row_map, static_cast<Offset>(0));
            GraphType graph(entries, row_map);
            weights = SparseMatrixType("weights_empty", 0, w_values, graph);
            d_weights = SparseMatrixType("d_weights_empty", 0, dw_values, graph);
            d_weights_sum = SparseMatrixType("d_weights_sum_empty", 0, dw_sum_values, graph);
            weight_mask = BoolView1D("weight_mask_empty", 0); // Empty mask
        }
    }

    // Input Layer constructor
    Layer(int _layer_size) : Layer(0, _layer_size, nullptr) {}

    // --- Methods ---
    // Forward pass (Unchanged - SpMV handles zeros correctly)
    virtual void forward(const View1D& prev_layer_a) {
        if (input_size == 0) return;
        if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) throw std::runtime_error("Forward size mismatch.");

        KokkosSparse::spmv("N", 1.0, weights, prev_layer_a, 0.0, z);
        Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) { z(i) += biases(i); });
        if (activation) activation->apply(z, a); else Kokkos::deep_copy(a, z);
    }

    // Compute gradients (MODIFIED for mask)
    virtual void compute_gradients(const Layer& next_layer, const View1D& prev_layer_a) {
         if (input_size == 0) return;
         // Dimension checks...
         if (next_layer.weights.numCols() != static_cast<size_t>(layer_size) || next_layer.delta.extent(0) != next_layer.weights.numRows()) {
             throw std::runtime_error("Gradient dimension mismatch (next layer).");
         }
         if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
            throw std::runtime_error("Gradient dimension mismatch (prev_layer_a).");
         }

        // 1. f'(z)
        if (activation) activation->apply_derivative(z, tmp_deriv); else Kokkos::deep_copy(tmp_deriv, 1.0f);

        // 2. delta_l = (W_{l+1}^T * δ_{l+1}) .* f'(z_l)
        View1D delta_prop("delta_prop", layer_size);
        KokkosSparse::spmv("T", 1.0, next_layer.weights, next_layer.delta, 0.0, delta_prop);
        Kokkos::parallel_for("hadamard_delta_deriv", layer_size, KOKKOS_LAMBDA(int i) { delta(i) = delta_prop(i) * tmp_deriv(i); });

        // 3. Instantaneous gradients (d_biases, d_weights)
        Kokkos::deep_copy(d_biases, delta);

        auto dw_vals = d_weights.values;
        auto graph = d_weights.graph; // Assumes same structure
        auto current_mask = weight_mask; // Capture mask
        if (dw_vals.extent(0) != current_mask.extent(0) && dw_vals.extent(0) > 0) {
             throw std::runtime_error("d_weights mask size mismatch");
        }

        // *** MODIFIED: Only calculate gradient if weight is active ***
        Kokkos::parallel_for("compute_d_weights_sparse_masked", layer_size, KOKKOS_LAMBDA(const int i) {
             const auto row_start = graph.row_map(i);
             const auto row_end = graph.row_map(i+1);
             for (auto k = row_start; k < row_end; ++k) {
                 if (current_mask(k)) { // Check mask
                    const int j = graph.entries(k);
                    dw_vals(k) = delta(i) * prev_layer_a(j);
                 } else {
                    dw_vals(k) = 0.0f; // Ensure gradient is zero if pruned
                 }
             }
         });

        // 4. Accumulate gradients
        auto dw_sum_vals = d_weights_sum.values;
        const int nnz = dw_vals.extent_int(0);
         if (dw_sum_vals.extent(0) != nnz) throw std::runtime_error("Accumulated d_weights nnz mismatch.");

         // Accumulation naturally handles zeros from masked d_weights
         Kokkos::parallel_for("accumulate_weight_gradients_sparse", nnz, KOKKOS_LAMBDA (const int k) {
            Kokkos::atomic_add(&dw_sum_vals(k), dw_vals(k));
         });
         Kokkos::parallel_for("accumulate_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
            Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
         });
    }

    // Zero accumulated gradients (MODIFIED to ensure mask consistency)
    void zero_accumulated_gradients() {
         if (input_size == 0 && layer_size == 0) return;
         if (d_biases_sum.data() != nullptr) Kokkos::deep_copy(d_biases_sum, 0.0);

         if (d_weights_sum.values.data() != nullptr && d_weights_sum.nnz() > 0) {
              // Zero all accumulated weight gradients
              Kokkos::deep_copy(d_weights_sum.values, 0.0);
              // Optionally: can explicitly zero only masked ones, but deep_copy is simpler
              // auto dw_sum_vals = d_weights_sum.values;
              // auto mask = weight_mask;
              // Kokkos::parallel_for(mask.extent(0), KOKKOS_LAMBDA(const int k){
              //    if (!mask(k)) dw_sum_vals(k) = 0.0;
              // });
         }
    }

    // *** NEW: Pruning Method ***
    virtual void prune(real target_sparsity) {
        if (weights.nnz() == 0 || target_sparsity <= 0.0) {
            return; // No weights to prune or no pruning requested
        }
        if (target_sparsity >= 1.0) { // Prune everything
             Kokkos::deep_copy(weight_mask, false);
             Kokkos::deep_copy(weights.values, 0.0);
             Kokkos::deep_copy(d_weights_sum.values, 0.0);
             return;
        }

        const size_t total_weights = weights.nnz();
        size_t num_active_weights = 0;

        // Count currently active weights
        Kokkos::parallel_reduce("count_active", total_weights, KOKKOS_LAMBDA (const int k, size_t& count) {
            if (weight_mask(k)) {
                count++;
            }
        }, num_active_weights);
        Kokkos::fence(); // Ensure count is finished

        if (num_active_weights == 0) return; // Nothing active to prune

        // Calculate how many weights should remain active
        size_t num_to_keep = static_cast<size_t>(std::round(static_cast<real>(total_weights) * (1.0f - target_sparsity)));
        num_to_keep = std::min(num_active_weights, num_to_keep); // Cannot keep more than currently active

        if (num_to_keep >= num_active_weights) {
             // Target sparsity already met or exceeded by current active weights
             // Optional: Could un-prune here if needed, but standard pruning usually only removes weights.
             // Let's just ensure the number of active weights matches the current mask count.
             num_to_keep = num_active_weights;
             // We don't need to prune further if num_to_keep >= num_active_weights
             if (num_to_keep == num_active_weights) return;
             // If somehow num_to_keep > num_active_weights (rounding?), adjust.
             num_to_keep = num_active_weights;
        }


        // Get magnitudes of *active* weights only
        MagView active_magnitudes("active_magnitudes", num_active_weights);
        ValuesType w_vals = weights.values;
        BoolView1D mask = weight_mask;
	Kokkos::View<MagnitudeIndex, Kokkos::HostSpace> h_threshold_value("h_threshold_value"); // Rank 0 view on host
        View1D::HostMirror h_threshold("h_threshold", 1); // To store threshold on host

        // --- Find Threshold ---
        // 1. Fill view with magnitudes and original indices of active weights
        Kokkos::parallel_scan("gather_active_magnitudes", total_weights, KOKKOS_LAMBDA (const int k, size_t& update, const bool final) {
            const bool is_active = mask(k);
            if (final && is_active) {
                 // Store magnitude and original index k
                active_magnitudes(update) = MagnitudeIndex{Kokkos::abs(w_vals(k)), k};
            }
            if (is_active) { // Increment count for active elements
                 update++;
             }
        });
         Kokkos::fence();

        // 2. Sort the active magnitudes on the device
        // Using Kokkos::sort directly on the MagView with the custom < operator
        Kokkos::Experimental::sort(active_magnitudes); // Sorts in ascending order of magnitude
        Kokkos::fence();

        // 3. Find the threshold magnitude (value of the smallest weight to KEEP)
        // The threshold is the magnitude of the (num_active_weights - num_to_keep)-th element
        // after sorting (0-based index). If num_to_keep == num_active_weights, this index is -1 (all kept).
        // If num_to_keep == 0, index is num_active_weights-1 (threshold is max magnitude).
        real magnitude_threshold = 0.0f;
        if (num_to_keep < num_active_weights) {
             size_t threshold_idx = num_active_weights - num_to_keep; // Index of first element to prune (after sorting)
             if (threshold_idx < active_magnitudes.extent(0)) { // Check bounds
                 // Copy the threshold magnitude to the host
                 auto threshold_subview = Kokkos::subview(active_magnitudes, threshold_idx);
                 // Need a temporary MagView on host
                 MagView::HostMirror h_active_magnitudes_thresh = Kokkos::create_mirror_view(threshold_subview);
                 Kokkos::deep_copy(h_active_magnitudes_value, threshold_subview);
                 magnitude_threshold = h_threshold_value().magnitude;
             } else {
                 // Should not happen if num_to_keep < num_active_weights, but handle defensively
                 // If threshold_idx is out of bounds, maybe keep all? Or use max magnitude?
                 // Let's keep all remaining active weights in this edge case.
                 num_to_keep = num_active_weights;
                 magnitude_threshold = -1.0; // Indicate keep all active
             }
        } else {
            // Keep all currently active weights
            magnitude_threshold = -1.0; // Special value to indicate keep all active
        }


        // --- Apply Pruning ---
        // Update the mask: prune weights below the threshold
        auto dw_sum_vals = d_weights_sum.values; // Need this to zero gradients

        Kokkos::parallel_for("apply_pruning_mask", total_weights, KOKKOS_LAMBDA(const int k) {
            if (mask(k)) { // Only consider currently active weights
                bool should_prune = false;
                 if (magnitude_threshold < 0.0) { // Keep all active
                      should_prune = false;
                  } else if (Kokkos::abs(w_vals(k)) < magnitude_threshold) {
                      // Magnitude is below threshold, prune it
                      should_prune = true;
                  } else if (Kokkos::abs(w_vals(k)) == magnitude_threshold) {
                      // Handle ties: potentially prune weights exactly at the threshold
                      // To ensure we reach num_to_keep, we might need to prune some ties.
                      // A simple approach: prune ties arbitrarily based on index 'k'
                      // Or better: rely on the sort order (those earlier in sorted list are pruned first)
                      // This requires knowing which original index 'k' corresponds to the threshold_idx
                      // Simpler: Just prune strict inequality for now. Might keep slightly more than num_to_keep.
                      // Let's stick to strict inequality for simplicity.
                      should_prune = false;
                  }

                  if (should_prune) {
                      mask(k) = false;        // Update mask
                      w_vals(k) = 0.0f;       // Zero weight value
                      dw_sum_vals(k) = 0.0f;  // Zero accumulated gradient
                      // Note: d_weights (instantaneous) is already zeroed in compute_gradients
                  }
            } else {
                 // If already pruned, ensure weight and gradient sum remain zero
                 // w_vals(k) = 0.0f;
                 // dw_sum_vals(k) = 0.0f;
             }
        });
        Kokkos::fence(); // Ensure pruning is complete
    }


    // Display layer info (MODIFIED for sparse weights & mask)
    void show() const {
        // ... (Existing info: size, activation) ...
        if (input_size == 0 && activation == nullptr) { // Input Layer
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
            size_t total_nnz = weights.nnz();
            size_t current_active = 0;
            if (weight_mask.data() != nullptr && weight_mask.extent(0) == total_nnz) {
                Kokkos::parallel_reduce("count_active_show", total_nnz, KOKKOS_LAMBDA(const int k, size_t& count){
                    if(weight_mask(k)) count++;
                }, current_active);
                 Kokkos::fence();
            } else {
                current_active = total_nnz; // Assume all active if mask invalid/missing
            }

            real current_sparsity = (total_nnz > 0) ? 1.0f - (static_cast<real>(current_active) / total_nnz) : 0.0f;

            std::cout << "  Weights (Sparse): " << weights.numRows() << "x" << weights.numCols()
                      << ", Potential NNZ: " << total_nnz
                      << ", Active NNZ: " << current_active
                      << " (Sparsity: " << std::fixed << std::setprecision(2) << current_sparsity * 100.0 << "%)"
                      << std::endl;

            // ... (Optional: print sample values) ...

        } else {
             std::cout << "  Weights: N/A" << std::endl;
        }

        // --- Display Biases Info (Unchanged) ---
        if (biases.data() != nullptr && biases.extent(0) > 0) {
            // ... (Existing bias print logic) ...
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
             std::cout << "  Biases: N/A" << std::endl;
         }
    }

    // Move/Copy Semantics (Defaults are OK now)
    Layer(Layer&& other) = default;
    Layer& operator=(Layer&& other) = default;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    virtual ~Layer() = default;
};


// --- Classe InputLayer (hérite de Layer modifié, prune fait rien) ---
class InputLayer : public Layer {
public:
    InputLayer(int _layer_size) : Layer(_layer_size) {}

    void set_input(const View1D& input_data) {
         if (input_data.extent(0) != static_cast<size_t>(layer_size)) throw std::runtime_error("Input size mismatch.");
         if (a.data() == nullptr || a.extent(0) != static_cast<size_t>(layer_size)) throw std::runtime_error("InputLayer 'a' invalid.");
         Kokkos::deep_copy(a, input_data);
     }
     void forward(const View1D&) override { }
     void compute_gradients(const Layer&, const View1D&) override { }
     // Prune does nothing for InputLayer
     void prune(real) override { }
};


// --- Classe OutputLayer (hérite de Layer modifié) ---
class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        Layer(_input_size, _layer_size, std::move(act_func)) {}

    // Special gradient computation (MODIFIED for mask)
    void compute_gradients(const View1D& target, const View1D& prev_layer_a) {
        // Dimension checks...
        if (target.extent(0) != static_cast<size_t>(layer_size)) throw std::runtime_error("Output target size mismatch.");
        if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) throw std::runtime_error("Output prev_a size mismatch.");

        // 1. f'(z_L)
        if (activation) activation->apply_derivative(z, tmp_deriv); else Kokkos::deep_copy(tmp_deriv, 1.0f);

        // 2. δ_L = (a_L - y) .* f'(z_L)
        Kokkos::parallel_for("compute_output_delta", layer_size, KOKKOS_LAMBDA(int i) { delta(i) = (a(i) - target(i)) * tmp_deriv(i); });

       // 3. Instantaneous gradients (d_biases, d_weights)
        Kokkos::deep_copy(d_biases, delta);

        auto dw_vals = d_weights.values;
        auto graph = d_weights.graph;
        auto current_mask = weight_mask; // Capture mask
         if (dw_vals.extent(0) != current_mask.extent(0) && dw_vals.extent(0) > 0) {
             throw std::runtime_error("Output d_weights mask size mismatch");
        }

        // *** MODIFIED: Only calculate gradient if weight is active ***
        Kokkos::parallel_for("compute_output_d_weights_sparse_masked", layer_size, KOKKOS_LAMBDA(const int i) {
             const auto row_start = graph.row_map(i);
             const auto row_end = graph.row_map(i+1);
             for (auto k = row_start; k < row_end; ++k) {
                 if (current_mask(k)) { // Check mask
                     const int j = graph.entries(k);
                     dw_vals(k) = delta(i) * prev_layer_a(j);
                 } else {
                     dw_vals(k) = 0.0f; // Ensure gradient is zero if pruned
                 }
             }
        });

        // 4. Accumulate gradients
        auto dw_sum_vals = d_weights_sum.values;
        const int nnz = dw_vals.extent_int(0);
        if (dw_sum_vals.extent(0) != nnz) throw std::runtime_error("Output accumulated d_weights nnz mismatch.");

        Kokkos::parallel_for("accumulate_output_weight_gradients_sparse", nnz, KOKKOS_LAMBDA (const int k) {
            Kokkos::atomic_add(&dw_sum_vals(k), dw_vals(k));
        });
        Kokkos::parallel_for("accumulate_output_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
        });
    }

     void compute_gradients(const Layer&, const View1D&) override {
         throw std::logic_error("OutputLayer::compute_gradients called incorrectly.");
     }
     // Pruning is handled by the base Layer::prune method
};


// --- Classes Dataset / BatchHandler (inchangées) ---
class Dataset {};
class BatchHandler {};


// --- Classe Network (MODIFIED for Pruning Control) ---
class Network {
public:
    std::map<int, int> layer_sizes_map;
    InputLayer input_layer;
    std::vector<Layer> hidden_layers;
    OutputLayer output_layer;
    std::unique_ptr<Optimizer> optimizer;
    Dataset dataset;
    BatchHandler batch_handler;

    // Constructor (inchangé)
    Network(const std::map<int, int>& _layer_sizes_map,
            const std::vector<std::string>& activation_types,
            std::unique_ptr<Optimizer> opt)
        : layer_sizes_map(_layer_sizes_map),
          input_layer(get_size(0)),
          output_layer(get_size(num_layers() - 2), get_size(num_layers() - 1), create_activation(activation_types.back())),
          optimizer(std::move(opt))
    {
        if (layer_sizes_map.size() < 2) throw std::runtime_error("Network needs >= 2 layers.");
        if (activation_types.size() != layer_sizes_map.size() - 1) throw std::runtime_error("Activation types mismatch.");
        if (!optimizer) throw std::runtime_error("Optimizer required.");

        int num_hidden_layers = layer_sizes_map.size() - 2;
        hidden_layers.reserve(num_hidden_layers);
        for (int i = 0; i < num_hidden_layers; ++i) {
            int current_idx = i + 1;
            hidden_layers.emplace_back(get_size(current_idx - 1), get_size(current_idx), create_activation(activation_types[i]));
        }
    }

    // Helper methods (inchangés)
    int get_size(int layer_index) const { /* ... */
        try { return layer_sizes_map.at(layer_index); }
        catch (const std::out_of_range&) { throw std::out_of_range("Layer index not found."); }
    }
    int num_layers() const { return layer_sizes_map.size(); }

    // Forward pass (inchangé)
    View1D forward(const View1D& input_data) { /* ... */
        input_layer.set_input(input_data);
        const View1D* current_a = &input_layer.a;
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.forward(*current_a); current_a = &hidden_layer.a;
        }
        output_layer.forward(*current_a);
        return output_layer.a;
     }

    // Backward pass (inchangé)
    void backward(const View1D& target) { /* ... */
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

     // Update weights (MODIFIED: passes mask to optimizer)
    void update(int batch_size) {
        if (!optimizer) throw std::runtime_error("Optimizer not set.");
        if (batch_size <= 0) throw std::runtime_error("Batch size must be positive.");

        // Update hidden layers
        for (Layer& layer : hidden_layers) {
            if (layer.input_size > 0) {
                optimizer->update(layer.weights, layer.biases,
                                  layer.d_weights_sum, layer.d_biases_sum,
                                  layer.weight_mask, // Pass the mask
                                  batch_size);
            }
        }
        // Update output layer
         if (output_layer.input_size > 0) {
            optimizer->update(output_layer.weights, output_layer.biases,
                              output_layer.d_weights_sum, output_layer.d_biases_sum,
                              output_layer.weight_mask, // Pass the mask
                              batch_size);
         }
    }

    // Zero accumulated gradients (inchangé)
    void zero_accumulated_gradients() { /* ... */
        for (Layer& hidden_layer : hidden_layers) hidden_layer.zero_accumulated_gradients();
        output_layer.zero_accumulated_gradients();
    }

    // Calculate Cost (inchangé)
    real calculate_cost(const View1D& prediction, const View1D& target) { /* ... */
        int output_size = prediction.extent_int(0);
        if (target.extent_int(0) != output_size) throw std::runtime_error("Cost size mismatch.");
        if (output_size == 0) return 0.0;
        real squared_error_sum = 0.0;
        Kokkos::parallel_reduce("compute_cost", output_size, KOKKOS_LAMBDA (int i, real& lsum) {
            real diff = prediction(i) - target(i); lsum += diff * diff;
        }, squared_error_sum);
        Kokkos::fence(); return 0.5 * squared_error_sum;
     }

    // *** NEW: Prune Network Method ***
    void prune_network(real target_sparsity) {
        std::cout << "  Pruning network to target sparsity: "
                  << std::fixed << std::setprecision(3) << target_sparsity << std::endl;
        for (Layer& layer : hidden_layers) {
            if (layer.input_size > 0) {
                layer.prune(target_sparsity);
            }
        }
        if (output_layer.input_size > 0) {
             output_layer.prune(target_sparsity);
        }
        Kokkos::fence(); // Ensure pruning kernels complete before proceeding
         // Optional: Show sparsity after pruning
         // show_sparsity();
    }

    // --- Optimizer Management (inchangé) ---
    void set_optimizer(std::unique_ptr<Optimizer> opt) { /* ... */
       if (!opt) throw std::runtime_error("Cannot set null optimizer.");
       optimizer = std::move(opt);
    }
    Optimizer* get_optimizer() const { return optimizer.get(); }
    void set_learning_rate(real lr) { /* ... */
       if (!optimizer) throw std::runtime_error("Optimizer not set.");
       optimizer->set_learning_rate(lr);
     }
     real get_learning_rate() const { /* ... */
        if (!optimizer) return 0.0;
        return optimizer->get_learning_rate();
      }

    // Display Network Info (uses modified Layer::show)
    void show() const { /* ... */
        std::cout << "--- Network Structure (Dynamic Sparse Training) ---" << std::endl;
         input_layer.show();
         int i = 1;
         for (const auto& layer : hidden_layers) {
              std::cout << "\n--- Hidden Layer " << i++ << " ---" << std::endl; layer.show();
         }
         std::cout << "\n--- Output Layer ---" << std::endl; output_layer.show();
         std::cout << "\n--- Optimizer Info ---" << std::endl;
         if(optimizer) std::cout << "  " << optimizer->get_info() << std::endl;
         else std::cout << "  Optimizer: Not set" << std::endl;
         std::cout << "---------------------------------------------------" << std::endl;
    }


    // Helper to show current sparsity (FIXED lambda captures)
    void show_sparsity() const {
        std::cout << "--- Current Network Sparsity ---" << std::endl;
        size_t network_total_weights = 0;
        size_t network_active_weights = 0;
        int i = 1;
        for (const auto& layer : hidden_layers) {
             if (layer.input_size > 0 && layer.weights.nnz() > 0) {
                size_t total_nnz = layer.weights.nnz();
                size_t current_active = 0;
                 // *** FIX: Capture mask by value ***
                 auto mask = layer.weight_mask;
                 Kokkos::parallel_reduce("count_active_show_h", total_nnz, KOKKOS_LAMBDA(const int k, size_t& count){ if(mask(k)) count++;}, current_active); Kokkos::fence();
                 network_total_weights += total_nnz;
                 network_active_weights += current_active;
                 real sparsity = 1.0 - (static_cast<real>(current_active) / total_nnz);
                 std::cout << "  Hidden Layer " << i << ": Active NNZ = " << current_active << " / " << total_nnz
                           << " (Sparsity: " << std::fixed << std::setprecision(2) << sparsity * 100.0 << "%)" << std::endl;
             }
             i++; // Increment layer index regardless
        }
         if (output_layer.input_size > 0 && output_layer.weights.nnz() > 0) {
            size_t total_nnz = output_layer.weights.nnz();
            size_t current_active = 0;
             // *** FIX: Capture mask by value ***
             auto mask_out = output_layer.weight_mask;
             Kokkos::parallel_reduce("count_active_show_o", total_nnz, KOKKOS_LAMBDA(const int k, size_t& count){ if(mask_out(k)) count++;}, current_active); Kokkos::fence();
             network_total_weights += total_nnz;
             network_active_weights += current_active;
             real sparsity = 1.0 - (static_cast<real>(current_active) / total_nnz);
             std::cout << "  Output Layer: Active NNZ = " << current_active << " / " << total_nnz
                       << " (Sparsity: " << std::fixed << std::setprecision(2) << sparsity * 100.0 << "%)" << std::endl;
         }
         // ... (rest of show_sparsity) ...
    }


    // Move/Copy Semantics (inchangé)
    Network(Network&&) = default; Network& operator=(Network&&) = default;
    Network(const Network&) = delete; Network& operator=(const Network&) = delete;
    virtual ~Network() = default;
};


// --- Fonctions d'entraînement (MODIFIED for Pruning Schedule) ---

// Base training function with pruning parameters
void train_network(Network& dnn,
                  Kokkos::View<real**> train_inputs, Kokkos::View<real**> train_outputs,
                  int epochs, int batch_size,
                  int pruning_start_epoch, int pruning_end_epoch, int pruning_frequency,
                  real initial_sparsity, real final_sparsity)
{
    const int num_samples = train_inputs.extent_int(0);
    if (num_samples == 0) { std::cerr << "Warning: No training samples provided." << std::endl; return; }

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << ", Samples: " << num_samples << std::endl;
    std::cout << "Pruning: Start=" << pruning_start_epoch << ", End=" << pruning_end_epoch
              << ", Freq=" << pruning_frequency << ", Sparsity=" << initial_sparsity*100 << "%->" << final_sparsity*100 << "%" << std::endl;

    std::vector<int> indices(num_samples); std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + 42);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;

        // --- Pruning Step ---
        // Check if pruning should occur at this epoch
        if (epoch >= pruning_start_epoch && epoch <= pruning_end_epoch && (epoch - pruning_start_epoch) % pruning_frequency == 0)
        {
            // Calculate the target sparsity for this pruning step using the schedule
            real current_target_sparsity = cubic_pruning_schedule(epoch, pruning_start_epoch, pruning_end_epoch, initial_sparsity, final_sparsity);
            dnn.prune_network(current_target_sparsity);
        }

        // --- Training Step ---
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
                total_epoch_cost += dnn.calculate_cost(prediction, target_subview); // Accumulate cost before backprop
                dnn.backward(target_subview);
            }
            dnn.update(current_batch_size); // Update weights after batch
        }

        // --- Logging ---
        real avg_cost = total_epoch_cost / num_samples;
        if ((epoch + 1) % (epochs / 20 + 1) == 0 || epoch == 0 || epoch == epochs - 1) {
             std::cout << "Epoch: " << std::setw(5) << epoch + 1
                       << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost;
             // Show sparsity periodically
             if (epoch == 0 || (epoch + 1) % (epochs / 5 + 1) == 0 || epoch == epochs - 1) {
                 // Recalculate active count for accurate reporting
                 size_t network_total_weights = 0; size_t network_active_weights = 0;
                 for (const auto& layer : dnn.hidden_layers) { 
			 if(layer.input_size > 0 && layer.weights.nnz() > 0){
				 network_total_weights += layer.weights.nnz(); 
				 auto current_mask = layer.weight_mask; // Capture mask
				 Kokkos::parallel_reduce(layer.weights.nnz(), KOKKOS_LAMBDA(int k, size_t& c){ 
						 if(weight_mask(k)) c++;}, network_active_weights);
			 }
		 }
                 if(dnn.output_layer.input_size > 0 && dnn.output_layer.weights.nnz() > 0) { 
			 network_total_weights += dnn.output_layer.weights.nnz(); 
			 auto output_mask = dnn.output_layer.weight_mask;
			 Kokkos::parallel_reduce(dnn.output_layer.weights.nnz(), KOKKOS_LAMBDA(int k, size_t& c){ 
					 if(output_mask(k)) c++;
					 }, network_active_weights);}
			 Kokkos::fence();

                 real current_sparsity = (network_total_weights > 0) ? 1.0 - (static_cast<real>(network_active_weights) / network_total_weights) : 0.0;
                  std::cout << ", Sparsity: " << std::fixed << std::setprecision(2) << current_sparsity * 100.0 << "%";
             }
             std::cout << std::endl;
        }

        // Convergence check (optional)
        // if (avg_cost < 1e-4) { /* ... break ... */ }

    } // End epoch loop
    std::cout << "-- TRAINING END --" << std::endl;
    dnn.show_sparsity(); // Show final sparsity details
}


void xor_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- XOR Training Example (Dynamic Sparse) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    std::map<int, int> sizes; sizes[0] = 2; sizes[1] = 16; sizes[2] = 8; sizes[3] = 1; // Slightly larger for pruning
    std::vector<std::string> activations = {"relu", "relu", "sigmoid"};

    real learning_rate; int epochs; int batch_size = 4;
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.5; epochs = 2000; // SGD might need more epochs/tuning with sparsity
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.01; epochs = 1200;
        optimizer = std::make_unique<Adam>(learning_rate);
    }
    Network dnn(sizes, activations, std::move(optimizer));
    // dnn.show();

    // --- Pruning Parameters ---
    int pruning_start_epoch = epochs / 5;    // Start pruning after initial convergence
    int pruning_end_epoch = epochs * 4 / 5;  // Stop pruning before the end
    int pruning_frequency = epochs / 20;     // Prune somewhat frequently
    real initial_sparsity = 0.0;
    real final_sparsity = 0.80; // Target 80% sparsity

    // Données XOR
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int num_samples = 4;
    const int input_dim = sizes.at(0); const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_xor_inputs("h_xor_inputs", num_samples, input_dim);
    HostView2D h_xor_outputs("h_xor_outputs", num_samples, output_dim);
    h_xor_inputs(0, 0) = 0.0; h_xor_inputs(0, 1) = 0.0; h_xor_outputs(0, 0) = 0.0;
    h_xor_inputs(1, 0) = 1.0; h_xor_inputs(1, 1) = 0.0; h_xor_outputs(1, 0) = 1.0;
    h_xor_inputs(2, 0) = 0.0; h_xor_inputs(2, 1) = 1.0; h_xor_outputs(2, 0) = 1.0;
    h_xor_inputs(3, 0) = 1.0; h_xor_inputs(3, 1) = 1.0; h_xor_outputs(3, 0) = 0.0;
    auto xor_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_inputs);
    auto xor_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_outputs);

    // Lancer l'entraînement générique
    train_network(dnn, xor_inputs, xor_outputs, epochs, batch_size,
                  pruning_start_epoch, pruning_end_epoch, pruning_frequency,
                  initial_sparsity, final_sparsity);

    // Tester les prédictions finales
    std::cout << "\n-- FINAL PREDICTIONS --" << std::endl;
    // ... (test logic remains the same) ...
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
    std::cout << "\n--- Sine Function Approx Training (Dynamic Sparse) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    std::map<int, int> sizes; sizes[0] = 1; sizes[1] = 64; sizes[2] = 64; sizes[3] = 1;
    std::vector<std::string> activations = {"relu", "relu", "linear"};

    real learning_rate; int epochs; int batch_size = 32; int num_samples = 4096; // More samples/bs
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.01; epochs = 800; // Adjusted for potentially harder task with pruning
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.001; epochs = 1000;
        optimizer = std::make_unique<Adam>(learning_rate);
    }
    Network dnn(sizes, activations, std::move(optimizer));

    // --- Pruning Parameters ---
    int pruning_start_epoch = epochs / 5;
    int pruning_end_epoch = epochs * 4 / 5;
    int pruning_frequency = epochs / 25; // Prune a bit more often maybe
    real initial_sparsity = 0.0;
    real final_sparsity = 0.90; // Target 90% sparsity for a regression task

    // Générer données Sinus
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0); const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_inputs("h_sine_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_sine_outputs", num_samples, output_dim);
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count()+1);
    std::uniform_real_distribution<real> distrib(-M_PI, M_PI);
    for(int i=0; i < num_samples; ++i) { real x = distrib(gen); h_inputs(i, 0) = x; h_outputs(i, 0) = std::sin(x); }
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    // Lancer l'entraînement générique
    train_network(dnn, train_inputs, train_outputs, epochs, batch_size,
                  pruning_start_epoch, pruning_end_epoch, pruning_frequency,
                  initial_sparsity, final_sparsity);

    // Tester prédictions finales
    std::cout << "\n-- FINAL PREDICTIONS (Sample) --" << std::endl;
    // ... (test logic same) ...
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


// Linear separation doesn't benefit much from hidden layers/pruning
// Keep it simple or skip pruning for this one.
void linear_sep_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- Linear Separation Training (No Pruning Applied) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

    std::map<int, int> sizes; sizes[0] = 2; sizes[1] = 1; // No hidden layers
    std::vector<std::string> activations = {"sigmoid"};

    real learning_rate; int epochs; int batch_size = 16; int num_samples = 2560;
    std::unique_ptr<Optimizer> optimizer;
     if (optimizer_choice == "sgd") { learning_rate = 0.1; epochs = 100; optimizer = std::make_unique<SGD>(learning_rate); }
     else { learning_rate = 0.01; epochs = 150; optimizer = std::make_unique<Adam>(learning_rate); }
    Network dnn(sizes, activations, std::move(optimizer));

    // --- NO Pruning Parameters ---
    int pruning_start_epoch = epochs; // Start after training ends (no pruning)
    int pruning_end_epoch = epochs;
    int pruning_frequency = 1;
    real initial_sparsity = 0.0; real final_sparsity = 0.0;

    // Générer données
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0); const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_inputs("h_linear_inputs", num_samples, input_dim); HostView2D h_outputs("h_linear_outputs", num_samples, output_dim);
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + 2);
    std::uniform_real_distribution<real> distrib(-1.0, 1.0); real margin = 0.1;
    int count_class0 = 0, count_class1 = 0;
    for(int i=0; i < num_samples; ++i) { /* ... generation logic ... */
        real x = distrib(gen); real y = distrib(gen);
        h_inputs(i, 0) = x; h_inputs(i, 1) = y;
        if (y < x - margin) { h_outputs(i, 0) = 0.0; count_class0++; }
        else if (y > x + margin) { h_outputs(i, 0) = 1.0; count_class1++; }
        else { i--; continue; }
    }
    std::cout << "Generated " << count_class0 << " Class 0 and " << count_class1 << " Class 1." << std::endl;
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    // Lancer l'entraînement générique (pruning désactivé par les paramètres)
    train_network(dnn, train_inputs, train_outputs, epochs, batch_size,
                  pruning_start_epoch, pruning_end_epoch, pruning_frequency,
                  initial_sparsity, final_sparsity);

    // Test final & Accuracy
    std::cout << "\n-- FINAL PREDICTIONS & ACCURACY --" << std::endl;
    // ... (accuracy test logic same) ...
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
        int predicted_class = std::round(h_prediction_result(0));
        int target_class = static_cast<int>(h_outputs(i, 0));
        if (predicted_class == target_class) correct_predictions++;
         if (i < 10) { /* print sample predictions */ }
    }
    real accuracy = static_cast<real>(correct_predictions) / num_samples;
    std::cout << "Final Average Cost: " << std::fixed << std::setprecision(8) << final_total_cost / num_samples << std::endl;
    std::cout << "Final Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%" << std::endl;

}


// --- Main (appelle les fonctions d'entraînement modifiées) ---
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        try {
            std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
            std::cout << "*** Using KokkosSparse::CrsMatrix with Dynamic Pruning ***" << std::endl;

            xor_train("adam");
            std::cout << "\n---------------------------\n" << std::endl;

            sine_train("adam");
            std::cout << "\n---------------------------\n" << std::endl;

            linear_sep_train("adam"); // Note: Pruning désactivé pour celui-ci

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl; Kokkos::finalize(); return 1;
        } catch (...) {
            std::cerr << "An unknown error occurred." << std::endl; Kokkos::finalize(); return 1;
        }
    }
    Kokkos::finalize();
    return 0;
}
