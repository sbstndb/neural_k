#include "optimizers.hpp"

// --- Optimizer base class ---
Optimizer::Optimizer(real lr) : learning_rate(lr) {}

void Optimizer::set_learning_rate(real lr) { learning_rate = lr; }

real Optimizer::get_learning_rate() const { return learning_rate; }

std::string Optimizer::get_info() const {
    return "Optimizer(LR=" + std::to_string(learning_rate) + ")";
}

// --- SGD implementation ---
SGD::SGD(real lr) : Optimizer(lr) {}

void SGD::update(SparseMatrixType& weights, View1D biases,
            const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
            int batch_size) {

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

std::string SGD::get_info() const {
    return "SGD(LR=" + std::to_string(learning_rate) + ")";
}

// --- Adam ParameterState implementation ---
Adam::ParameterState::ParameterState(size_t size) : t(0) {
    if (size > std::numeric_limits<int>::max()) {
         throw std::runtime_error("Adam state bias size exceeds limits.");
    }
    m_1d = View1D("adam_m1d", size);
    v_1d = View1D("adam_v1d", size);
    Kokkos::deep_copy(m_1d, 0.0);
    Kokkos::deep_copy(v_1d, 0.0);
}

Adam::ParameterState::ParameterState(size_t rows, size_t cols) : t(0) {
     if (rows > std::numeric_limits<int>::max() || cols > std::numeric_limits<int>::max()) {
         throw std::runtime_error("Adam state weight dimensions exceed limits.");
    }
    // Allocate dense views for moments, matching the *logical* dense dimensions
    m_2d = Kokkos::View<real**>("adam_m2d", rows, cols);
    v_2d = Kokkos::View<real**>("adam_v2d", rows, cols);
    Kokkos::deep_copy(m_2d, 0.0);
    Kokkos::deep_copy(v_2d, 0.0);
}

// --- Adam implementation ---
Adam::Adam(real lr, real b1, real b2, real eps)
    : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps) {
    if (lr <= 0 || b1 < 0 || b1 >= 1 || b2 < 0 || b2 >= 1 || eps <= 0) {
        throw std::runtime_error("Invalid Adam hyperparameters.");
    }
}

void Adam::update(SparseMatrixType& weights, View1D biases,
            const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
            int batch_size) {

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
}

std::string Adam::get_info() const {
     return "Adam(LR=" + std::to_string(learning_rate) +
            ", beta1=" + std::to_string(beta1) +
            ", beta2=" + std::to_string(beta2) +
            ", epsilon=" + std::to_string(epsilon) + ")";
} 