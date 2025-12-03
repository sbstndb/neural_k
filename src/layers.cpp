#include "layers.hpp"

// Helper function to initialize sparse graph structure (extracted from constructor for CUDA compatibility)
// CUDA does not allow extended lambdas in constructors
namespace {
    void init_sparse_graph(RowMapType& row_map, EntriesType& entries, int layer_size, int input_size, size_t nnz) {
        Kokkos::parallel_for("create_dense_graph", layer_size, KOKKOS_LAMBDA(const int i) {
            row_map(i) = static_cast<Offset>(i) * input_size;
            for (int j = 0; j < input_size; ++j) {
                Ordinal k = static_cast<Ordinal>(i) * input_size + j;
                entries(k) = j;
            }
        });
        // Set the last element of row_map separately
        auto h_row_map = Kokkos::create_mirror_view(row_map);
        Kokkos::deep_copy(h_row_map, row_map);
        h_row_map(layer_size) = nnz;
        Kokkos::deep_copy(row_map, h_row_map);
        Kokkos::fence();
    }
}

// --- Layer implementation ---
Layer::Layer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
    input_size(_input_size),
    layer_size(_layer_size),
    activation(std::move(act_func))
{
    if (input_size < 0 || layer_size <= 0) {
         throw std::runtime_error("Layer sizes must be non-negative, layer_size > 0.");
    }

    // Allocate dense views first
    biases = View1D("biases", layer_size);
    z = View1D("z", layer_size);
    delta = View1D("delta", layer_size);
    d_biases = View1D("d_biases", layer_size);
    d_biases_sum = View1D("d_biases_sum", layer_size);
    tmp_deriv = View1D("tmp_deriv", layer_size);
    a = View1D("a", layer_size);

    // Initialize to zero
    Kokkos::deep_copy(biases, 0.0);
    Kokkos::deep_copy(z, 0.0);
    Kokkos::deep_copy(delta, 0.0);
    Kokkos::deep_copy(d_biases, 0.0);
    Kokkos::deep_copy(d_biases_sum, 0.0);
    Kokkos::deep_copy(tmp_deriv, 0.0);
    Kokkos::deep_copy(a, 0.0);

    // Initialize sparse matrices
    if (input_size > 0) {
        // Create dense graph structure
        size_t nnz = static_cast<size_t>(layer_size) * input_size;
        RowMapType row_map("row_map", layer_size + 1);
        EntriesType entries("entries", nnz);
        ValuesType w_values("w_values", nnz);
        ValuesType dw_values("dw_values", nnz);
        ValuesType dw_sum_values("dw_sum_values", nnz);

        // Create graph structure using helper function (CUDA compatible)
        init_sparse_graph(row_map, entries, layer_size, input_size, nnz);

        // Initialize weights with Xavier
        Kokkos::Random_XorShift64_Pool<> rand_pool(std::chrono::high_resolution_clock::now().time_since_epoch().count() + reinterpret_cast<uintptr_t>(this));
        real limit = (input_size + layer_size > 0) ? std::sqrt(6.0f / (input_size + layer_size)) : 1.0f;
        Kokkos::fill_random(w_values, rand_pool, static_cast<real>(-limit), static_cast<real>(limit));

        Kokkos::deep_copy(dw_values, 0.0);
        Kokkos::deep_copy(dw_sum_values, 0.0);

        GraphType graph(entries, row_map);
        weights = SparseMatrixType("weights", input_size, w_values, graph);
        d_weights = SparseMatrixType("d_weights", input_size, dw_values, graph);
        d_weights_sum = SparseMatrixType("d_weights_sum", input_size, dw_sum_values, graph);
    } else {
        // Empty matrices for input layer
        RowMapType row_map("row_map_empty", layer_size + 1);
        EntriesType entries("entries_empty", 0);
        ValuesType w_values("w_values_empty", 0);
        ValuesType dw_values("dw_values_empty", 0);
        ValuesType dw_sum_values("dw_sum_values_empty", 0);
        Kokkos::deep_copy(row_map, 0);

        GraphType graph(entries, row_map);
        weights = SparseMatrixType("weights_empty", 0, w_values, graph);
        d_weights = SparseMatrixType("d_weights_empty", 0, dw_values, graph);
        d_weights_sum = SparseMatrixType("d_weights_sum_empty", 0, dw_sum_values, graph);
    }
}

Layer::Layer(int _layer_size) :
    Layer(0, _layer_size, nullptr) // Delegate to the main constructor
{
     // 'a' is allocated by delegated constructor. Sparse matrices are empty.
}

void Layer::forward(const View1D& prev_layer_a) {
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

    // Use Sparse Matrix-Vector Multiply (SpMV)
    KokkosSparse::spmv("N", 1.0, weights, prev_layer_a, 0.0, z);

    // Add biases
    if (biases.extent(0) != static_cast<size_t>(layer_size)) {
         throw std::runtime_error("Bias dimension mismatch in forward pass.");
    }
    Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) {
        z(i) += biases(i);
    });

    // 2. a = activation(z)
    if (a.extent(0) != static_cast<size_t>(layer_size)) {
         throw std::runtime_error("Activation output 'a' dimension mismatch in forward pass.");
    }
    if (activation) {
        activation->apply(z, a);
    } else {
        Kokkos::deep_copy(a, z);
    }
}

void Layer::compute_gradients(const Layer& next_layer, const View1D& prev_layer_a) {
     if (input_size == 0) return; // InputLayer has no gradients

     // Check dimensions
     if (next_layer.weights.numCols() != static_cast<size_t>(layer_size) ||
         next_layer.delta.extent(0) != next_layer.weights.numRows()) {
         throw std::runtime_error("Dimension mismatch between current layer and next layer for gradient computation.");
     }
      if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
         throw std::runtime_error("Dimension mismatch for prev_layer_a in gradient computation.");
     }

    // 1. Calculate f'(z) -> stored in tmp_deriv
    if (activation) {
         activation->apply_derivative(z, tmp_deriv);
    } else {
        Kokkos::deep_copy(tmp_deriv, 1.0f);
    }

    // 2. Calculate delta for this layer
    View1D delta_prop("delta_prop", layer_size);
    KokkosSparse::spmv("T", 1.0, next_layer.weights, next_layer.delta, 0.0, delta_prop);

    // Element-wise product (Hadamard product)
     Kokkos::parallel_for("hadamard_delta_deriv", layer_size, KOKKOS_LAMBDA(int i) {
        delta(i) = delta_prop(i) * tmp_deriv(i);
    });

    // 3. Calculate INSTANTANEOUS gradients
    Kokkos::deep_copy(d_biases, delta);

    // d_weights = delta * prev_layer_a^T (outer product, store in sparse values)
    auto dw_vals = d_weights.values;
    auto graph = d_weights.graph;
    Kokkos::parallel_for("compute_d_weights_sparse", layer_size, KOKKOS_LAMBDA(const int i) {
         const auto row_start = graph.row_map(i);
         const auto row_end = graph.row_map(i+1);
         for (auto k = row_start; k < row_end; ++k) {
             const int j = graph.entries(k);
             dw_vals(k) = delta(i) * prev_layer_a(j);
         }
     });

    // 4. ACCUMULATE gradients
    auto dw_sum_vals = d_weights_sum.values;
    const int nnz = dw_vals.extent_int(0);
    Kokkos::parallel_for("accumulate_weight_gradients_sparse", nnz, KOKKOS_LAMBDA (const int k) {
             Kokkos::atomic_add(&dw_sum_vals(k), dw_vals(k));
     });

     Kokkos::parallel_for("accumulate_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
     });
}

void Layer::zero_accumulated_gradients() {
     if (input_size == 0 && layer_size == 0) return;

     if (d_biases_sum.data() != nullptr) Kokkos::deep_copy(d_biases_sum, 0.0);

     if (d_weights_sum.values.data() != nullptr && d_weights_sum.nnz() > 0) {
          Kokkos::deep_copy(d_weights_sum.values, 0.0);
     }
}

void Layer::show() const {
    if (input_size == 0 && activation == nullptr) {
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

    // Display Sparse Weights Info
    if (weights.values.data() != nullptr && weights.numRows() > 0 && weights.numCols() > 0) {
        std::cout << "  Weights (Sparse): " << weights.numRows() << "x" << weights.numCols()
                  << ", NNZ: " << weights.nnz() << std::endl;

        const int max_print_vals = 10;
        auto h_values = Kokkos::create_mirror_view(weights.values);
        Kokkos::deep_copy(h_values, weights.values);
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

    // Display Biases Info
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

// --- InputLayer implementation ---
InputLayer::InputLayer(int _layer_size) : Layer(_layer_size) {}

void InputLayer::set_input(const View1D& input_data) {
     if (input_data.extent(0) != static_cast<size_t>(layer_size)) {
         throw std::runtime_error("Input data size mismatch for InputLayer: Expected "
            + std::to_string(layer_size) + ", Got " + std::to_string(input_data.extent(0)));
     }
     if (a.data() == nullptr || a.extent(0) != static_cast<size_t>(layer_size)) {
          throw std::runtime_error("InputLayer 'a' view is not correctly initialized or sized.");
     }
     Kokkos::deep_copy(a, input_data);
 }

void InputLayer::forward(const View1D& /*prev_layer_a*/) { }

void InputLayer::compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) { }

// --- OutputLayer implementation ---
OutputLayer::OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
    Layer(_input_size, _layer_size, std::move(act_func)) {}

void OutputLayer::compute_gradients(const View1D& target, const View1D& prev_layer_a) {
    // Check dimensions
     if (target.extent(0) != static_cast<size_t>(layer_size)) {
         throw std::runtime_error("Target size mismatch for OutputLayer.");
     }
     if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
         throw std::runtime_error("prev_layer_a size mismatch in OutputLayer gradient.");
     }

    // 1. Calculate f'(z_L) -> stored in tmp_deriv
    if (activation) {
        activation->apply_derivative(z, tmp_deriv);
    } else {
        Kokkos::deep_copy(tmp_deriv, 1.0f);
    }

    // 2. Calculate delta for the output layer
    Kokkos::parallel_for("compute_output_delta", layer_size, KOKKOS_LAMBDA(int i) {
        delta(i) = (a(i) - target(i)) * tmp_deriv(i);
    });

   // 3. Calculate INSTANTANEOUS gradients
    Kokkos::deep_copy(d_biases, delta);

    // d_weights = delta * prev_layer_a^T (outer product, store in sparse values)
    auto dw_vals = d_weights.values;
    auto graph = d_weights.graph;
     Kokkos::parallel_for("compute_output_d_weights_sparse", layer_size, KOKKOS_LAMBDA(const int i) {
         const auto row_start = graph.row_map(i);
         const auto row_end = graph.row_map(i+1);
         for (auto k = row_start; k < row_end; ++k) {
             const int j = graph.entries(k);
             dw_vals(k) = delta(i) * prev_layer_a(j);
         }
    });

    // 4. ACCUMULATE gradients into sum views
    auto dw_sum_vals = d_weights_sum.values;
    const int nnz = dw_vals.extent_int(0);
    Kokkos::parallel_for("accumulate_output_weight_gradients_sparse", nnz, KOKKOS_LAMBDA (const int k) {
        Kokkos::atomic_add(&dw_sum_vals(k), dw_vals(k));
    });

     Kokkos::parallel_for("accumulate_output_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
         Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
    });
}

void OutputLayer::compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) {
     throw std::logic_error("OutputLayer::compute_gradients should be called with target, not next_layer.");
} 