#pragma once

#include "types.hpp"
#include "activations.hpp"



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
    Layer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func);

    // Constructor specifically for InputLayer (sets input_size to 0)
    Layer(int _layer_size);

    // --- Methods ---
    virtual void forward(const View1D& prev_layer_a);

    // Computes gradients for a hidden layer (MODIFIED for sparse weights)
    virtual void compute_gradients(const Layer& next_layer, const View1D& prev_layer_a);

    // Zero out the gradient sums (MODIFIED for sparse weights)
    void zero_accumulated_gradients();

    // Display layer info (MODIFIED for sparse weights)
    void show() const;

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
    InputLayer(int _layer_size);

    void set_input(const View1D& input_data);

    void forward(const View1D& /*prev_layer_a*/) override;
    void compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) override;
};

// --- Classe OutputLayer (hérite de Layer modifié) ---
class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func);

    // Special gradient computation for the output layer (MODIFIED for sparse weights)
    void compute_gradients(const View1D& target, const View1D& prev_layer_a);

    void compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) override;
}; 