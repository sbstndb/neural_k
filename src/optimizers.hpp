#pragma once

#include "types.hpp"

class Optimizer {
public:
    real learning_rate;

    Optimizer(real lr);
    virtual ~Optimizer() = default;

    virtual void update(SparseMatrixType& weights, View1D biases,
                        const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
                        int batch_size) = 0;

    void set_learning_rate(real lr);
    real get_learning_rate() const;

    virtual std::string get_info() const;
};

class SGD : public Optimizer {
public:
    SGD(real lr = 0.1);

    void update(SparseMatrixType& weights, View1D biases,
                const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override;

    std::string get_info() const override;
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

        ParameterState(size_t size);
        ParameterState(size_t rows, size_t cols);
        ParameterState() = default;
        ParameterState(ParameterState&&) = default;
        ParameterState& operator=(ParameterState&&) = default;
        ParameterState(const ParameterState&) = delete;
        ParameterState& operator=(const ParameterState&) = delete;
    };

    std::map<real*, ParameterState> state_map_1d; // For biases
    std::map<real*, ParameterState> state_map_sparse; // For sparse weights

public:
    Adam(real lr = 0.001, real b1 = 0.9, real b2 = 0.999, real eps = 1e-8);

    void update(SparseMatrixType& weights, View1D biases,
                const SparseMatrixType& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override;

    std::string get_info() const override;
    virtual ~Adam() override = default;
}; 