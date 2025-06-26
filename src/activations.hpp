#pragma once

#include "types.hpp"

// --- Classes Activation ---
class Activation {
public:
    virtual ~Activation() = default;
    virtual void apply(const View1D& input, View1D& output) const = 0;
    virtual void apply_derivative(const View1D& input_z, View1D& output_deriv) const = 0;
};

class LinearActivation : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override;
    void apply_derivative(const View1D& /*input_z*/, View1D& output_deriv) const override;
};

class RELU : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override;
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override;
};

class SIGMOID : public Activation {
public:
    KOKKOS_INLINE_FUNCTION real scalar_sigmoid(real x) const;
    void apply(const View1D& input, View1D& output) const override;
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override;
};

class TANH : public Activation {
public:
    KOKKOS_INLINE_FUNCTION real scalar_tanh(real x) const;
    void apply(const View1D& input, View1D& output) const override;
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override;
};

// Helper pour créer des activations par type
std::unique_ptr<Activation> create_activation(const std::string& type); 