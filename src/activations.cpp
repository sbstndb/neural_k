#include "activations.hpp"

void LinearActivation::apply(const View1D& input, View1D& output) const {
    Kokkos::deep_copy(output, input);
}

void LinearActivation::apply_derivative(const View1D& /*input_z*/, View1D& output_deriv) const {
    Kokkos::deep_copy(output_deriv, 1.0f);
}

void RELU::apply(const View1D& input, View1D& output) const {
    const int size = input.extent_int(0);
    Kokkos::parallel_for("relu_apply", size, KOKKOS_LAMBDA(const int i) {
        output(i) = (input(i) > 0.0f) ? input(i) : 0.0f;
    });
}

void RELU::apply_derivative(const View1D& input_z, View1D& output_deriv) const {
    const int size = input_z.extent_int(0);
    Kokkos::parallel_for("relu_deriv", size, KOKKOS_LAMBDA(const int i) {
        output_deriv(i) = (input_z(i) > 0.0f) ? 1.0f : 0.0f;
    });
}

KOKKOS_INLINE_FUNCTION real SIGMOID::scalar_sigmoid(real x) const {
    x = Kokkos::max(-30.0f, Kokkos::min(30.0f, x));
    return 1.0f / (1.0f + Kokkos::exp(-x));
}

void SIGMOID::apply(const View1D& input, View1D& output) const {
    const int size = input.extent_int(0);
    Kokkos::parallel_for("sigmoid_apply", size, KOKKOS_LAMBDA(const int i) {
        output(i) = scalar_sigmoid(input(i));
    });
}

void SIGMOID::apply_derivative(const View1D& input_z, View1D& output_deriv) const {
    const int size = input_z.extent_int(0);
    Kokkos::parallel_for("sigmoid_deriv", size, KOKKOS_LAMBDA(const int i) {
        real s = scalar_sigmoid(input_z(i));
        output_deriv(i) = s * (1.0f - s);
    });
}

KOKKOS_INLINE_FUNCTION real TANH::scalar_tanh(real x) const {
    return Kokkos::tanh(x);
}

void TANH::apply(const View1D& input, View1D& output) const {
    const int size = input.extent_int(0);
    Kokkos::parallel_for("tanh_apply", size, KOKKOS_LAMBDA(const int i) {
        output(i) = scalar_tanh(input(i));
    });
}

void TANH::apply_derivative(const View1D& input_z, View1D& output_deriv) const {
    const int size = input_z.extent_int(0);
    Kokkos::parallel_for("tanh_deriv", size, KOKKOS_LAMBDA(const int i) {
        real t = scalar_tanh(input_z(i));
        output_deriv(i) = 1.0f - t * t;
    });
}

std::unique_ptr<Activation> create_activation(const std::string& type) {
    if (type == "relu") return std::make_unique<RELU>();
    if (type == "sigmoid") return std::make_unique<SIGMOID>();
    if (type == "tanh") return std::make_unique<TANH>();
    if (type == "linear") return std::make_unique<LinearActivation>();
    throw std::runtime_error("Unknown activation type: " + type);
} 