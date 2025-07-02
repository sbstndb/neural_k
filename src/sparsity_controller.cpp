#include "sparsity_controller.hpp"
#include "network.hpp"
#include "layers.hpp"
#include <Kokkos_Core.hpp>

void ThresholdSparsityStrategy::apply(Network& net) {
    // Parcourt chaque couche entraînable et met à zéro les petits poids
    net.iterate_trainable_layers([&](Layer& layer, bool /*is_output*/, size_t /*idx*/) {
        auto w_vals = layer.weights.values;
        int nnz = w_vals.extent_int(0);
        if (nnz == 0) return;
        const real thr = threshold;
        Kokkos::parallel_for("threshold_sparsity", nnz, KOKKOS_LAMBDA(const int k) {
            if (Kokkos::abs(w_vals(k)) < thr) {
                w_vals(k) = 0.0;
            }
        });
    });
} 