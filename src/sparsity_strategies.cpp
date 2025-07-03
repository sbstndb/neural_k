#include "sparsity_strategies.hpp"
#include "network.hpp"
#include "layers.hpp"
#include <Kokkos_Core.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include <cmath>
#include <iomanip>

L1RegularizationStrategy::L1RegularizationStrategy(real lambda) : lambda_(lambda) {}

void L1RegularizationStrategy::apply(Network& net) {
    std::cout << "\n=== APPLICATION DE LA RÉGULARISATION L1 ===" << std::endl;
    std::cout << "Coefficient lambda: " << lambda_ << std::endl;

    // Couches cachées
    for (auto& layer : net.hidden_layers) {
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            auto dw_sum_vals = layer.d_weights_sum.values;
            int nnz = w_vals.extent_int(0);
            const real lam = lambda_;
            Kokkos::parallel_for("apply_l1_regularization_hidden", nnz,
                KOKKOS_LAMBDA(const int k) {
                    real sign_w = (w_vals(k) > 0) ? 1.0 : ((w_vals(k) < 0) ? -1.0 : 0.0);
                    dw_sum_vals(k) += lam * sign_w;
                });
        }
    }

    // Couche de sortie
    if (net.output_layer.input_size > 0) {
        auto w_vals = net.output_layer.weights.values;
        auto dw_sum_vals = net.output_layer.d_weights_sum.values;
        int nnz = w_vals.extent_int(0);
        const real lam = lambda_;
        Kokkos::parallel_for("apply_l1_regularization_output", nnz,
            KOKKOS_LAMBDA(const int k) {
                real sign_w = (w_vals(k) > 0) ? 1.0 : ((w_vals(k) < 0) ? -1.0 : 0.0);
                dw_sum_vals(k) += lam * sign_w;
            });
    }

    std::cout << "Régularisation L1 appliquée avec lambda = " << lambda_ << std::endl;
    std::cout << "==========================================\n" << std::endl;
}

// === StructuralPruningStrategy ===
StructuralPruningStrategy::StructuralPruningStrategy(real threshold) : threshold_(threshold) {}

void StructuralPruningStrategy::apply(Network& net) {
    std::cout << "\n=== PRUNING STRUCTUREL ===" << std::endl;
    std::cout << "Seuil pour suppression de neurones: " << threshold_ << std::endl;

    for (size_t layer_idx = 0; layer_idx < net.hidden_layers.size(); ++layer_idx) {
        auto& layer = net.hidden_layers[layer_idx];
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            int input_size = layer.input_size;
            int output_size = layer.layer_size;

            // Calcule la norme L2 de chaque neurone
            std::vector<real> neuron_norms(output_size, 0.0);
            for (int neuron = 0; neuron < output_size; ++neuron) {
                real norm = 0.0;
                for (int input = 0; input < input_size; ++input) {
                    int weight_idx = neuron * input_size + input;
                    if (weight_idx < w_vals.extent_int(0)) {
                        norm += w_vals(weight_idx) * w_vals(weight_idx);
                    }
                }
                neuron_norms[neuron] = std::sqrt(norm);
            }

            // Mise à zéro des neurones peu importants
            int neurons_removed = 0;
            for (int neuron = 0; neuron < output_size; ++neuron) {
                if (neuron_norms[neuron] < threshold_) {
                    for (int input = 0; input < input_size; ++input) {
                        int weight_idx = neuron * input_size + input;
                        if (weight_idx < w_vals.extent_int(0)) {
                            w_vals(weight_idx) = 0.0;
                        }
                    }
                    neurons_removed++;
                }
            }
            std::cout << "Couche " << (layer_idx + 1) << ": " << neurons_removed << "/" << output_size << " neurones supprimés" << std::endl;
        }
    }
    std::cout << "Pruning structurel terminé!" << std::endl;
    std::cout << "==========================\n" << std::endl;
}

// === SensitivityPruningStrategy ===
SensitivityPruningStrategy::SensitivityPruningStrategy(real threshold) : threshold_(threshold) {}

void SensitivityPruningStrategy::apply(Network& net) {
    std::cout << "\n=== PRUNING BASÉ SUR LA SENSIBILITÉ ===" << std::endl;
    std::cout << "Seuil de sensibilité: " << threshold_ << std::endl;

    // Couches cachées
    for (auto& layer : net.hidden_layers) {
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            auto dw_sum_vals = layer.d_weights_sum.values;
            int nnz = w_vals.extent_int(0);
            const real thr = threshold_;
            int weights_removed = 0;
            Kokkos::parallel_reduce("sensitivity_pruning_hidden", nnz,
                KOKKOS_LAMBDA(const int k, int& local_count) {
                    real sensitivity = Kokkos::abs(dw_sum_vals(k) * w_vals(k));
                    if (sensitivity < thr) {
                        w_vals(k) = 0.0;
                        dw_sum_vals(k) = 0.0;
                        local_count++;
                    }
                }, weights_removed);
            std::cout << "Couche cachée: " << weights_removed << "/" << nnz << " poids supprimés" << std::endl;
        }
    }

    // Couche de sortie
    if (net.output_layer.input_size > 0) {
        auto w_vals = net.output_layer.weights.values;
        auto dw_sum_vals = net.output_layer.d_weights_sum.values;
        int nnz = w_vals.extent_int(0);
        const real thr = threshold_;
        int weights_removed = 0;
        Kokkos::parallel_reduce("sensitivity_pruning_output", nnz,
            KOKKOS_LAMBDA(const int k, int& local_count) {
                real sensitivity = Kokkos::abs(dw_sum_vals(k) * w_vals(k));
                if (sensitivity < thr) {
                    w_vals(k) = 0.0;
                    dw_sum_vals(k) = 0.0;
                    local_count++;
                }
            }, weights_removed);
        std::cout << "Couche sortie: " << weights_removed << "/" << nnz << " poids supprimés" << std::endl;
    }
    std::cout << "Pruning par sensibilité terminé!" << std::endl;
    std::cout << "================================\n" << std::endl;
}

// === ImportanceBasedPruningStrategy ===
ImportanceBasedPruningStrategy::ImportanceBasedPruningStrategy(real sparsity_target) : sparsity_target_(sparsity_target) {}

void ImportanceBasedPruningStrategy::apply(Network& net) {
    std::cout << "\n=== PRUNING BASÉ SUR L'IMPORTANCE ===" << std::endl;
    std::cout << "Sparsité cible: " << std::fixed << std::setprecision(1) << (sparsity_target_ * 100.0) << "%" << std::endl;

    // Collecte de toutes les importances
    std::vector<real> importances;
    importances.reserve(10000); // estimation grossière

    auto accumulate_importance = [&](auto& w_vals) {
        auto h_weights = Kokkos::create_mirror_view(w_vals);
        Kokkos::deep_copy(h_weights, w_vals);
        Kokkos::fence();
        int nnz = w_vals.extent_int(0);
        for (int k = 0; k < nnz; ++k) {
            importances.push_back(Kokkos::abs(h_weights(k)));
        }
    };

    for (auto& layer : net.hidden_layers) {
        if (layer.input_size > 0) accumulate_importance(layer.weights.values);
    }
    if (net.output_layer.input_size > 0) accumulate_importance(net.output_layer.weights.values);

    if (importances.empty()) {
        std::cout << "Pas de poids à analyser." << std::endl;
        return;
    }

    std::vector<real> sorted_importances = importances;
    std::sort(sorted_importances.begin(), sorted_importances.end(), std::greater<real>());

    int weights_to_keep = static_cast<int>((1.0 - sparsity_target_) * sorted_importances.size());
    real threshold = sorted_importances[weights_to_keep];

    std::cout << "Seuil d'importance calculé: " << std::fixed << std::setprecision(6) << threshold << std::endl;

    // Réutilise la stratégie Threshold pour appliquer
    ThresholdSparsityStrategy(threshold).apply(net);

    std::cout << "Pruning basé sur l'importance terminé!" << std::endl;
    std::cout << "=====================================\n" << std::endl;
}

// === ProgressivePruningStrategy ===
ProgressivePruningStrategy::ProgressivePruningStrategy(real initial_threshold, real final_threshold, int epochs)
    : initial_threshold_(initial_threshold), final_threshold_(final_threshold), epochs_(epochs) {}

void ProgressivePruningStrategy::apply(Network& net) {
    std::cout << "\n=== PRUNING PROGRESSIF ===" << std::endl;
    std::cout << "Seuil initial: " << initial_threshold_ << " | Seuil final: " << final_threshold_ << " | Époques: " << epochs_ << std::endl;
    for (int epoch = 0; epoch < epochs_; ++epoch) {
        real current_threshold = initial_threshold_ + (final_threshold_ - initial_threshold_) * (static_cast<real>(epoch) / epochs_);
        std::cout << "Époque " << epoch << "/" << epochs_ << " - Seuil: " << std::fixed << std::setprecision(4) << current_threshold << std::endl;
        ThresholdSparsityStrategy(current_threshold).apply(net);
    }
    std::cout << "Pruning progressif terminé!" << std::endl;
    std::cout << "==========================\n" << std::endl;
}

// === LayerSpecificPruningStrategy ===
LayerSpecificPruningStrategy::LayerSpecificPruningStrategy(const std::vector<real>& thresholds) : thresholds_(thresholds) {}

void LayerSpecificPruningStrategy::apply(Network& net) {
    std::cout << "\n=== PRUNING SPÉCIFIQUE PAR COUCHE ===" << std::endl;

    // Couches cachées
    for (size_t i = 0; i < net.hidden_layers.size() && i < thresholds_.size(); ++i) {
        real thr = thresholds_[i];
        std::cout << "Couche cachée " << (i+1) << ": seuil = " << thr << std::endl;

        auto& layer = net.hidden_layers[i];
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            int nnz = w_vals.extent_int(0);
            const real thr_c = thr;
            int weights_removed = 0;
            Kokkos::parallel_reduce("layer_specific_pruning_hidden", nnz,
                KOKKOS_LAMBDA(const int k, int& local_count) {
                    if (Kokkos::abs(w_vals(k)) < thr_c) {
                        w_vals(k) = 0.0;
                        local_count++;
                    }
                }, weights_removed);
            std::cout << "  → " << weights_removed << "/" << nnz << " poids supprimés" << std::endl;
        }
    }

    // Couche de sortie
    if (net.output_layer.input_size > 0 && thresholds_.size() > net.hidden_layers.size()) {
        real thr = thresholds_[net.hidden_layers.size()];
        std::cout << "Couche sortie: seuil = " << thr << std::endl;
        auto w_vals = net.output_layer.weights.values;
        int nnz = w_vals.extent_int(0);
        const real thr_c = thr;
        int weights_removed = 0;
        Kokkos::parallel_reduce("layer_specific_pruning_output", nnz,
            KOKKOS_LAMBDA(const int k, int& local_count) {
                if (Kokkos::abs(w_vals(k)) < thr_c) {
                    w_vals(k) = 0.0;
                    local_count++;
                }
            }, weights_removed);
        std::cout << "  → " << weights_removed << "/" << nnz << " poids supprimés" << std::endl;
    }

    std::cout << "Pruning spécifique par couche terminé!" << std::endl;
    std::cout << "=====================================\n" << std::endl;
} 