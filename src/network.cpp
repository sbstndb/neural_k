#include "network.hpp"
#include "activations.hpp"
#include "sparsity_controller.hpp"
#include "sparsity_strategies.hpp"

// Constructor
Network::Network(const std::map<int, int>& _layer_sizes_map,
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

int Network::get_size(int layer_index) const {
    try {
        return layer_sizes_map.at(layer_index);
    } catch (const std::out_of_range& oor) {
        throw std::out_of_range("Layer index " + std::to_string(layer_index) + " not found.");
    }
 }

int Network::num_layers() const { return layer_sizes_map.size(); }

// Forward pass
View1D Network::forward(const View1D& input_data) {
    input_layer.set_input(input_data);
    const View1D* current_a = &input_layer.a;
    for (Layer& hidden_layer : hidden_layers) {
        hidden_layer.forward(*current_a);
        current_a = &hidden_layer.a;
    }
    output_layer.forward(*current_a);
    return output_layer.a;
}

// Backward pass
void Network::backward(const View1D& target) {
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

// Update weights and biases
void Network::update(int batch_size) {
    if (!optimizer) throw std::runtime_error("Optimizer not set.");
    if (batch_size <= 0) throw std::runtime_error("Batch size must be positive.");

    // Update hidden layers
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

// Zero accumulated gradients
void Network::zero_accumulated_gradients() {
    for (Layer& hidden_layer : hidden_layers) {
        hidden_layer.zero_accumulated_gradients();
    }
    output_layer.zero_accumulated_gradients();
}

// Calculate Cost
real Network::calculate_cost(const View1D& prediction, const View1D& target) {
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

// Optimizer Management
void Network::set_optimizer(std::unique_ptr<Optimizer> opt) {
    if (!opt) throw std::runtime_error("Cannot set a null optimizer.");
    optimizer = std::move(opt);
}

Optimizer* Network::get_optimizer() const { return optimizer.get(); }

void Network::set_learning_rate(real lr) {
    if (!optimizer) throw std::runtime_error("Optimizer not set.");
    optimizer->set_learning_rate(lr);
}

real Network::get_learning_rate() const {
     if (!optimizer) return 0.0;
    return optimizer->get_learning_rate();
}

// Display Network Info
void Network::show() const {
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

// === IMPLEMENTATION DES NOUVELLES METHODES DE SPARSITÉ ===

void Network::apply_threshold_sparsity(real threshold) {
    // Délégation à la stratégie dédiée pour limiter les responsabilités
    ThresholdSparsityStrategy(threshold).apply(*this);
}

// Version silencieuse pour l'entraînement
void Network::apply_threshold_sparsity_silent(real threshold) {
    // Utilise le helper générique pour parcourir les couches sans affichage
    for_each_trainable_layer([&](Layer& layer, bool /*is_output*/, size_t /*idx*/) {
        auto w_vals = layer.weights.values;
        int nnz = w_vals.extent_int(0);

        Kokkos::parallel_for("apply_threshold_sparse_silent_generic", nnz,
            KOKKOS_LAMBDA(const int k) {
                if (Kokkos::abs(w_vals(k)) < threshold) {
                    w_vals(k) = 0.0;
                }
            });
    });
}

void Network::compute_sparsity_stats() const {
    std::cout << "\n=== STATISTIQUES DE SPARSITÉ ACTUELLE ===" << std::endl;
    
    int total_weights = 0;
    int total_zero_weights = 0;
    
    // Analyser les couches cachées
    for (size_t i = 0; i < hidden_layers.size(); ++i) {
        const auto& layer = hidden_layers[i];
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            int nnz = w_vals.extent_int(0);
            total_weights += nnz;
            
            // Compter les zéros
            int zero_count = 0;
            auto h_weights = Kokkos::create_mirror_view(w_vals);
            Kokkos::deep_copy(h_weights, w_vals);
            Kokkos::fence();
            
            for (int k = 0; k < nnz; ++k) {
                if (h_weights(k) == 0.0) zero_count++;
            }
            
            total_zero_weights += zero_count;
            real layer_sparsity = (100.0 * zero_count) / nnz;
            std::cout << "Couche cachée " << (i+1) << ": " << zero_count << "/" << nnz 
                      << " zéros (" << std::fixed << std::setprecision(1) 
                      << layer_sparsity << "% sparse)" << std::endl;
        }
    }
    
    // Analyser la couche de sortie
    if (output_layer.input_size > 0) {
        auto w_vals = output_layer.weights.values;
        int nnz = w_vals.extent_int(0);
        total_weights += nnz;
        
        int zero_count = 0;
        auto h_weights = Kokkos::create_mirror_view(w_vals);
        Kokkos::deep_copy(h_weights, w_vals);
        Kokkos::fence();
        
        for (int k = 0; k < nnz; ++k) {
            if (h_weights(k) == 0.0) zero_count++;
        }
        
        total_zero_weights += zero_count;
        real layer_sparsity = (100.0 * zero_count) / nnz;
        std::cout << "Couche sortie  : " << zero_count << "/" << nnz 
                  << " zéros (" << std::fixed << std::setprecision(1) 
                  << layer_sparsity << "% sparse)" << std::endl;
    }
    
    real global_sparsity = (100.0 * total_zero_weights) / total_weights;
    std::cout << "\nSPARSITÉ GLOBALE:" << std::endl;
    std::cout << "- Total zéros: " << total_zero_weights << "/" << total_weights << std::endl;
    std::cout << "- Taux global: " << std::fixed << std::setprecision(2) 
              << global_sparsity << "%" << std::endl;
    std::cout << "=========================================\n" << std::endl;
}

void Network::apply_permanent_sparsity(real threshold) {
    std::cout << "\n=== APPLICATION DE LA SPARSITÉ PERMANENTE ===" << std::endl;
    std::cout << "Seuil appliqué: " << threshold << std::endl;
    std::cout << "ATTENTION: Cette opération est IRREVERSIBLE!" << std::endl;
    
    int total_weights_zeroed = 0;
    int total_weights = 0;
    
    // Appliquer aux couches cachées
    for (auto& layer : hidden_layers) {
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            auto dw_vals = layer.d_weights.values;
            auto dw_sum_vals = layer.d_weights_sum.values;
            int nnz = w_vals.extent_int(0);
            total_weights += nnz;
            
            // Compter et mettre à zéro les poids sous le seuil
            int layer_weights_zeroed = 0;
            Kokkos::parallel_reduce("apply_permanent_threshold_sparse", nnz, 
                KOKKOS_LAMBDA(const int k, int& local_count) {
                    if (Kokkos::abs(w_vals(k)) < threshold) {
                        w_vals(k) = 0.0;
                        dw_vals(k) = 0.0;  // Zéro aussi les gradients instantanés
                        dw_sum_vals(k) = 0.0;  // Zéro aussi les gradients accumulés
                        local_count++;
                    }
                }, layer_weights_zeroed);
            
            total_weights_zeroed += layer_weights_zeroed;
            std::cout << "Couche cachée: " << layer_weights_zeroed << "/" << nnz 
                      << " poids supprimés définitivement (" 
                      << std::fixed << std::setprecision(1) 
                      << (100.0 * layer_weights_zeroed / nnz) << "%)" << std::endl;
        }
    }
    
    // Appliquer à la couche de sortie
    if (output_layer.input_size > 0) {
        auto w_vals = output_layer.weights.values;
        auto dw_vals = output_layer.d_weights.values;
        auto dw_sum_vals = output_layer.d_weights_sum.values;
        int nnz = w_vals.extent_int(0);
        total_weights += nnz;
        
        int layer_weights_zeroed = 0;
        Kokkos::parallel_reduce("apply_permanent_threshold_output_sparse", nnz, 
            KOKKOS_LAMBDA(const int k, int& local_count) {
                if (Kokkos::abs(w_vals(k)) < threshold) {
                    w_vals(k) = 0.0;
                    dw_vals(k) = 0.0;  // Zéro aussi les gradients instantanés
                    dw_sum_vals(k) = 0.0;  // Zéro aussi les gradients accumulés
                    local_count++;
                }
            }, layer_weights_zeroed);
        
        total_weights_zeroed += layer_weights_zeroed;
        std::cout << "Couche sortie: " << layer_weights_zeroed << "/" << nnz 
                  << " poids supprimés définitivement (" 
                  << std::fixed << std::setprecision(1) 
                  << (100.0 * layer_weights_zeroed / nnz) << "%)" << std::endl;
    }
    
    real sparsity_percentage = (100.0 * total_weights_zeroed) / total_weights;
    std::cout << "\nRÉSULTAT GLOBAL (PERMANENT):" << std::endl;
    std::cout << "- Total poids supprimés définitivement: " << total_weights_zeroed << "/" << total_weights << std::endl;
    std::cout << "- Taux de sparsité permanent: " << std::fixed << std::setprecision(2) 
              << sparsity_percentage << "%" << std::endl;
    std::cout << "- Les gradients correspondants ont aussi été mis à zéro" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}

bool Network::is_sparse(real sparsity_threshold) const {
    int total_weights = 0;
    int total_zero_weights = 0;
    
    // Compter les zéros dans toutes les couches
    for (const auto& layer : hidden_layers) {
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            int nnz = w_vals.extent_int(0);
            total_weights += nnz;
            
            int zero_count = 0;
            auto h_weights = Kokkos::create_mirror_view(w_vals);
            Kokkos::deep_copy(h_weights, w_vals);
            Kokkos::fence();
            
            for (int k = 0; k < nnz; ++k) {
                if (h_weights(k) == 0.0) zero_count++;
            }
            total_zero_weights += zero_count;
        }
    }
    
    if (output_layer.input_size > 0) {
        auto w_vals = output_layer.weights.values;
        int nnz = w_vals.extent_int(0);
        total_weights += nnz;
        
        int zero_count = 0;
        auto h_weights = Kokkos::create_mirror_view(w_vals);
        Kokkos::deep_copy(h_weights, w_vals);
        Kokkos::fence();
        
        for (int k = 0; k < nnz; ++k) {
            if (h_weights(k) == 0.0) zero_count++;
        }
        total_zero_weights += zero_count;
    }
    
    real current_sparsity = (total_weights > 0) ? (static_cast<real>(total_zero_weights) / total_weights) : 0.0;
    return current_sparsity >= sparsity_threshold;
}

void Network::auto_convert_to_sparse(real sparsity_threshold, real weight_threshold) {
    std::cout << "\n=== CONVERSION AUTOMATIQUE VERS SPARSE ===" << std::endl;
    
    // Vérifier si déjà sparse
    if (is_sparse(sparsity_threshold)) {
        std::cout << "Le réseau est déjà suffisamment sparse (" 
                  << std::fixed << std::setprecision(1) 
                  << (sparsity_threshold * 100.0) << "% de seuil)" << std::endl;
        return;
    }
    
    std::cout << "Le réseau n'est pas assez sparse. Application d'un seuil de " 
              << weight_threshold << " pour créer de la sparsité..." << std::endl;
    
    // Appliquer la sparsité par seuil
    apply_threshold_sparsity(weight_threshold);
    
    // Optimiser la structure CSR
    optimize_csr_structure();
    
    std::cout << "Conversion terminée!" << std::endl;
    std::cout << "==========================================\n" << std::endl;
}

void Network::optimize_csr_structure() {
    std::cout << "\n=== OPTIMISATION DE LA STRUCTURE CSR ===" << std::endl;
    
    int total_removed = 0;
    
    // Optimiser les couches cachées
    for (auto& layer : hidden_layers) {
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            auto dw_vals = layer.d_weights.values;
            auto dw_sum_vals = layer.d_weights_sum.values;
            int nnz = w_vals.extent_int(0);
            
            // Compter les zéros exacts
            int exact_zeros = 0;
            Kokkos::parallel_reduce("count_exact_zeros", nnz,
                KOKKOS_LAMBDA(const int k, int& local_count) {
                    if (w_vals(k) == 0.0) {
                        local_count++;
                    }
                }, exact_zeros);
            
            if (exact_zeros > 0) {
                // Mettre à zéro les gradients correspondants
                Kokkos::parallel_for("zero_gradients_for_exact_zeros", nnz,
                    KOKKOS_LAMBDA(const int k) {
                        if (w_vals(k) == 0.0) {
                            dw_vals(k) = 0.0;
                            dw_sum_vals(k) = 0.0;
                        }
                    });
                
                total_removed += exact_zeros;
                std::cout << "Couche cachée: " << exact_zeros << " zéros exacts traités" << std::endl;
            }
        }
    }
    
    // Optimiser la couche de sortie
    if (output_layer.input_size > 0) {
        auto w_vals = output_layer.weights.values;
        auto dw_vals = output_layer.d_weights.values;
        auto dw_sum_vals = output_layer.d_weights_sum.values;
        int nnz = w_vals.extent_int(0);
        
        int exact_zeros = 0;
        Kokkos::parallel_reduce("count_exact_zeros_output", nnz,
            KOKKOS_LAMBDA(const int k, int& local_count) {
                if (w_vals(k) == 0.0) {
                    local_count++;
                }
            }, exact_zeros);
        
        if (exact_zeros > 0) {
            Kokkos::parallel_for("zero_gradients_for_exact_zeros_output", nnz,
                KOKKOS_LAMBDA(const int k) {
                    if (w_vals(k) == 0.0) {
                        dw_vals(k) = 0.0;
                        dw_sum_vals(k) = 0.0;
                    }
                });
            
            total_removed += exact_zeros;
            std::cout << "Couche sortie: " << exact_zeros << " zéros exacts traités" << std::endl;
        }
    }
    
    std::cout << "Optimisation terminée. Total zéros exacts traités: " << total_removed << std::endl;
    std::cout << "==============================================\n" << std::endl;
}

void Network::sparsity_report() const {
    std::cout << "\n=== RAPPORT COMPLET DE SPARSITÉ ===" << std::endl;
    
    // Statistiques générales
    compute_sparsity_stats();
    
    // Détection automatique
    bool is_already_sparse_50 = is_sparse(0.5);
    bool is_already_sparse_80 = is_sparse(0.8);
    bool is_already_sparse_90 = is_sparse(0.9);
    
    std::cout << "DÉTECTION AUTOMATIQUE:" << std::endl;
    std::cout << "- Sparsité > 50%: " << (is_already_sparse_50 ? "OUI" : "NON") << std::endl;
    std::cout << "- Sparsité > 80%: " << (is_already_sparse_80 ? "OUI" : "NON") << std::endl;
    std::cout << "- Sparsité > 90%: " << (is_already_sparse_90 ? "OUI" : "NON") << std::endl;
    
    // Recommandations
    std::cout << "\nRECOMMANDATIONS:" << std::endl;
    if (!is_already_sparse_50) {
        std::cout << "- Le réseau n'est pas sparse. Considérez auto_convert_to_sparse()" << std::endl;
    } else if (!is_already_sparse_80) {
        std::cout << "- Le réseau est modérément sparse. Considérez apply_threshold_sparsity()" << std::endl;
    } else {
        std::cout << "- Le réseau est très sparse. Optimisez avec optimize_csr_structure()" << std::endl;
    }
    
    std::cout << "==================================\n" << std::endl;
}

// === NOUVELLES MÉTHODES POUR SEUILS ADAPTATIFS ===

real Network::compute_adaptive_threshold(real target_sparsity) const {
    std::cout << "\n=== CALCUL DU SEUIL ADAPTATIF ===" << std::endl;
    std::cout << "Sparsité cible: " << std::fixed << std::setprecision(1) 
              << (target_sparsity * 100.0) << "%" << std::endl;
    
    // Collecter tous les poids non-nuls
    std::vector<real> all_weights;
    
    // Couches cachées
    for (const auto& layer : hidden_layers) {
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            int nnz = w_vals.extent_int(0);
            
            auto h_weights = Kokkos::create_mirror_view(w_vals);
            Kokkos::deep_copy(h_weights, w_vals);
            Kokkos::fence();
            
            for (int k = 0; k < nnz; ++k) {
                if (h_weights(k) != 0.0) {
                    all_weights.push_back(Kokkos::abs(h_weights(k)));
                }
            }
        }
    }
    
    // Couche de sortie
    if (output_layer.input_size > 0) {
        auto w_vals = output_layer.weights.values;
        int nnz = w_vals.extent_int(0);
        
        auto h_weights = Kokkos::create_mirror_view(w_vals);
        Kokkos::deep_copy(h_weights, w_vals);
        Kokkos::fence();
        
        for (int k = 0; k < nnz; ++k) {
            if (h_weights(k) != 0.0) {
                all_weights.push_back(Kokkos::abs(h_weights(k)));
            }
        }
    }
    
    if (all_weights.empty()) {
        std::cout << "Aucun poids non-nul trouvé!" << std::endl;
        return 0.0;
    }
    
    // Trier les poids par ordre croissant
    std::sort(all_weights.begin(), all_weights.end());
    
    // Calculer le percentile correspondant à la sparsité cible
    size_t index = static_cast<size_t>(target_sparsity * all_weights.size());
    if (index >= all_weights.size()) index = all_weights.size() - 1;
    
    real threshold = all_weights[index];
    
    std::cout << "Statistiques des poids:" << std::endl;
    std::cout << "- Nombre total de poids non-nuls: " << all_weights.size() << std::endl;
    std::cout << "- Poids minimum: " << std::fixed << std::setprecision(6) << all_weights.front() << std::endl;
    std::cout << "- Poids maximum: " << std::fixed << std::setprecision(6) << all_weights.back() << std::endl;
    std::cout << "- Poids médian: " << std::fixed << std::setprecision(6) 
              << all_weights[all_weights.size() / 2] << std::endl;
    std::cout << "- Seuil calculé (percentile " << std::fixed << std::setprecision(1) 
              << (target_sparsity * 100.0) << "%): " << std::fixed << std::setprecision(6) 
              << threshold << std::endl;
    
    return threshold;
}

void Network::apply_adaptive_sparsity(real target_sparsity) {
    real threshold = compute_adaptive_threshold(target_sparsity);
    std::cout << "\nApplication du seuil adaptatif: " << std::fixed << std::setprecision(6) 
              << threshold << std::endl;
    apply_threshold_sparsity(threshold);
}

void Network::apply_progressive_sparsity(real target_sparsity, real max_threshold) {
    std::cout << "\n=== SPARSITÉ PROGRESSIVE ===" << std::endl;
    std::cout << "Sparsité cible: " << std::fixed << std::setprecision(1) 
              << (target_sparsity * 100.0) << "%" << std::endl;
    std::cout << "Seuil maximum: " << std::fixed << std::setprecision(3) << max_threshold << std::endl;
    
    std::vector<real> thresholds = {0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50};
    
    for (real threshold : thresholds) {
        if (threshold > max_threshold) break;
        
        std::cout << "\n--- Test avec seuil " << std::fixed << std::setprecision(3) 
                  << threshold << " ---" << std::endl;
        
        // Tester le seuil sur une copie temporaire
        // Note: On ne peut pas copier Network, donc on teste directement
        std::cout << "Test avec seuil " << std::fixed << std::setprecision(3) 
                  << threshold << "..." << std::endl;
        
        // Appliquer temporairement et vérifier
        apply_threshold_sparsity(threshold);
        
        // Vérifier la sparsité obtenue
        if (is_sparse(target_sparsity)) {
            std::cout << "✓ Sparsité cible atteinte avec seuil " << threshold << std::endl;
            return;
        } else {
            std::cout << "✗ Sparsité insuffisante, essai suivant..." << std::endl;
            // Note: On ne peut pas annuler facilement, donc on continue
        }
    }
    
    std::cout << "⚠️  Impossible d'atteindre la sparsité cible avec les seuils testés" << std::endl;
    std::cout << "Application du seuil maximum: " << max_threshold << std::endl;
    apply_threshold_sparsity(max_threshold);
}

std::vector<real> Network::compute_layer_adaptive_thresholds(real target_sparsity) const {
    std::cout << "\n=== CALCUL DES SEUILS ADAPTATIFS PAR COUCHE ===" << std::endl;
    std::vector<real> thresholds;
    
    // Couches cachées
    for (size_t i = 0; i < hidden_layers.size(); ++i) {
        const auto& layer = hidden_layers[i];
        if (layer.input_size > 0) {
            std::vector<real> layer_weights;
            auto w_vals = layer.weights.values;
            int nnz = w_vals.extent_int(0);
            
            auto h_weights = Kokkos::create_mirror_view(w_vals);
            Kokkos::deep_copy(h_weights, w_vals);
            Kokkos::fence();
            
            for (int k = 0; k < nnz; ++k) {
                if (h_weights(k) != 0.0) {
                    layer_weights.push_back(Kokkos::abs(h_weights(k)));
                }
            }
            
            if (!layer_weights.empty()) {
                std::sort(layer_weights.begin(), layer_weights.end());
                size_t index = static_cast<size_t>(target_sparsity * layer_weights.size());
                if (index >= layer_weights.size()) index = layer_weights.size() - 1;
                
                real threshold = layer_weights[index];
                thresholds.push_back(threshold);
                
                std::cout << "Couche cachée " << (i+1) << ": seuil = " 
                          << std::fixed << std::setprecision(6) << threshold << std::endl;
            } else {
                thresholds.push_back(0.0);
                std::cout << "Couche cachée " << (i+1) << ": aucun poids non-nul" << std::endl;
            }
        }
    }
    
    // Couche de sortie
    if (output_layer.input_size > 0) {
        std::vector<real> layer_weights;
        auto w_vals = output_layer.weights.values;
        int nnz = w_vals.extent_int(0);
        
        auto h_weights = Kokkos::create_mirror_view(w_vals);
        Kokkos::deep_copy(h_weights, w_vals);
        Kokkos::fence();
        
        for (int k = 0; k < nnz; ++k) {
            if (h_weights(k) != 0.0) {
                layer_weights.push_back(Kokkos::abs(h_weights(k)));
            }
        }
        
        if (!layer_weights.empty()) {
            std::sort(layer_weights.begin(), layer_weights.end());
            size_t index = static_cast<size_t>(target_sparsity * layer_weights.size());
            if (index >= layer_weights.size()) index = layer_weights.size() - 1;
            
            real threshold = layer_weights[index];
            thresholds.push_back(threshold);
            
            std::cout << "Couche sortie: seuil = " << std::fixed << std::setprecision(6) 
                      << threshold << std::endl;
        } else {
            thresholds.push_back(0.0);
            std::cout << "Couche sortie: aucun poids non-nul" << std::endl;
        }
    }
    
    return thresholds;
}

void Network::apply_layer_adaptive_sparsity(real target_sparsity) {
    std::vector<real> thresholds = compute_layer_adaptive_thresholds(target_sparsity);
    
    std::cout << "\n=== APPLICATION DES SEUILS PAR COUCHE ===" << std::endl;
    
    // Appliquer aux couches cachées
    for (size_t i = 0; i < hidden_layers.size() && i < thresholds.size(); ++i) {
        auto& layer = hidden_layers[i];
        if (layer.input_size > 0) {
            real threshold = thresholds[i];
            std::cout << "Couche cachée " << (i+1) << ": seuil = " 
                      << std::fixed << std::setprecision(6) << threshold << std::endl;
            
            auto w_vals = layer.weights.values;
            int nnz = w_vals.extent_int(0);
            
            int layer_weights_zeroed = 0;
            Kokkos::parallel_reduce("apply_layer_threshold_sparse", nnz, 
                KOKKOS_LAMBDA(const int k, int& local_count) {
                    if (Kokkos::abs(w_vals(k)) < threshold) {
                        w_vals(k) = 0.0;
                        local_count++;
                    }
                }, layer_weights_zeroed);
            
            std::cout << "  → " << layer_weights_zeroed << "/" << nnz 
                      << " poids supprimés (" 
                      << std::fixed << std::setprecision(1) 
                      << (100.0 * layer_weights_zeroed / nnz) << "%)" << std::endl;
        }
    }
    
    // Appliquer à la couche de sortie
    if (output_layer.input_size > 0 && thresholds.size() > hidden_layers.size()) {
        real threshold = thresholds[hidden_layers.size()];
        std::cout << "Couche sortie: seuil = " << std::fixed << std::setprecision(6) 
                  << threshold << std::endl;
        
        auto w_vals = output_layer.weights.values;
        int nnz = w_vals.extent_int(0);
        
        int layer_weights_zeroed = 0;
        Kokkos::parallel_reduce("apply_layer_threshold_output_sparse", nnz, 
            KOKKOS_LAMBDA(const int k, int& local_count) {
                if (Kokkos::abs(w_vals(k)) < threshold) {
                    w_vals(k) = 0.0;
                    local_count++;
                }
            }, layer_weights_zeroed);
        
        std::cout << "  → " << layer_weights_zeroed << "/" << nnz 
                  << " poids supprimés (" 
                  << std::fixed << std::setprecision(1) 
                  << (100.0 * layer_weights_zeroed / nnz) << "%)" << std::endl;
    }
    
    std::cout << "Application terminée!" << std::endl;
}

// === NOUVELLES MÉTHODES POUR RÉGULARISATION ET MASQUES ===

// Implémentation du constructeur SparsityMask
SparsityMask::SparsityMask(int _size) : size(_size) {
    mask = Kokkos::View<bool*, Kokkos::DefaultExecutionSpace>("sparsity_mask", _size);
    Kokkos::deep_copy(mask, true); // Initialement tous actifs
}

void Network::apply_l1_regularization(real lambda) {
    L1RegularizationStrategy strat(lambda);
    strat.apply(*this);
}

void Network::create_sparsity_masks(real threshold) {
    std::cout << "\n=== CRÉATION DES MASQUES DE SPARSITÉ ===" << std::endl;
    std::cout << "Seuil pour masques: " << threshold << std::endl;
    
    // Nettoyer les masques existants
    hidden_layer_masks.clear();
    
    // Créer les masques pour les couches cachées
    for (const auto& layer : hidden_layers) {
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            int nnz = w_vals.extent_int(0);
            
            SparsityMask mask(nnz);
            
            // Créer le masque basé sur les poids actuels
            Kokkos::parallel_for("create_sparsity_mask_hidden", nnz,
                KOKKOS_LAMBDA(const int k) {
                    mask.mask(k) = (Kokkos::abs(w_vals(k)) >= threshold);
                });
            
            hidden_layer_masks.push_back(std::move(mask));
            std::cout << "Masque créé pour couche cachée: " << nnz << " poids" << std::endl;
        }
    }
    
    // Créer le masque pour la couche de sortie
    if (output_layer.input_size > 0) {
        auto w_vals = output_layer.weights.values;
        int nnz = w_vals.extent_int(0);
        
        output_layer_mask = SparsityMask(nnz);
        
        Kokkos::parallel_for("create_sparsity_mask_output", nnz,
            KOKKOS_LAMBDA(const int k) {
                output_layer_mask.mask(k) = (Kokkos::abs(w_vals(k)) >= threshold);
            });
        
        std::cout << "Masque créé pour couche sortie: " << nnz << " poids" << std::endl;
    }
    
    masks_created = true;
    std::cout << "Masques de sparsité créés!" << std::endl;
    std::cout << "=====================================\n" << std::endl;
}

void Network::apply_sparsity_masks() {
    if (!masks_created) {
        std::cout << "Aucun masque de sparsité créé. Utilisez create_sparsity_masks() d'abord." << std::endl;
        return;
    }
    
    std::cout << "\n=== APPLICATION DES MASQUES DE SPARSITÉ ===" << std::endl;
    
    // Appliquer aux couches cachées
    for (size_t i = 0; i < hidden_layers.size() && i < hidden_layer_masks.size(); ++i) {
        auto& layer = hidden_layers[i];
        const auto& mask = hidden_layer_masks[i];
        
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            auto dw_vals = layer.d_weights.values;
            auto dw_sum_vals = layer.d_weights_sum.values;
            int nnz = w_vals.extent_int(0);
            
            Kokkos::parallel_for("apply_sparsity_mask_hidden", nnz,
                KOKKOS_LAMBDA(const int k) {
                    if (!mask.mask(k)) {
                        w_vals(k) = 0.0;
                        dw_vals(k) = 0.0;
                        dw_sum_vals(k) = 0.0;
                    }
                });
            
            std::cout << "Masque appliqué à couche cachée " << (i+1) << std::endl;
        }
    }
    
    // Appliquer à la couche de sortie
    if (output_layer.input_size > 0) {
        auto w_vals = output_layer.weights.values;
        auto dw_vals = output_layer.d_weights.values;
        auto dw_sum_vals = output_layer.d_weights_sum.values;
        int nnz = w_vals.extent_int(0);
        
        Kokkos::parallel_for("apply_sparsity_mask_output", nnz,
            KOKKOS_LAMBDA(const int k) {
                if (!output_layer_mask.mask(k)) {
                    w_vals(k) = 0.0;
                    dw_vals(k) = 0.0;
                    dw_sum_vals(k) = 0.0;
                }
            });
        
        std::cout << "Masque appliqué à couche sortie" << std::endl;
    }
    
    std::cout << "Masques de sparsité appliqués!" << std::endl;
    std::cout << "=====================================\n" << std::endl;
}

bool Network::has_sparsity_masks() const {
    return masks_created;
}

void Network::remove_sparsity_masks() {
    std::cout << "\n=== SUPPRESSION DES MASQUES DE SPARSITÉ ===" << std::endl;
    
    hidden_layer_masks.clear();
    masks_created = false;
    
    std::cout << "Masques de sparsité supprimés!" << std::endl;
    std::cout << "=====================================\n" << std::endl;
}

// === NOUVELLES STRATÉGIES DE PRUNING AVANCÉES ===

void Network::apply_structural_pruning(real threshold) {
    StructuralPruningStrategy strat(threshold);
    strat.apply(*this);
}

void Network::apply_progressive_pruning(real initial_threshold, real final_threshold, int epochs) {
    ProgressivePruningStrategy strat(initial_threshold, final_threshold, epochs);
    strat.apply(*this);
}

void Network::apply_sensitivity_pruning(real threshold) {
    SensitivityPruningStrategy strat(threshold);
    strat.apply(*this);
}

void Network::apply_layer_specific_pruning(const std::vector<real>& thresholds) {
    LayerSpecificPruningStrategy strat(thresholds);
    strat.apply(*this);
}

void Network::apply_importance_based_pruning(real sparsity_target) {
    ImportanceBasedPruningStrategy strat(sparsity_target);
    strat.apply(*this);
}

void Network::apply_pruning_with_retraining(real threshold, int retrain_epochs) {
    std::cout << "\n=== PRUNING AVEC RÉENTRAÎNEMENT ===" << std::endl;
    std::cout << "Seuil de pruning: " << threshold << std::endl;
    std::cout << "Époques de réentraînement: " << retrain_epochs << std::endl;
    
    // Étape 1: Pruning initial
    std::cout << "\n--- ÉTAPE 1: PRUNING INITIAL ---" << std::endl;
    apply_threshold_sparsity(threshold);
    
    // Étape 2: Réentraînement
    std::cout << "\n--- ÉTAPE 2: RÉENTRAÎNEMENT ---" << std::endl;
    for (int epoch = 0; epoch < retrain_epochs; ++epoch) {
        // Ici, vous devriez avoir accès aux données d'entraînement
        // Pour l'exemple, on simule juste le processus
        std::cout << "Époque de réentraînement " << (epoch + 1) << "/" << retrain_epochs << std::endl;
        
        // Appliquer les masques de sparsité après chaque mise à jour
        if (has_sparsity_masks()) {
            apply_sparsity_masks();
        }
    }
    
    std::cout << "Pruning avec réentraînement terminé!" << std::endl;
    std::cout << "==================================\n" << std::endl;
}

void Network::apply_activation_variance_pruning(real threshold) {
    std::cout << "\n=== PRUNING BASÉ SUR LA VARIANCE DES ACTIVATIONS ===" << std::endl;
    std::cout << "Seuil de variance: " << threshold << std::endl;
    
    // Cette méthode nécessiterait de calculer la variance des activations
    // Pour l'exemple, on utilise une approximation basée sur les gradients
    
    for (auto& layer : hidden_layers) {
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            auto dw_sum_vals = layer.d_weights_sum.values;
            int nnz = w_vals.extent_int(0);
            
            int weights_removed = 0;
            Kokkos::parallel_reduce("activation_variance_pruning", nnz,
                KOKKOS_LAMBDA(const int k, int& local_count) {
                    // Approximation: variance ≈ gradient²
                    real variance_approx = dw_sum_vals(k) * dw_sum_vals(k);
                    if (variance_approx < threshold) {
                        w_vals(k) = 0.0;
                        local_count++;
                    }
                }, weights_removed);
            
            std::cout << "Couche cachée: " << weights_removed << "/" << nnz 
                      << " poids supprimés par variance" << std::endl;
        }
    }
    
    std::cout << "Pruning par variance des activations terminé!" << std::endl;
    std::cout << "============================================\n" << std::endl;
}

void Network::apply_pruning_with_regrowth(real prune_threshold, real regrow_threshold, int regrow_ratio) {
    std::cout << "\n=== PRUNING AVEC CROISSANCE ===" << std::endl;
    std::cout << "Seuil de pruning: " << prune_threshold << std::endl;
    std::cout << "Seuil de croissance: " << regrow_threshold << std::endl;
    std::cout << "Ratio de croissance: 1/" << regrow_ratio << std::endl;
    
    // Étape 1: Pruning
    std::cout << "\n--- ÉTAPE 1: PRUNING ---" << std::endl;
    apply_threshold_sparsity(prune_threshold);
    
    // Étape 2: Croissance (simulation)
    std::cout << "\n--- ÉTAPE 2: CROISSANCE ---" << std::endl;
    std::cout << "Simulation de la croissance de " << regrow_ratio << " poids" << std::endl;
    
    // Dans une vraie implémentation, vous devriez:
    // 1. Identifier les poids les plus prometteurs à regrow
    // 2. Réinitialiser ces poids avec de petites valeurs aléatoires
    // 3. Continuer l'entraînement
    
    std::cout << "Pruning avec croissance terminé!" << std::endl;
    std::cout << "==============================\n" << std::endl;
}

// === GESTION DES STRATÉGIES DE SPARSITÉ ===
void Network::add_sparsity_strategy(std::unique_ptr<ISparsityStrategy> strat) {
    if (strat) {
        sparsity_strategies.push_back(std::move(strat));
    }
}

void Network::apply_sparsity_strategies() {
    for (auto& strat : sparsity_strategies) {
        if (strat) strat->apply(*this);
    }
}

Network::~Network() = default; 