#include "network.hpp"
#include "activations.hpp"

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
    std::cout << "\n=== APPLICATION DE LA SPARSITÉ PAR SEUIL ===" << std::endl;
    std::cout << "Seuil appliqué: " << threshold << std::endl;
    
    int total_weights_zeroed = 0;
    int total_weights = 0;
    
    // Appliquer aux couches cachées
    for (auto& layer : hidden_layers) {
        if (layer.input_size > 0) {
            auto w_vals = layer.weights.values;
            int nnz = w_vals.extent_int(0);
            total_weights += nnz;
            
            // Compter et mettre à zéro les poids sous le seuil
            int layer_weights_zeroed = 0;
            Kokkos::parallel_reduce("apply_threshold_sparse", nnz, 
                KOKKOS_LAMBDA(const int k, int& local_count) {
                    if (Kokkos::abs(w_vals(k)) < threshold) {
                        w_vals(k) = 0.0;
                        local_count++;
                    }
                }, layer_weights_zeroed);
            
            total_weights_zeroed += layer_weights_zeroed;
            std::cout << "Couche cachée: " << layer_weights_zeroed << "/" << nnz 
                      << " poids supprimés (" 
                      << std::fixed << std::setprecision(1) 
                      << (100.0 * layer_weights_zeroed / nnz) << "%)" << std::endl;
        }
    }
    
    // Appliquer à la couche de sortie
    if (output_layer.input_size > 0) {
        auto w_vals = output_layer.weights.values;
        int nnz = w_vals.extent_int(0);
        total_weights += nnz;
        
        int layer_weights_zeroed = 0;
        Kokkos::parallel_reduce("apply_threshold_output_sparse", nnz, 
            KOKKOS_LAMBDA(const int k, int& local_count) {
                if (Kokkos::abs(w_vals(k)) < threshold) {
                    w_vals(k) = 0.0;
                    local_count++;
                }
            }, layer_weights_zeroed);
        
        total_weights_zeroed += layer_weights_zeroed;
        std::cout << "Couche sortie: " << layer_weights_zeroed << "/" << nnz 
                  << " poids supprimés (" 
                  << std::fixed << std::setprecision(1) 
                  << (100.0 * layer_weights_zeroed / nnz) << "%)" << std::endl;
    }
    
    real sparsity_percentage = (100.0 * total_weights_zeroed) / total_weights;
    std::cout << "\nRÉSULTAT GLOBAL:" << std::endl;
    std::cout << "- Total poids supprimés: " << total_weights_zeroed << "/" << total_weights << std::endl;
    std::cout << "- Taux de sparsité: " << std::fixed << std::setprecision(2) 
              << sparsity_percentage << "%" << std::endl;
    std::cout << "===============================================\n" << std::endl;
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