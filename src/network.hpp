#pragma once

#include "types.hpp"
#include "layers.hpp"
#include "optimizers.hpp"

// --- Classes Dataset / BatchHandler (Placeholders) ---
class Dataset {};
class BatchHandler {};

// --- Classe Network ---
class Network {
public:
    std::map<int, int> layer_sizes_map; // Store original sizes map if needed
    InputLayer input_layer;
    std::vector<Layer> hidden_layers; // Uses base Layer class (now with sparse weights)
    OutputLayer output_layer;         // Uses Layer class (now with sparse weights)
    std::unique_ptr<Optimizer> optimizer;
    Dataset dataset;
    BatchHandler batch_handler;

    // Constructor
    Network(const std::map<int, int>& _layer_sizes_map,
            const std::vector<std::string>& activation_types,
            std::unique_ptr<Optimizer> opt);

    int get_size(int layer_index) const;
    int num_layers() const;

    // Forward pass
    View1D forward(const View1D& input_data);

    // Backward pass
    void backward(const View1D& target);

    // Update weights and biases
    void update(int batch_size);

    // Zero accumulated gradients
    void zero_accumulated_gradients();

    // Calculate Cost
    real calculate_cost(const View1D& prediction, const View1D& target);

    // Optimizer Management
    void set_optimizer(std::unique_ptr<Optimizer> opt);
    Optimizer* get_optimizer() const;
    void set_learning_rate(real lr);
    real get_learning_rate() const;

    // Display Network Info
    void show() const;

    // Move/Copy Semantics
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    virtual ~Network() = default;
    
    // === NOUVELLES METHODES POUR LA SPARSITÉ ===
    
    // Applique la sparsité par seuil : met à zéro les poids |w| < threshold
    void apply_threshold_sparsity(real threshold);
    
    // Calcule et affiche les statistiques de sparsité réelle
    void compute_sparsity_stats() const;
    
    // === NOUVELLE MÉTHODE : SPARSITÉ AVEC MASQUE PERMANENT ===
    
    // Applique la sparsité avec masque permanent (empêche la récupération)
    void apply_permanent_sparsity(real threshold);
}; 