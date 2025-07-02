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
    
    // === NOUVELLES MÉTHODES POUR DÉTECTION ET CONVERSION AUTOMATIQUE ===
    
    // Détecte si une matrice est déjà sparse (taux de zéros > threshold)
    bool is_sparse(real sparsity_threshold = 0.5) const;
    
    // Convertit automatiquement en sparse si nécessaire
    void auto_convert_to_sparse(real sparsity_threshold = 0.5, real weight_threshold = 0.01);
    
    // Optimise la structure CSR en supprimant les zéros exacts
    void optimize_csr_structure();
    
    // Affiche un rapport complet de sparsité
    void sparsity_report() const;
    
    // === NOUVELLES MÉTHODES POUR SEUILS ADAPTATIFS ===
    
    // Calcule un seuil adaptatif basé sur la distribution des poids
    real compute_adaptive_threshold(real target_sparsity = 0.3) const;
    
    // Applique la sparsité avec seuil adaptatif
    void apply_adaptive_sparsity(real target_sparsity = 0.3);
    
    // Applique la sparsité progressive jusqu'à atteindre la cible
    void apply_progressive_sparsity(real target_sparsity = 0.3, real max_threshold = 0.5);
    
    // Calcule des seuils adaptatifs par couche
    std::vector<real> compute_layer_adaptive_thresholds(real target_sparsity = 0.3) const;
    
    // Applique la sparsité avec seuils différents par couche
    void apply_layer_adaptive_sparsity(real target_sparsity = 0.3);
}; 