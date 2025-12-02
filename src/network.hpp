#pragma once

#include "types.hpp"
#include "layers.hpp"
#include "optimizers.hpp"
#include <vector>
#include <memory>

// Forward declaration pour éviter inclusion circulaire
class ISparsityStrategy;

// --- Classes Dataset / BatchHandler (Placeholders) ---
class Dataset {};
class BatchHandler {};

// Structure pour stocker les masques de sparsité
struct SparsityMask {
    Kokkos::View<bool*, Kokkos::DefaultExecutionSpace> mask;
    int size;
    
    SparsityMask(int _size);
};

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

    // === GESTION DES STRATÉGIES DE SPARSITÉ ===
    void add_sparsity_strategy(std::unique_ptr<ISparsityStrategy> strat);
    void apply_sparsity_strategies();

    template <typename Func>
    void iterate_trainable_layers(Func&& func) {
        for_each_trainable_layer(std::forward<Func>(func));
    }

    // Move/Copy Semantics
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    virtual ~Network();
    
    // === NOUVELLES METHODES POUR LA SPARSITÉ ===
    
    // Applique la sparsité par seuil : met à zéro les poids |w| < threshold
    void apply_threshold_sparsity(real threshold);
    
    // Version silencieuse pour l'entraînement (sans affichage)
    void apply_threshold_sparsity_silent(real threshold);
    
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
    
    // === NOUVELLES MÉTHODES POUR RÉGULARISATION ET MASQUES ===
    
    // Applique la régularisation L1 pendant l'entraînement
    void apply_l1_regularization(real lambda);
    
    // Crée et applique des masques de sparsité permanents
    void create_sparsity_masks(real threshold);
    
    // Applique les masques de sparsité (empêche la récupération)
    void apply_sparsity_masks();
    
    // Vérifie si les masques sont actifs
    bool has_sparsity_masks() const;
    
    // Supprime les masques de sparsité
    void remove_sparsity_masks();
    
    // === NOUVELLES MÉTHODES POUR STRATÉGIES DE PRUNING AVANCÉES ===
    
    // Pruning structurel : supprime des neurones entiers
    void apply_structural_pruning(real threshold);
    
    // Pruning progressif : augmente progressivement le seuil
    void apply_progressive_pruning(real initial_threshold, real final_threshold, int epochs);
    
    // Pruning basé sur la sensibilité (gradient * poids)
    void apply_sensitivity_pruning(real threshold);
    
    // Pruning par couche avec seuils différents
    void apply_layer_specific_pruning(const std::vector<real>& thresholds);
    
    // Pruning adaptatif basé sur l'importance des connexions
    void apply_importance_based_pruning(real sparsity_target);
    
    // Pruning avec réentraînement (fine-tuning)
    void apply_pruning_with_retraining(real threshold, int retrain_epochs);
    
    // Pruning basé sur la variance des activations
    void apply_activation_variance_pruning(real threshold);
    
    // Pruning avec masque de croissance (regrow)
    void apply_pruning_with_regrowth(real prune_threshold, real regrow_threshold, int regrow_ratio);

private:
    // Membres privés pour les masques de sparsité
    std::vector<SparsityMask> hidden_layer_masks;
    SparsityMask output_layer_mask{0}; // Initialisation par défaut
    bool masks_created = false;

    // Nouveau : stockage des stratégies
    std::vector<std::unique_ptr<ISparsityStrategy>> sparsity_strategies;

    // === Helper interne : itérer sur toutes les couches entraînables ===
    template <typename Func>
    void for_each_trainable_layer(Func&& func) {
        size_t idx = 0;
        for (Layer& layer : hidden_layers) {
            if (layer.input_size > 0) {
                func(layer, /*is_output=*/false, idx);
                ++idx;
            }
        }
        if (output_layer.input_size > 0) {
            func(static_cast<Layer&>(output_layer), /*is_output=*/true, idx);
        }
    }

    // Version const pour les méthodes const
    template <typename Func>
    void for_each_trainable_layer(Func&& func) const {
        size_t idx = 0;
        for (const Layer& layer : hidden_layers) {
            if (layer.input_size > 0) {
                func(layer, /*is_output=*/false, idx);
                ++idx;
            }
        }
        if (output_layer.input_size > 0) {
            func(output_layer, /*is_output=*/true, idx);
        }
    }
}; 