#pragma once

#include "types.hpp"
#include "network.hpp"

// Training functions
void xor_train(const std::string& optimizer_choice = "adam");
void sine_train(const std::string& optimizer_choice = "adam");
void linear_sep_train(const std::string& optimizer_choice = "adam");
void spiral_train(const std::string& optimizer_choice = "adam");
void gaussian_clusters_train(const std::string& optimizer_choice = "adam");
void time_series_train(const std::string& optimizer_choice = "adam");

// Fonctions de test avec sparsité dynamique
void test_dynamic_sparsity_xor(const std::string& optimizer_choice = "adam");
void test_dynamic_sparsity_sine(const std::string& optimizer_choice = "adam");

// Fonction de test rapide pour validation
void test_quick_dynamic_sparsity_xor(const std::string& optimizer_choice = "adam");

// === NOUVELLE FONCTION DE DÉMONSTRATION SPARSITÉ AUTOMATIQUE ===
void demo_automatic_sparsity_conversion(const std::string& optimizer_choice = "adam");

// === NOUVELLE FONCTION DE DÉMONSTRATION SEUILS ADAPTATIFS ===
void demo_adaptive_sparsity_thresholds(const std::string& optimizer_choice = "adam");

// === NOUVELLE FONCTION DE DÉMONSTRATION RÉGULARISATION L1 ET MASQUES ===
void demo_l1_regularization_and_masks(const std::string& optimizer_choice = "adam");

// === NOUVELLE FONCTION DE DÉMONSTRATION STRATÉGIES DE PRUNING AVANCÉES ===
void demo_advanced_pruning_strategies(const std::string& optimizer_choice = "adam");

// === NOUVELLE FONCTION DE DÉMONSTRATION PRUNING ÉQUILIBRÉ ===
void demo_balanced_pruning_strategies(const std::string& optimizer_choice = "adam"); 