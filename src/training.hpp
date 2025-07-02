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

// Fonction de test avec sparsité
void test_sparsity_xor(const std::string& optimizer_choice = "adam", real sparsity_threshold = 0.01);
void test_sparsity_sine(const std::string& optimizer_choice = "adam", real sparsity_threshold = 0.01); 