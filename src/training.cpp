#include "training.hpp"
#include "optimizers.hpp"
#include <utility>   // std::pair

// === FONCTIONS UTILITAIRES MINIMALISTES ===

void print_header(const std::string& title) {
    std::cout << "\n" << title << std::endl;
    std::cout << std::string(title.length(), '-') << std::endl;
}

void print_progress(int epoch, int total_epochs, real cost) {
    int progress = (epoch * 20) / total_epochs;
    std::cout << "\r[" << std::string(progress, '#') << std::string(20-progress, '.') << "] ";
    std::cout << epoch << "/" << total_epochs << " | Cost: " << std::scientific << std::setprecision(3) << cost;
    std::cout.flush();
}

// === UTILITAIRE GENERIQUE : PARTITION TRAIN / TEST ===
std::pair<std::vector<int>, std::vector<int>> train_test_split(int num_samples,
                                                              double train_ratio,
                                                              std::mt19937& gen) {
    std::vector<int> all_indices(num_samples);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::shuffle(all_indices.begin(), all_indices.end(), gen);
    int split = static_cast<int>(train_ratio * num_samples);
    std::vector<int> train_indices(all_indices.begin(), all_indices.begin() + split);
    std::vector<int> test_indices (all_indices.begin() + split, all_indices.end());
    return {std::move(train_indices), std::move(test_indices)};
}

// === STRUCTURES ET FONCTIONS GENERIQUES ===

struct TrainingConfig {
    std::string problem_name;
    std::map<int, int> network_sizes;
    std::vector<std::string> activations;
    real sgd_lr, sgd_epochs, adam_lr, adam_epochs;
    int batch_size;
    int num_samples;
    real convergence_threshold;
    bool enable_dynamic_sparsity = false;
    real sparsity_threshold = 0.01;
    int sparsity_frequency = 50;
    int sparsity_start_epoch = 100;
};

struct TrainingData {
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    HostView2D h_inputs, h_outputs;
    Kokkos::View<real**, Kokkos::DefaultExecutionSpace> inputs, outputs;
    std::vector<int> train_indices, test_indices;
    int input_dim, output_dim;
};

std::unique_ptr<Optimizer> create_optimizer(const std::string& choice, const TrainingConfig& config) {
    if (choice == "sgd") {
        return std::make_unique<SGD>(config.sgd_lr);
    } else {
        return std::make_unique<Adam>(config.adam_lr);
    }
}

int get_epochs(const std::string& choice, const TrainingConfig& config) {
    return (choice == "sgd") ? config.sgd_epochs : config.adam_epochs;
}

real get_learning_rate(const std::string& choice, const TrainingConfig& config) {
    return (choice == "sgd") ? config.sgd_lr : config.adam_lr;
}

// Fonction générique d'entraînement minimaliste
void generic_train_network(Network& network, TrainingData& data, const TrainingConfig& config, 
                          const std::string& optimizer_choice, std::mt19937& gen) {
    int epochs = get_epochs(optimizer_choice, config);
    
    std::cout << "Training... ";
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        std::shuffle(data.train_indices.begin(), data.train_indices.end(), gen);
        
        for (int batch_start = 0; batch_start < static_cast<int>(data.train_indices.size()); batch_start += config.batch_size) {
            int current_batch_size = std::min(config.batch_size, static_cast<int>(data.train_indices.size()) - batch_start);
            if (current_batch_size <= 0) continue;
            
            network.zero_accumulated_gradients();
            for (int j = 0; j < current_batch_size; ++j) {
                int sample_index = data.train_indices[batch_start + j];
                auto input_subview = Kokkos::subview(data.inputs, sample_index, Kokkos::ALL());
                auto target_subview = Kokkos::subview(data.outputs, sample_index, Kokkos::ALL());
                View1D prediction = network.forward(input_subview);
                total_epoch_cost += network.calculate_cost(prediction, target_subview);
                network.backward(target_subview);
            }
            network.update(current_batch_size);
        }
        
        real avg_cost = total_epoch_cost / data.train_indices.size();
        
        // Sparsité dynamique
        if (config.enable_dynamic_sparsity && 
            epoch >= config.sparsity_start_epoch && 
            (epoch - config.sparsity_start_epoch) % config.sparsity_frequency == 0) {
            network.apply_threshold_sparsity_silent(config.sparsity_threshold);
        }
        
        // Affichage de progression simplifié
        if (epoch % 50 == 0 || epoch == epochs - 1) {
            print_progress(epoch + 1, epochs, avg_cost);
        }
        
        if (avg_cost < config.convergence_threshold) {
            std::cout << " | Convergence atteinte!" << std::endl;
            break;
        }
    }
    std::cout << std::endl;
}

// Fonction générique d'évaluation avec précision
real generic_evaluate_network(Network& network, TrainingData& data) {
    View1D prediction_result("prediction_result", data.output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0;
    int correct_predictions = 0;
    
    for (int idx : data.test_indices) {
        auto input_subview = Kokkos::subview(data.inputs, idx, Kokkos::ALL());
        auto target_subview = Kokkos::subview(data.outputs, idx, Kokkos::ALL());
        View1D prediction = network.forward(input_subview);
        final_total_cost += network.calculate_cost(prediction, target_subview);
        
        // Calcul de la précision pour classification
        if (data.output_dim == 1) {
            Kokkos::deep_copy(prediction_result, prediction);
            Kokkos::deep_copy(h_prediction_result, prediction_result); 
            Kokkos::fence();
            int predicted_class = std::round(h_prediction_result(0));
            int target_class = static_cast<int>(data.h_outputs(idx, 0));
            if (predicted_class == target_class) correct_predictions++;
        } else if (data.output_dim > 1) {
            // Classification multi-classes
            Kokkos::deep_copy(prediction_result, prediction);
            Kokkos::deep_copy(h_prediction_result, prediction_result);
            Kokkos::fence();
            
            int predicted_class = 0;
            real max_val = h_prediction_result(0);
            for (int j = 1; j < data.output_dim; ++j) {
                if (h_prediction_result(j) > max_val) {
                    max_val = h_prediction_result(j);
                    predicted_class = j;
                }
            }
            
            int target_class = 0;
            real max_target = data.h_outputs(idx, 0);
            for (int j = 1; j < data.output_dim; ++j) {
                if (data.h_outputs(idx, j) > max_target) {
                    max_target = data.h_outputs(idx, j);
                    target_class = j;
                }
            }
            
            if (predicted_class == target_class) correct_predictions++;
        }
    }
    
    real accuracy = static_cast<real>(correct_predictions) / data.test_indices.size();
    real avg_cost = final_total_cost / data.test_indices.size();
    
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy * 100.0 << "%";
    std::cout << " | Cost: " << std::scientific << std::setprecision(3) << avg_cost << std::endl;
    
    return accuracy;
}

// === CONFIGURATIONS SIMPLIFIÉES ===

TrainingConfig get_xor_config() {
    return {
        "XOR",
        {{0, 2}, {1, 20}, {2, 10}, {3, 1}}, {"relu", "relu", "sigmoid"},
        0.3, 1200, 0.01, 800, 4, 4, 1e-8
    };
}

TrainingConfig get_sine_config() {
    return {
        "SINUS",
        {{0, 1}, {1, 16}, {2, 16}, {3, 1}}, {"relu", "relu", "linear"},
        0.02, 600, 0.001, 800, 16, 1024, 1e-9
    };
}

TrainingConfig get_linear_config() {
    return {
        "SEPARATION LINEAIRE",
        {{0, 2}, {1, 1}}, {"sigmoid"},
        0.1, 120, 0.01, 160, 16, 800, 1e-7
    };
}

TrainingConfig get_spiral_config() {
    return {
        "SPIRALES",
        {{0, 2}, {1, 50}, {2, 25}, {3, 3}}, {"relu", "relu", "sigmoid"},
        0.3, 800, 0.005, 600, 32, 900, 1e-6
    };
}

TrainingConfig get_gaussian_config() {
    return {
        "CLUSTERS GAUSSIENS",
        {{0, 2}, {1, 40}, {2, 30}, {3, 4}}, {"relu", "relu", "sigmoid"},
        0.4, 600, 0.008, 400, 40, 800, 1e-6
    };
}

TrainingConfig get_xor_dynamic_sparsity_config() {
    return {
        "XOR DYNAMIQUE",
        {{0, 2}, {1, 20}, {2, 10}, {3, 1}}, {"relu", "relu", "sigmoid"},
        0.3, 1500, 0.01, 1200, 4, 4, 1e-8,
        true, 0.025, 40, 80
    };
}

TrainingConfig get_sine_dynamic_sparsity_config() {
    return {
        "SINUS DYNAMIQUE",
        {{0, 1}, {1, 24}, {2, 24}, {3, 1}}, {"relu", "relu", "linear"},
        0.02, 900, 0.001, 1200, 16, 1024, 1e-9,
        true, 0.015, 30, 60
    };
}

// === GÉNÉRATEURS DE DONNÉES ===

TrainingData generate_xor_data(std::mt19937& gen) {
    TrainingData data;
    data.input_dim = 2; data.output_dim = 1;
    data.h_inputs = TrainingData::HostView2D("xor_inputs", 4, 2);
    data.h_outputs = TrainingData::HostView2D("xor_outputs", 4, 1);
    
    data.h_inputs(0,0)=0; data.h_inputs(0,1)=0; data.h_outputs(0,0)=0;
    data.h_inputs(1,0)=1; data.h_inputs(1,1)=0; data.h_outputs(1,0)=1;
    data.h_inputs(2,0)=0; data.h_inputs(2,1)=1; data.h_outputs(2,0)=1;
    data.h_inputs(3,0)=1; data.h_inputs(3,1)=1; data.h_outputs(3,0)=0;
    
    data.inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_inputs);
    data.outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_outputs);
    
    data.train_indices = {0, 1, 2, 3};
    data.test_indices = {0, 1, 2, 3};
    return data;
}

TrainingData generate_sine_data(int num_samples, std::mt19937& gen) {
    TrainingData data;
    data.input_dim = 1; data.output_dim = 1;
    data.h_inputs = TrainingData::HostView2D("sine_inputs", num_samples, 1);
    data.h_outputs = TrainingData::HostView2D("sine_outputs", num_samples, 1);
    
    std::uniform_real_distribution<real> distrib(-M_PI, M_PI);
    for(int i = 0; i < num_samples; ++i) {
        real x = distrib(gen);
        data.h_inputs(i, 0) = x;
        data.h_outputs(i, 0) = std::sin(x);
    }
    
    data.inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_inputs);
    data.outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_outputs);
    
    auto [train_idx, test_idx] = train_test_split(num_samples, 0.8, gen);
    data.train_indices = std::move(train_idx);
    data.test_indices = std::move(test_idx);
    return data;
}

TrainingData generate_linear_data(int num_samples, std::mt19937& gen) {
    TrainingData data;
    data.input_dim = 2; data.output_dim = 1;
    data.h_inputs = TrainingData::HostView2D("linear_inputs", num_samples, 2);
    data.h_outputs = TrainingData::HostView2D("linear_outputs", num_samples, 1);
    
    std::uniform_real_distribution<real> distrib(-1.0, 1.0);
    real margin = 0.1;
    for(int i = 0; i < num_samples; ++i) {
        real x, y;
        do {
            x = distrib(gen); y = distrib(gen);
        } while (std::abs(y - x) < margin);
        
        data.h_inputs(i, 0) = x; data.h_inputs(i, 1) = y;
        data.h_outputs(i, 0) = (y > x) ? 1.0 : 0.0;
    }
    
    data.inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_inputs);
    data.outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_outputs);
    
    auto [train_idx, test_idx] = train_test_split(num_samples, 0.8, gen);
    data.train_indices = std::move(train_idx);
    data.test_indices = std::move(test_idx);
    return data;
}

TrainingData generate_spiral_data(int num_samples, std::mt19937& gen) {
    TrainingData data;
    data.input_dim = 2; data.output_dim = 3;
    data.h_inputs = TrainingData::HostView2D("spiral_inputs", num_samples, 2);
    data.h_outputs = TrainingData::HostView2D("spiral_outputs", num_samples, 3);
    
    std::uniform_real_distribution<real> noise_distrib(-0.1, 0.1);
    int samples_per_class = num_samples / 3;
    for (int class_id = 0; class_id < 3; ++class_id) {
        for (int i = 0; i < samples_per_class; ++i) {
            int sample_idx = class_id * samples_per_class + i;
            real t = static_cast<real>(i) / samples_per_class * 2.0 * M_PI;
            real radius = 0.3 + t * 0.15;
            real angle_offset = class_id * 2.0 * M_PI / 3.0;
            real x = radius * std::cos(t + angle_offset) + noise_distrib(gen);
            real y = radius * std::sin(t + angle_offset) + noise_distrib(gen);
            
            data.h_inputs(sample_idx, 0) = x;
            data.h_inputs(sample_idx, 1) = y;
            for (int j = 0; j < 3; ++j) {
                data.h_outputs(sample_idx, j) = (j == class_id) ? 1.0 : 0.0;
            }
        }
    }
    
    data.inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_inputs);
    data.outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_outputs);
    
    auto [train_idx, test_idx] = train_test_split(num_samples, 0.8, gen);
    data.train_indices = std::move(train_idx);
    data.test_indices = std::move(test_idx);
    return data;
}

TrainingData generate_gaussian_data(int num_samples, std::mt19937& gen) {
    TrainingData data;
    data.input_dim = 2; data.output_dim = 4;
    data.h_inputs = TrainingData::HostView2D("gaussian_inputs", num_samples, 2);
    data.h_outputs = TrainingData::HostView2D("gaussian_outputs", num_samples, 4);
    
    std::vector<std::array<real, 2>> centers = {{-1.5, -1.5}, {1.5, -1.5}, {-1.5, 1.5}, {1.5, 1.5}};
    std::vector<real> stds = {0.4, 0.35, 0.45, 0.38};
    int samples_per_class = num_samples / 4;
    
    for (int class_id = 0; class_id < 4; ++class_id) {
        std::normal_distribution<real> normal_x(centers[class_id][0], stds[class_id]);
        std::normal_distribution<real> normal_y(centers[class_id][1], stds[class_id]);
        
        for (int i = 0; i < samples_per_class; ++i) {
            int sample_idx = class_id * samples_per_class + i;
            data.h_inputs(sample_idx, 0) = normal_x(gen);
            data.h_inputs(sample_idx, 1) = normal_y(gen);
            for (int j = 0; j < 4; ++j) {
                data.h_outputs(sample_idx, j) = (j == class_id) ? 1.0 : 0.0;
            }
        }
    }
    
    data.inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_inputs);
    data.outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_outputs);
    
    auto [train_idx, test_idx] = train_test_split(num_samples, 0.8, gen);
    data.train_indices = std::move(train_idx);
    data.test_indices = std::move(test_idx);
    return data;
}

// === FONCTIONS DE TEST SIMPLIFIÉES ===

void run_simple_test(const TrainingConfig& config, 
                     std::function<TrainingData(std::mt19937&)> data_generator,
                     const std::string& optimizer_choice) {
    print_header(config.problem_name);
    
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = data_generator(gen);
    
    generic_train_network(network, data, config, optimizer_choice, gen);
    generic_evaluate_network(network, data);
}

// === FONCTIONS D'ENTRAÎNEMENT PRINCIPALES ===

void xor_train(const std::string& optimizer_choice) {
    run_simple_test(get_xor_config(), generate_xor_data, optimizer_choice);
}

void sine_train(const std::string& optimizer_choice) {
    auto config = get_sine_config();
    run_simple_test(config, [&config](std::mt19937& gen) { return generate_sine_data(config.num_samples, gen); }, optimizer_choice);
}

void linear_sep_train(const std::string& optimizer_choice) {
    auto config = get_linear_config();
    run_simple_test(config, [&config](std::mt19937& gen) { return generate_linear_data(config.num_samples, gen); }, optimizer_choice);
}

void spiral_train(const std::string& optimizer_choice) {
    auto config = get_spiral_config();
    run_simple_test(config, [&config](std::mt19937& gen) { return generate_spiral_data(config.num_samples, gen); }, optimizer_choice);
}

void gaussian_clusters_train(const std::string& optimizer_choice) {
    auto config = get_gaussian_config();
    run_simple_test(config, [&config](std::mt19937& gen) { return generate_gaussian_data(config.num_samples, gen); }, optimizer_choice);
}

// === FONCTIONS AVEC SPARSITÉ DYNAMIQUE ===

void test_dynamic_sparsity_xor(const std::string& optimizer_choice) {
    print_header("XOR - SPARSITÉ DYNAMIQUE");
    
    auto config = get_xor_dynamic_sparsity_config();
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_xor_data(gen);
    
    generic_train_network(network, data, config, optimizer_choice, gen);
    generic_evaluate_network(network, data);
}

void test_dynamic_sparsity_sine(const std::string& optimizer_choice) {
    print_header("SINUS - SPARSITÉ DYNAMIQUE");
    
    auto config = get_sine_dynamic_sparsity_config();
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_sine_data(config.num_samples, gen);
    
    generic_train_network(network, data, config, optimizer_choice, gen);
    generic_evaluate_network(network, data);
}

// === FONCTIONS DE DÉMONSTRATION SIMPLIFIÉES ===

void demo_automatic_sparsity_conversion(const std::string& optimizer_choice) {
    print_header("CONVERSION AUTOMATIQUE VERS SPARSE");
    
    auto config = get_xor_config();
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_xor_data(gen);
    
    // Entraînement normal
    generic_train_network(network, data, config, optimizer_choice, gen);
    std::cout << "Avant sparsité: ";
    generic_evaluate_network(network, data);
    
    // Application de la sparsité
    network.apply_threshold_sparsity(0.01);
    std::cout << "Après sparsité: ";
    generic_evaluate_network(network, data);
}

void demo_adaptive_sparsity_thresholds(const std::string& optimizer_choice) {
    print_header("SEUILS ADAPTATIFS POUR SPARSITÉ");
    
    auto config = get_sine_config();
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_sine_data(config.num_samples, gen);
    
    // Entraînement normal
    generic_train_network(network, data, config, optimizer_choice, gen);
    std::cout << "Avant sparsité: ";
    generic_evaluate_network(network, data);
    
    // Test avec différents seuils sur le même réseau
    std::vector<real> thresholds = {0.005, 0.01, 0.02, 0.05};
    for (real threshold : thresholds) {
        network.apply_threshold_sparsity_silent(threshold);
        std::cout << "Seuil " << threshold << ": ";
        generic_evaluate_network(network, data);
    }
}

void demo_l1_regularization_and_masks(const std::string& optimizer_choice) {
    print_header("RÉGULARISATION L1 ET MASQUES DE SPARSITÉ");
    
    auto config = get_sine_config();
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_sine_data(config.num_samples, gen);
    
    // Entraînement normal
    generic_train_network(network, data, config, optimizer_choice, gen);
    std::cout << "Sans régularisation: ";
    generic_evaluate_network(network, data);
    
    // Application de masques de sparsité
    network.apply_threshold_sparsity_silent(0.01);
    std::cout << "Avec masques: ";
    generic_evaluate_network(network, data);
}

void demo_advanced_pruning_strategies(const std::string& optimizer_choice) {
    print_header("STRATÉGIES DE PRUNING AVANCÉES");
    
    auto config = get_sine_config();
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_sine_data(config.num_samples, gen);
    
    // Entraînement normal
    generic_train_network(network, data, config, optimizer_choice, gen);
    std::cout << "Réseau dense: ";
    generic_evaluate_network(network, data);
    
    // Pruning progressif
    for (real threshold : {0.005, 0.01, 0.02}) {
        network.apply_threshold_sparsity_silent(threshold);
        std::cout << "Pruning " << threshold << ": ";
        generic_evaluate_network(network, data);
    }
}

void demo_balanced_pruning_strategies(const std::string& optimizer_choice) {
    print_header("PRUNING ÉQUILIBRÉ");
    
    auto config = get_sine_config();
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_sine_data(config.num_samples, gen);
    
    // Entraînement normal
    generic_train_network(network, data, config, optimizer_choice, gen);
    std::cout << "Réseau original: ";
    generic_evaluate_network(network, data);
    
    // Pruning équilibré
    network.apply_threshold_sparsity_silent(0.01);
    std::cout << "Pruning équilibré: ";
    generic_evaluate_network(network, data);
} 