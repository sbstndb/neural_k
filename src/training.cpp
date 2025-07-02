#include "training.hpp"
#include "optimizers.hpp"
#include <utility>   // std::pair

// === FONCTIONS UTILITAIRES POUR L'AFFICHAGE ===

void print_separator(const std::string& title = "", char border_char = '=', int width = 80) {
    std::string line(width, border_char);
    std::cout << line << std::endl;
    if (!title.empty()) {
        int padding = (width - title.length() - 2) / 2;
        int remaining = width - title.length() - 2 - padding;
        std::string padded_title = std::string(padding, ' ') + title + std::string(remaining, ' ');
        std::cout << "|" << padded_title << "|" << std::endl;
        std::cout << line << std::endl;
    }
}

void print_section_header(const std::string& title, char border_char = '-', int width = 70) {
    std::string line(width, border_char);
    std::cout << "\n" << line << std::endl;
    int padding = (width - title.length() - 2) / 2;
    int remaining = width - title.length() - 2 - padding;
    std::cout << " " << std::string(padding, ' ') << title << std::string(remaining, ' ') << std::endl;
    std::cout << line << std::endl;
}

void print_network_architecture(const Network& network) {
    print_section_header("ARCHITECTURE DU RESEAU", '-', 70);
    
    // Affichage de la structure
    std::cout << "+- Structure des couches:" << std::endl;
    std::cout << "|" << std::endl;
    
    // Input layer
    std::cout << "|  ENTREE      : " << std::setw(3) << network.get_size(0) << " neurones" << std::endl;
    std::cout << "|       |" << std::endl;
    
    // Hidden layers
    int num_hidden = network.num_layers() - 2;
    for (int i = 0; i < num_hidden; ++i) {
        int layer_size = network.get_size(i + 1);
        std::cout << "|  CACHEE " << std::setw(2) << (i+1) << "  : " << std::setw(3) << layer_size << " neurones (activation: RELU)" << std::endl;
        std::cout << "|       |" << std::endl;
    }
    
    // Output layer
    std::cout << "|  SORTIE      : " << std::setw(3) << network.get_size(network.num_layers()-1) << " neurones" << std::endl;
    std::cout << "+-" << std::endl;
    
    // Calcul du nombre total de paramètres
    int total_weights = 0;
    int total_biases = 0;
    for (int i = 0; i < network.num_layers() - 1; ++i) {
        int in_size = network.get_size(i);
        int out_size = network.get_size(i + 1);
        total_weights += in_size * out_size;
        total_biases += out_size;
    }
    
    std::cout << "\nStatistiques:" << std::endl;
    std::cout << "   - Nombre total de couches    : " << network.num_layers() << std::endl;
    std::cout << "   - Nombre de couches cachees  : " << num_hidden << std::endl;
    std::cout << "   - Nombre total de poids      : " << total_weights << std::endl;
    std::cout << "   - Nombre total de biais      : " << total_biases << std::endl;
    std::cout << "   - TOTAL PARAMETRES           : " << (total_weights + total_biases) << std::endl;
    
    // Informations sur l'optimiseur
    auto* opt = network.get_optimizer();
    if (opt) {
        std::cout << "\nOptimiseur: " << opt->get_info() << std::endl;
    }
}

void print_training_progress(int epoch, int total_epochs, real cost, bool is_final = false) {
    if (is_final) {
        std::cout << "| FINAL   ";
    } else {
        std::cout << "| Epoque " << std::setw(4) << epoch;
    }
    
    // Barre de progression
    if (!is_final) {
        int progress = (epoch * 30) / total_epochs;
        std::cout << " [";
        for (int i = 0; i < 30; ++i) {
            if (i < progress) std::cout << "#";
            else if (i == progress) std::cout << ">";
            else std::cout << ".";
        }
        std::cout << "] ";
        std::cout << std::setw(3) << (epoch * 100 / total_epochs) << "%";
    } else {
        std::cout << " [##############################] 100%";
    }
    
    // Coût
    std::cout << " | Cout: ";
    if (cost < 1e-6) {
        std::cout << std::scientific << std::setprecision(2) << cost;
    } else if (cost < 1e-3) {
        std::cout << std::scientific << std::setprecision(2) << cost;
    } else {
        std::cout << std::fixed << std::setprecision(6) << cost;
    }
    std::cout << std::endl;
}

void print_predictions_header(const std::string& task_type) {
    print_section_header("PREDICTIONS FINALES - " + task_type, '=', 70);
}

// === UTILITAIRE GENERIQUE : PARTITION TRAIN / TEST ===
// Retourne {train_indices, test_indices} après mélange aléatoire.
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

// === STRUCTURES ET FONCTIONS GENERIQUES POUR FACTORISATION ===

struct TrainingConfig {
    std::string problem_name;
    std::string description;
    std::map<int, int> network_sizes;
    std::vector<std::string> activations;
    real sgd_lr, sgd_epochs, adam_lr, adam_epochs;
    int batch_size;
    int num_samples;
    real convergence_threshold;
    int progress_frequency;
    
    // === NOUVEAUX PARAMETRES POUR SPARSITÉ DYNAMIQUE ===
    bool enable_dynamic_sparsity = false;     // Activer la sparsité dynamique
    real sparsity_threshold = 0.01;           // Seuil de sparsité
    int sparsity_frequency = 50;              // Fréquence d'application (en époques)
    int sparsity_start_epoch = 100;           // Époque de début de la sparsité
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

// Fonction générique d'entraînement
void generic_train_network(Network& network, TrainingData& data, const TrainingConfig& config, 
                          const std::string& optimizer_choice, std::mt19937& gen) {
    int epochs = get_epochs(optimizer_choice, config);
    real learning_rate = get_learning_rate(optimizer_choice, config);
    
    print_section_header("ENTRAINEMENT EN COURS", '-', 70);
    std::cout << "+- Parametres:" << std::endl;
    std::cout << "|  - Epoques         : " << epochs << std::endl;
    std::cout << "|  - Taille batch    : " << config.batch_size << std::endl;
    std::cout << "|  - Echantillons    : " << data.train_indices.size() << std::endl;
    std::cout << "|  - Taux apprentis. : " << learning_rate << std::endl;
    
    // Affichage des paramètres de sparsité dynamique
    if (config.enable_dynamic_sparsity) {
        std::cout << "|  - Sparsité dynamique : ACTIVÉE" << std::endl;
        std::cout << "|    * Seuil          : " << config.sparsity_threshold << std::endl;
        std::cout << "|    * Fréquence      : toutes les " << config.sparsity_frequency << " époques" << std::endl;
        std::cout << "|    * Début          : époque " << config.sparsity_start_epoch << std::endl;
    } else {
        std::cout << "|  - Sparsité dynamique : DÉSACTIVÉE" << std::endl;
    }
    
    std::cout << "+-" << std::endl;
    std::cout << "\n+- Progression:" << std::endl;

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
        
        // === SPARSITÉ DYNAMIQUE ===
        bool sparsity_applied = false;
        if (config.enable_dynamic_sparsity && 
            epoch >= config.sparsity_start_epoch && 
            (epoch - config.sparsity_start_epoch) % config.sparsity_frequency == 0) {
            
            network.apply_threshold_sparsity(config.sparsity_threshold);
            sparsity_applied = true;
        }
        
        // Affichage de progression avec indication de sparsité
        if ((epoch + 1) % (epochs / config.progress_frequency) == 0 || epoch == 0 || epoch == epochs - 1) {
            print_training_progress(epoch + 1, epochs, avg_cost, epoch == epochs - 1);
            if (sparsity_applied) {
                std::cout << "|   --> Sparsité appliquée à l'époque " << (epoch + 1) << std::endl;
            }
        }
        
        if (avg_cost < config.convergence_threshold) {
            std::cout << "| Convergence atteinte a l'epoque " << epoch + 1 << " !" << std::endl;
            break;
        }
    }
    std::cout << "+-" << std::endl;
}

// Fonction générique d'évaluation
void generic_evaluate_network(Network& network, TrainingData& data, const std::string& task_name) {
    print_predictions_header(task_name);
    View1D prediction_result("prediction_result", data.output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0;
    int correct_predictions = 0;
    
    // Calcul des statistiques sur le jeu de test
    for (int idx : data.test_indices) {
        auto input_subview = Kokkos::subview(data.inputs, idx, Kokkos::ALL());
        auto target_subview = Kokkos::subview(data.outputs, idx, Kokkos::ALL());
        View1D prediction = network.forward(input_subview);
        final_total_cost += network.calculate_cost(prediction, target_subview);
        
        // Pour classification binaire/multi-classe
        if (data.output_dim == 1) {
            Kokkos::deep_copy(prediction_result, prediction);
            Kokkos::deep_copy(h_prediction_result, prediction_result); 
            Kokkos::fence();
            int predicted_class = std::round(h_prediction_result(0));
            int target_class = static_cast<int>(data.h_outputs(idx, 0));
            if (predicted_class == target_class) correct_predictions++;
        }
    }
    
    real avg_cost = final_total_cost / data.test_indices.size();
    
    std::cout << "+- RESULTATS FINAUX:" << std::endl;
    std::cout << "|  - Cout final moyen (test): " << std::scientific << std::setprecision(4) << avg_cost << std::endl;
    
    if (data.output_dim == 1) {
        real accuracy = static_cast<real>(correct_predictions) / data.test_indices.size();
        std::cout << "|  - Precision (test) : " << std::fixed << std::setprecision(2) << accuracy * 100.0 << "%" << std::endl;
        std::cout << "|  - Echantillons OK  : " << correct_predictions << "/" << data.test_indices.size() << std::endl;
    } else {
        std::cout << "|  - Echantillons testes: " << data.test_indices.size() << std::endl;
    }
    std::cout << "+-" << std::endl;
}

// Configuration des cas de test simplifiés
TrainingConfig get_xor_config() {
    return {
        "XOR", "Classification binaire non-lineaire (XOR)",
        {{0, 2}, {1, 20}, {2, 10}, {3, 1}}, {"relu", "relu", "sigmoid"},
        0.3, 1200, 0.01, 800, 4, 4, 1e-8, 25  // Époques augmentées : SGD 800→1200, Adam 400→800, seuil 1e-6→1e-8
    };
}

TrainingConfig get_sine_config() {
    return {
        "SINUS", "Regression - Approximation de sin(x)",
        {{0, 1}, {1, 16}, {2, 16}, {3, 1}}, {"relu", "relu", "linear"},
        0.02, 600, 0.001, 800, 16, 1024, 1e-9, 25  // Époques augmentées : SGD 300→600, Adam 400→800, seuil 1e-7→1e-9
    };
}

TrainingConfig get_linear_config() {
    return {
        "SEPARATION LINEAIRE", "Classification binaire - Separation lineaire",
        {{0, 2}, {1, 1}}, {"sigmoid"},
        0.1, 120, 0.01, 160, 16, 800, 1e-7, 15  // Époques augmentées : SGD 60→120, Adam 80→160, seuil 1e-5→1e-7
    };
}

// === NOUVELLES CONFIGURATIONS AVEC SPARSITÉ DYNAMIQUE ===

TrainingConfig get_xor_dynamic_sparsity_config() {
    return {
        "XOR DYNAMIQUE", "XOR avec sparsité dynamique pendant l'entraînement",
        {{0, 2}, {1, 20}, {2, 10}, {3, 1}}, {"relu", "relu", "sigmoid"},
        0.3, 1500, 0.01, 1200, 4, 4, 1e-8, 30,  // Époques augmentées : SGD 1000→1500, Adam 800→1200, seuil 1e-6→1e-8
        true,  // enable_dynamic_sparsity
        0.025, // sparsity_threshold
        40,    // sparsity_frequency
        80     // sparsity_start_epoch
    };
}

TrainingConfig get_sine_dynamic_sparsity_config() {
    return {
        "SINUS DYNAMIQUE", "Sinus avec sparsité dynamique pendant l'entraînement",
        {{0, 1}, {1, 24}, {2, 24}, {3, 1}}, {"relu", "relu", "linear"},
        0.02, 900, 0.001, 1200, 16, 1024, 1e-9, 30,  // Époques augmentées : SGD 600→900, Adam 700→1200, seuil 1e-7→1e-9
        true,  // enable_dynamic_sparsity
        0.015, // sparsity_threshold
        30,    // sparsity_frequency
        60     // sparsity_start_epoch
    };
}

// Générateurs de données simplifiés
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
    
    // Pour XOR, on utilise tous les échantillons pour train et test
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
        } while (std::abs(y - x) < margin); // éviter la zone ambiguë
        
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

// Fonction générique de test simplifiée
void run_simple_test(const TrainingConfig& config, 
                     std::function<TrainingData(std::mt19937&)> data_generator,
                     const std::string& optimizer_choice) {
    print_separator("ENTRAINEMENT " + config.problem_name, '=', 80);
    std::cout << "Probleme: " << config.description << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));
    print_network_architecture(network);

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = data_generator(gen);
    
    generic_train_network(network, data, config, optimizer_choice, gen);
    generic_evaluate_network(network, data, config.problem_name);
}

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

TrainingConfig get_spiral_config() {
    return {
        "SPIRALES", "Classification multi-classes - Motifs en spirale (3 classes)",
        {{0, 2}, {1, 50}, {2, 25}, {3, 3}}, {"relu", "relu", "sigmoid"},
        0.3, 800, 0.005, 600, 32, 900, 1e-6, 25  // Époques augmentées : SGD 400→800, Adam 300→600, seuil 1e-4→1e-6
    };
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

TrainingConfig get_gaussian_config() {
    return {
        "CLUSTERS GAUSSIENS", "Classification multi-classes - Clusters gaussiens (4 classes)",
        {{0, 2}, {1, 40}, {2, 30}, {3, 4}}, {"relu", "relu", "sigmoid"},
        0.4, 600, 0.008, 400, 40, 800, 1e-6, 25  // Époques augmentées : SGD 300→600, Adam 200→400, seuil 1e-4→1e-6
    };
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

TrainingConfig get_timeseries_config() {
    return {
        "SERIES TEMPORELLES", "Regression sequentielle - Prediction de series temporelles",
        {{0, 10}, {1, 40}, {2, 30}, {3, 20}, {4, 10}, {5, 1}}, {"relu", "relu", "relu", "relu", "linear"},
        0.01, 1000, 0.005, 1000, 10, 2000, 1e-7, 25  // Époques augmentées : SGD 500→1000, Adam 500→1000, seuil 1e-5→1e-7
    };
}

TrainingData generate_timeseries_data(int num_samples, std::mt19937& gen) {
    TrainingData data;
    data.input_dim = 10; data.output_dim = 1;
    data.h_inputs = TrainingData::HostView2D("timeseries_inputs", num_samples, 10);
    data.h_outputs = TrainingData::HostView2D("timeseries_outputs", num_samples, 1);
    
    std::uniform_real_distribution<real> freq1_dist(0.1, 0.3);
    std::uniform_real_distribution<real> freq2_dist(0.2, 0.8);
    std::uniform_real_distribution<real> amp_dist(0.5, 1.5);
    std::uniform_real_distribution<real> phase_dist(0, 2 * M_PI);
    std::uniform_real_distribution<real> noise_dist(-0.1, 0.1);
    
    for (int sample = 0; sample < num_samples; ++sample) {
        real freq1 = freq1_dist(gen), freq2 = freq2_dist(gen);
        real amp1 = amp_dist(gen), amp2 = amp_dist(gen);
        real phase1 = phase_dist(gen), phase2 = phase_dist(gen);
        
        std::vector<real> time_series(11);
        for (int t = 0; t < 11; ++t) {
            time_series[t] = amp1 * std::sin(freq1 * t + phase1) + 
                           amp2 * std::sin(freq2 * t + phase2) + noise_dist(gen);
        }
        
        for (int i = 0; i < 10; ++i) {
            data.h_inputs(sample, i) = time_series[i];
        }
        data.h_outputs(sample, 0) = time_series[10];
    }
    
    data.inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_inputs);
    data.outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), data.h_outputs);
    
    auto [train_idx, test_idx] = train_test_split(num_samples, 0.8, gen);
    data.train_indices = std::move(train_idx);
    data.test_indices = std::move(test_idx);
    return data;
}

void gaussian_clusters_train(const std::string& optimizer_choice) {
    auto config = get_gaussian_config();
    run_simple_test(config, [&config](std::mt19937& gen) { return generate_gaussian_data(config.num_samples, gen); }, optimizer_choice);
}

void spiral_train(const std::string& optimizer_choice) {
    auto config = get_spiral_config();
    run_simple_test(config, [&config](std::mt19937& gen) { return generate_spiral_data(config.num_samples, gen); }, optimizer_choice);
}

void time_series_train(const std::string& optimizer_choice) {
    auto config = get_timeseries_config();
    run_simple_test(config, [&config](std::mt19937& gen) { return generate_timeseries_data(config.num_samples, gen); }, optimizer_choice);
}

// === NOUVELLES FONCTIONS : TESTS AVEC SPARSITÉ DYNAMIQUE ===

void test_dynamic_sparsity_xor(const std::string& optimizer_choice) {
    print_separator("SPARSITÉ DYNAMIQUE - XOR", '=', 80);
    std::cout << "Probleme: XOR avec sparsité dynamique pendant l'entraînement" << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    auto config = get_xor_dynamic_sparsity_config();
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));
    
    print_network_architecture(network);
    
    // Afficher l'état initial
    std::cout << "\n=== ÉTAT INITIAL (DENSE) ===" << std::endl;
    network.compute_sparsity_stats();

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_xor_data(gen);
    
    // Entraînement avec sparsité dynamique
    generic_train_network(network, data, config, optimizer_choice, gen);
    
    // Afficher l'état final après sparsité dynamique
    std::cout << "\n=== ÉTAT FINAL (APRÈS SPARSITÉ DYNAMIQUE) ===" << std::endl;
    network.compute_sparsity_stats();
    
    // Évaluation finale
    generic_evaluate_network(network, data, "XOR SPARSITÉ DYNAMIQUE");
}

void test_dynamic_sparsity_sine(const std::string& optimizer_choice) {
    print_separator("SPARSITÉ DYNAMIQUE - SINUS", '=', 80);
    std::cout << "Probleme: Régression sinus avec sparsité dynamique pendant l'entraînement" << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    auto config = get_sine_dynamic_sparsity_config();
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));
    
    print_network_architecture(network);
    
    // Afficher l'état initial
    std::cout << "\n=== ÉTAT INITIAL (DENSE) ===" << std::endl;
    network.compute_sparsity_stats();

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_sine_data(config.num_samples, gen);
    
    // Entraînement avec sparsité dynamique
    generic_train_network(network, data, config, optimizer_choice, gen);
    
    // Afficher l'état final après sparsité dynamique
    std::cout << "\n=== ÉTAT FINAL (APRÈS SPARSITÉ DYNAMIQUE) ===" << std::endl;
    network.compute_sparsity_stats();
    
    // Évaluation finale
    generic_evaluate_network(network, data, "SINUS SPARSITÉ DYNAMIQUE");
}

// === FONCTION DE TEST RAPIDE POUR VALIDATION ===

void test_quick_dynamic_sparsity_xor(const std::string& optimizer_choice) {
    print_separator("TEST RAPIDE - SPARSITÉ DYNAMIQUE XOR", '=', 80);
    std::cout << "Probleme: Test rapide des nouveaux paramètres de sparsité" << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    // Configuration avec moins d'époques pour test rapide
    TrainingConfig config = {
        "XOR TEST RAPIDE", "XOR avec sparsité dynamique - Test rapide",
        {{0, 2}, {1, 20}, {2, 10}, {3, 1}}, {"relu", "relu", "sigmoid"},
        0.3, 200, 0.01, 150, 4, 4, 1e-6, 10,  // Seulement 150 époques pour test rapide
        true,  // enable_dynamic_sparsity
        0.025, // sparsity_threshold NOUVEAU : plus agressif
        25,    // sparsity_frequency : toutes les 25 époques
        50     // sparsity_start_epoch : commence à l'époque 50
    };
    
    auto optimizer = create_optimizer(optimizer_choice, config);
    Network network(config.network_sizes, config.activations, std::move(optimizer));
    
    print_network_architecture(network);
    
    std::cout << "\n=== ÉTAT INITIAL ===" << std::endl;
    network.compute_sparsity_stats();

    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto data = generate_xor_data(gen);
    
    // Entraînement avec sparsité dynamique
    generic_train_network(network, data, config, optimizer_choice, gen);
    
    std::cout << "\n=== ÉTAT FINAL ===" << std::endl;
    network.compute_sparsity_stats();
    
    generic_evaluate_network(network, data, "XOR TEST RAPIDE");
} 