#include "training.hpp"
#include "optimizers.hpp"

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

void xor_train(const std::string& optimizer_choice) {
    print_separator("ENTRAINEMENT XOR", '=', 80);
    std::cout << "Probleme: Classification binaire non-lineaire (XOR)" << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    // Structure du réseau
    std::map<int, int> sizes;
    sizes[0] = 2; sizes[1] = 80; sizes[2] = 40; sizes[3] = 1;
    std::vector<std::string> activations = {"relu", "relu", "sigmoid"};

    // Paramètres
    real learning_rate; int epochs; int batch_size = 4;
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.8; epochs = 1500;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.01; epochs = 800;
        optimizer = std::make_unique<Adam>(learning_rate);
    }

    Network dnn(sizes, activations, std::move(optimizer));
    
    // Affichage de l'architecture
    print_network_architecture(dnn);

    // Données XOR
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int num_samples = 4;
    const int input_dim = sizes.at(0);
    const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_xor_inputs("h_xor_inputs", num_samples, input_dim);
    HostView2D h_xor_outputs("h_xor_outputs", num_samples, output_dim);
    h_xor_inputs(0, 0) = 0.0; h_xor_inputs(0, 1) = 0.0; h_xor_outputs(0, 0) = 0.0;
    h_xor_inputs(1, 0) = 1.0; h_xor_inputs(1, 1) = 0.0; h_xor_outputs(1, 0) = 1.0;
    h_xor_inputs(2, 0) = 0.0; h_xor_inputs(2, 1) = 1.0; h_xor_outputs(2, 0) = 1.0;
    h_xor_inputs(3, 0) = 1.0; h_xor_inputs(3, 1) = 1.0; h_xor_outputs(3, 0) = 0.0;
    auto xor_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_inputs);
    auto xor_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_outputs);

    print_section_header("ENTRAINEMENT EN COURS", '-', 70);
    std::cout << "+- Parametres:" << std::endl;
    std::cout << "|  - Epoques         : " << epochs << std::endl;
    std::cout << "|  - Taille batch    : " << batch_size << std::endl;
    std::cout << "|  - Echantillons    : " << num_samples << std::endl;
    std::cout << "|  - Taux apprentis. : " << learning_rate << std::endl;
    std::cout << "+-" << std::endl;
    std::cout << "\n+- Progression:" << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        dnn.zero_accumulated_gradients();
        for (int i = 0; i < num_samples; ++i) {
            auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
            auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
            View1D prediction = dnn.forward(input_subview);
            dnn.backward(target_subview);
        }
        dnn.update(batch_size);

        // Recalculate cost for reporting
        real current_total_cost = 0.0;
        for (int i = 0; i < num_samples; ++i) {
             auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
             auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
             View1D prediction = dnn.forward(input_subview);
             current_total_cost += dnn.calculate_cost(prediction, target_subview);
        }
        real avg_cost = current_total_cost / num_samples;

        if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs -1) {
            print_training_progress(epoch + 1, epochs, avg_cost, epoch == epochs - 1);
        }
        if (avg_cost < 1e-4) {
             std::cout << "| Convergence atteinte a l'epoque " << epoch + 1 << " !" << std::endl;
             break;
        }
    }
    std::cout << "+-" << std::endl;

    print_predictions_header("XOR");
    View1D prediction_result("prediction_result", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0;
    
    std::cout << "+- Table de verite XOR:" << std::endl;
    std::cout << "|" << std::endl;
    std::cout << "|  Entree      | Cible | Prediction | Arrondi | Status" << std::endl;
    std::cout << "| ---------------------------------------------------" << std::endl;
    
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        
        real predicted = h_prediction_result(0);
        int rounded = std::round(predicted);
        int target = static_cast<int>(h_xor_outputs(i,0));
        
        std::cout << "| [" << std::fixed << std::setprecision(1) << h_xor_inputs(i,0) 
                  << ", " << h_xor_inputs(i,1) << "]     |   " << target << "   |   " 
                  << std::setprecision(4) << std::setw(7) << predicted << "   |    " << rounded << "    | ";
        
        if (rounded == target) {
            std::cout << "OK";
        } else {
            std::cout << "ERR";
        }
        std::cout << std::endl;
    }
    std::cout << "|" << std::endl;
    std::cout << "| Cout final moyen: " << std::scientific << std::setprecision(4) << final_total_cost / num_samples << std::endl;
    std::cout << "+-" << std::endl;
}

void sine_train(const std::string& optimizer_choice) {
    print_separator("APPROXIMATION FONCTION SINUS", '=', 80);
    std::cout << "Probleme: Regression - Approximation de sin(x)" << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    std::map<int, int> sizes;
    sizes[0] = 1; sizes[1] = 32; sizes[2] = 32; sizes[3] = 1;
    std::vector<std::string> activations = {"relu", "relu", "linear"};

    real learning_rate; int epochs; int batch_size = 16; int num_samples = 2048*4;
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.02; epochs = 500;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.001; epochs = 800;
        optimizer = std::make_unique<Adam>(learning_rate);
    }

    Network dnn(sizes, activations, std::move(optimizer));
    
    // Affichage de l'architecture
    print_network_architecture(dnn);

    using HostView1D = Kokkos::View<real*, Kokkos::HostSpace>;
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0); const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_inputs("h_sine_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_sine_outputs", num_samples, output_dim);
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<real> distrib(-M_PI, M_PI);
    for(int i=0; i < num_samples; ++i) {
        real x = distrib(gen); h_inputs(i, 0) = x; h_outputs(i, 0) = std::sin(x);
    }
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    print_section_header("ENTRAINEMENT EN COURS", '-', 60);
    std::cout << "+- Parametres:" << std::endl;
    std::cout << "|  - Epoques        : " << epochs << std::endl;
    std::cout << "|  - Taille batch   : " << batch_size << std::endl;
    std::cout << "|  - Echantillons   : " << num_samples << std::endl;
    std::cout << "|  - Taux apprentis.: " << learning_rate << std::endl;
    std::cout << "|  - Domaine       : [-PI, PI]" << std::endl;
    std::cout << "+-" << std::endl;
    std::cout << "\n+- Progression:" << std::endl;

    std::vector<int> indices(num_samples); std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        std::shuffle(indices.begin(), indices.end(), gen);
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - batch_start);
            if (current_batch_size <= 0) continue;
            dnn.zero_accumulated_gradients();
            for (int j = 0; j < current_batch_size; ++j) {
                int sample_index = indices[batch_start + j];
                auto input_subview = Kokkos::subview(train_inputs, sample_index, Kokkos::ALL());
                auto target_subview = Kokkos::subview(train_outputs, sample_index, Kokkos::ALL());
                View1D prediction = dnn.forward(input_subview);
                total_epoch_cost += dnn.calculate_cost(prediction, target_subview);
                dnn.backward(target_subview);
            }
            dnn.update(current_batch_size);
        }
        real avg_cost = total_epoch_cost / num_samples;
        if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs - 1) {
             print_training_progress(epoch + 1, epochs, avg_cost, epoch == epochs - 1);
        }
    }
    std::cout << "+-" << std::endl;

    print_predictions_header("SINUS");
    View1D prediction_result("prediction_result_sine", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; int num_test_samples = std::min(num_samples, 10);
    
    std::cout << "+- Echantillons de prediction:" << std::endl;
    std::cout << "|" << std::endl;
    std::cout << "|    x     | sin(x) cible | Prediction | Erreur abs | Qualite" << std::endl;
    std::cout << "| --------------------------------------------------------" << std::endl;
    
    for (int i = 0; i < num_test_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        real input_x = h_inputs(i, 0); 
        real target_y = h_outputs(i, 0);
        real predicted_y = h_prediction_result(0);
        real abs_error = std::abs(target_y - predicted_y);
        
        std::cout << "| " << std::fixed << std::setprecision(4) << std::setw(7) << input_x << " | " 
                  << std::setw(11) << target_y << " | " << std::setw(10) << predicted_y << " | " 
                  << std::setw(10) << abs_error << " | ";
        
        if (abs_error < 0.001) {
            std::cout << "Excellent";
        } else if (abs_error < 0.01) {
            std::cout << "Bon";
        } else {
            std::cout << "Moyen";
        }
        std::cout << std::endl;
    }
    std::cout << "|" << std::endl;
    std::cout << "| Cout final moyen: " << std::scientific << std::setprecision(4) << final_total_cost / num_test_samples << std::endl;
    std::cout << "+-" << std::endl;
}

void linear_sep_train(const std::string& optimizer_choice) {
    print_separator("SEPARATION LINEAIRE", '=', 80);
    std::cout << "Probleme: Classification binaire - Separation lineaire" << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    std::map<int, int> sizes; sizes[0] = 2; sizes[1] = 1;
    std::vector<std::string> activations = {"sigmoid"};

    real learning_rate; int epochs; int batch_size = 16; int num_samples = 2560;
    std::unique_ptr<Optimizer> optimizer;
    if (optimizer_choice == "sgd") {
        learning_rate = 0.1; epochs = 100;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.01; epochs = 150;
        optimizer = std::make_unique<Adam>(learning_rate);
    }

    Network dnn(sizes, activations, std::move(optimizer));
    
    // Affichage de l'architecture
    print_network_architecture(dnn);

    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0); const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_inputs("h_linear_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_linear_outputs", num_samples, output_dim);
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + 123);
    std::uniform_real_distribution<real> distrib(-1.0, 1.0);
    real margin = 0.1; int count_class0 = 0; int count_class1 = 0;
    for(int i=0; i < num_samples; ++i) {
        real x = distrib(gen); real y = distrib(gen);
        h_inputs(i, 0) = x; h_inputs(i, 1) = y;
        if (y < x - margin) { h_outputs(i, 0) = 0.0; count_class0++; }
        else if (y > x + margin) { h_outputs(i, 0) = 1.0; count_class1++; }
        else { i--; continue; }
    }
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    print_section_header("GENERATION DES DONNEES", '-', 60);
    std::cout << "+- Classes generees:" << std::endl;
    std::cout << "|  - Classe 0 (y < x - " << margin << ") : " << count_class0 << " echantillons" << std::endl;
    std::cout << "|  - Classe 1 (y > x + " << margin << ") : " << count_class1 << " echantillons" << std::endl;
    std::cout << "|  - Total                      : " << num_samples << " echantillons" << std::endl;
    std::cout << "+-" << std::endl;

    print_section_header("ENTRAINEMENT EN COURS", '-', 60);
    std::cout << "+- Parametres:" << std::endl;
    std::cout << "|  - Epoques        : " << epochs << std::endl;
    std::cout << "|  - Taille batch   : " << batch_size << std::endl;
    std::cout << "|  - Echantillons   : " << num_samples << std::endl;
    std::cout << "|  - Taux apprentis.: " << learning_rate << std::endl;
    std::cout << "|  - Marge separat.: " << margin << std::endl;
    std::cout << "+-" << std::endl;
    std::cout << "\n+- Progression:" << std::endl;

    std::vector<int> indices(num_samples); std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        std::shuffle(indices.begin(), indices.end(), gen);
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - batch_start);
             if (current_batch_size <= 0) continue;
            dnn.zero_accumulated_gradients();
            for (int j = 0; j < current_batch_size; ++j) {
                int sample_index = indices[batch_start + j];
                auto input_subview = Kokkos::subview(train_inputs, sample_index, Kokkos::ALL());
                auto target_subview = Kokkos::subview(train_outputs, sample_index, Kokkos::ALL());
                View1D prediction = dnn.forward(input_subview);
                total_epoch_cost += dnn.calculate_cost(prediction, target_subview);
                dnn.backward(target_subview);
            }
            dnn.update(current_batch_size);
        }
        real avg_cost = total_epoch_cost / num_samples;
        if ((epoch + 1) % (epochs / 10) == 0 || epoch == 0 || epoch == epochs - 1) {
             print_training_progress(epoch + 1, epochs, avg_cost, epoch == epochs - 1);
        }
         if (avg_cost < 1e-3) {
             std::cout << "| Convergence atteinte a l'epoque " << epoch + 1 << " !" << std::endl;
             break;
         }
    }
    std::cout << "+-" << std::endl;

    print_predictions_header("SEPARATION LINEAIRE");
    View1D prediction_result("prediction_result_linear", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; int correct_predictions = 0;
    
    // Calcul des statistiques complètes
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        real predicted_value = h_prediction_result(0); 
        int predicted_class = std::round(predicted_value);
        int target_class = static_cast<int>(h_outputs(i, 0));
        if (predicted_class == target_class) correct_predictions++;
    }
    real accuracy = static_cast<real>(correct_predictions) / num_samples;
    
    std::cout << "+- Echantillons de prediction (premiers 10):" << std::endl;
    std::cout << "|" << std::endl;
    std::cout << "|  Coordonnees  | Cible | Prediction | Classe | Status" << std::endl;
    std::cout << "| ------------------------------------------------" << std::endl;
    
    // Affichage des 10 premiers échantillons
    for (int i = 0; i < std::min(10, num_samples); ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        real predicted_value = h_prediction_result(0); 
        int predicted_class = std::round(predicted_value);
        int target_class = static_cast<int>(h_outputs(i, 0));
        
        std::cout << "| [" << std::fixed << std::setprecision(2) << std::setw(5) << h_inputs(i,0) 
                  << "," << std::setw(5) << h_inputs(i,1) << "] |   " << target_class << "   |   " 
                  << std::setprecision(3) << std::setw(7) << predicted_value << "   |    " 
                  << predicted_class << "    | ";
        
        if (predicted_class == target_class) {
            std::cout << "OK";
        } else {
            std::cout << "ERR";
        }
        std::cout << std::endl;
    }
    
    std::cout << "|" << std::endl;
    std::cout << "+- RESULTATS FINAUX:" << std::endl;
    std::cout << "|  - Cout final moyen : " << std::scientific << std::setprecision(4) << final_total_cost / num_samples << std::endl;
    std::cout << "|  - Precision        : " << std::fixed << std::setprecision(2) << accuracy * 100.0 << "%" << std::endl;
    std::cout << "|  - Echantillons OK  : " << correct_predictions << "/" << num_samples << std::endl;
    
    if (accuracy >= 0.95) {
        std::cout << "|  - Qualite         : Excellent (>=95%)" << std::endl;
    } else if (accuracy >= 0.85) {
        std::cout << "|  - Qualite         : Bon (>=85%)" << std::endl;
    } else {
        std::cout << "|  - Qualite         : A ameliorer (<85%)" << std::endl;
    }
    std::cout << "+-" << std::endl;
}

void spiral_train(const std::string& optimizer_choice) {
    print_separator("CLASSIFICATION SPIRALES", '=', 80);
    std::cout << "Probleme: Classification multi-classes - Motifs en spirale (3 classes)" << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    // Structure du réseau pour classification 3 classes
    std::map<int, int> sizes;
    sizes[0] = 2; sizes[1] = 100; sizes[2] = 50; sizes[3] = 3;  // 3 classes de sortie
    std::vector<std::string> activations = {"relu", "relu", "sigmoid"};

    // Paramètres d'entraînement
    real learning_rate; 
    int epochs; 
    int batch_size = 32; 
    int num_samples = 1200; // 400 échantillons par classe
    std::unique_ptr<Optimizer> optimizer;
    
    if (optimizer_choice == "sgd") {
        learning_rate = 0.3; 
        epochs = 800;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.005; 
        epochs = 500;
        optimizer = std::make_unique<Adam>(learning_rate);
    }

    Network dnn(sizes, activations, std::move(optimizer));
    
    // Affichage de l'architecture
    print_network_architecture(dnn);

    // Génération des données en spirale
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0); 
    const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_inputs("h_spiral_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_spiral_outputs", num_samples, output_dim);
    
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + 456);
    std::uniform_real_distribution<real> noise_distrib(-0.1, 0.1);
    
    print_section_header("GENERATION DES DONNEES SPIRALES", '-', 70);
    std::cout << "+- Parametres de generation:" << std::endl;
    std::cout << "|  - Nombre de classes      : 3" << std::endl;
    std::cout << "|  - Echantillons par classe: " << num_samples/3 << std::endl;
    std::cout << "|  - Bruit ajoute           : [-0.1, 0.1]" << std::endl;
    std::cout << "+-" << std::endl;
    
    int samples_per_class = num_samples / 3;
    for (int class_id = 0; class_id < 3; ++class_id) {
        for (int i = 0; i < samples_per_class; ++i) {
            int sample_idx = class_id * samples_per_class + i;
            
            // Paramètres de la spirale pour chaque classe
            real t = static_cast<real>(i) / samples_per_class * 2.0 * M_PI; // Angle de 0 à 2π
            real radius = 0.3 + t * 0.15; // Rayon croissant avec l'angle
            real angle_offset = class_id * 2.0 * M_PI / 3.0; // Décalage de 120° entre classes
            
            // Coordonnées de base de la spirale
            real x = radius * std::cos(t + angle_offset);
            real y = radius * std::sin(t + angle_offset);
            
            // Ajout de bruit
            x += noise_distrib(gen);
            y += noise_distrib(gen);
            
            h_inputs(sample_idx, 0) = x;
            h_inputs(sample_idx, 1) = y;
            
            // Encodage one-hot pour les classes
            for (int j = 0; j < 3; ++j) {
                h_outputs(sample_idx, j) = (j == class_id) ? 1.0 : 0.0;
            }
        }
    }
    
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    print_section_header("ENTRAINEMENT EN COURS", '-', 70);
    std::cout << "+- Parametres:" << std::endl;
    std::cout << "|  - Epoques         : " << epochs << std::endl;
    std::cout << "|  - Taille batch    : " << batch_size << std::endl;
    std::cout << "|  - Echantillons    : " << num_samples << std::endl;
    std::cout << "|  - Taux apprentis. : " << learning_rate << std::endl;
    std::cout << "+-" << std::endl;
    std::cout << "\n+- Progression:" << std::endl;

    std::vector<int> indices(num_samples); 
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        std::shuffle(indices.begin(), indices.end(), gen);
        
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - batch_start);
            if (current_batch_size <= 0) continue;
            
            dnn.zero_accumulated_gradients();
            for (int j = 0; j < current_batch_size; ++j) {
                int sample_index = indices[batch_start + j];
                auto input_subview = Kokkos::subview(train_inputs, sample_index, Kokkos::ALL());
                auto target_subview = Kokkos::subview(train_outputs, sample_index, Kokkos::ALL());
                View1D prediction = dnn.forward(input_subview);
                total_epoch_cost += dnn.calculate_cost(prediction, target_subview);
                dnn.backward(target_subview);
            }
            dnn.update(current_batch_size);
        }
        
        real avg_cost = total_epoch_cost / num_samples;
        if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs - 1) {
            print_training_progress(epoch + 1, epochs, avg_cost, epoch == epochs - 1);
        }
        
        if (avg_cost < 1e-4) {
            std::cout << "| Convergence atteinte a l'epoque " << epoch + 1 << " !" << std::endl;
            break;
        }
    }
    std::cout << "+-" << std::endl;

    print_predictions_header("SPIRALES - CLASSIFICATION 3 CLASSES");
    View1D prediction_result("prediction_result_spiral", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; 
    int correct_predictions = 0;
    std::vector<int> class_correct(3, 0);
    std::vector<int> class_total(3, 0);
    
    // Calcul des statistiques complètes
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); 
        Kokkos::fence();
        
        // Trouver la classe prédite (argmax)
        int predicted_class = 0;
        real max_pred = h_prediction_result(0);
        for (int c = 1; c < 3; ++c) {
            if (h_prediction_result(c) > max_pred) {
                max_pred = h_prediction_result(c);
                predicted_class = c;
            }
        }
        
        // Trouver la vraie classe
        int true_class = 0;
        for (int c = 0; c < 3; ++c) {
            if (h_outputs(i, c) > 0.5) {
                true_class = c;
                break;
            }
        }
        
        class_total[true_class]++;
        if (predicted_class == true_class) {
            correct_predictions++;
            class_correct[true_class]++;
        }
    }
    
    real accuracy = static_cast<real>(correct_predictions) / num_samples;
    
    std::cout << "+- Echantillons de prediction (premiers 12):" << std::endl;
    std::cout << "|" << std::endl;
    std::cout << "|  Coordonnees  | Vraie | Predictions [C0, C1, C2] | Pred. | Status" << std::endl;
    std::cout << "| -------------------------------------------------------------------" << std::endl;
    
    // Affichage des 12 premiers échantillons (4 par classe)
    for (int class_id = 0; class_id < 3; ++class_id) {
        for (int k = 0; k < 4; ++k) {
            int i = class_id * samples_per_class + k;
            auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
            auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
            View1D prediction = dnn.forward(input_subview);
            
            Kokkos::deep_copy(prediction_result, prediction);
            Kokkos::deep_copy(h_prediction_result, prediction_result); 
            Kokkos::fence();
            
            int predicted_class = 0;
            real max_pred = h_prediction_result(0);
            for (int c = 1; c < 3; ++c) {
                if (h_prediction_result(c) > max_pred) {
                    max_pred = h_prediction_result(c);
                    predicted_class = c;
                }
            }
            
            std::cout << "| [" << std::fixed << std::setprecision(2) << std::setw(5) << h_inputs(i,0) 
                      << "," << std::setw(5) << h_inputs(i,1) << "] |   " << class_id << "   | [" 
                      << std::setprecision(3) << std::setw(4) << h_prediction_result(0) << "," 
                      << std::setw(4) << h_prediction_result(1) << ","
                      << std::setw(4) << h_prediction_result(2) << "] |   " 
                      << predicted_class << "   | ";
            
            if (predicted_class == class_id) {
                std::cout << "OK";
            } else {
                std::cout << "ERR";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "|" << std::endl;
    std::cout << "+- RESULTATS FINAUX:" << std::endl;
    std::cout << "|  - Cout final moyen : " << std::scientific << std::setprecision(4) << final_total_cost / num_samples << std::endl;
    std::cout << "|  - Precision globale: " << std::fixed << std::setprecision(2) << accuracy * 100.0 << "%" << std::endl;
    std::cout << "|  - Echantillons OK  : " << correct_predictions << "/" << num_samples << std::endl;
    std::cout << "|" << std::endl;
    std::cout << "|  - Precision par classe:" << std::endl;
    
    for (int c = 0; c < 3; ++c) {
        real class_accuracy = (class_total[c] > 0) ? static_cast<real>(class_correct[c]) / class_total[c] : 0.0;
        std::cout << "|    * Spirale " << c << "       : " << std::setprecision(1) << class_accuracy * 100.0 
                  << "% (" << class_correct[c] << "/" << class_total[c] << ")" << std::endl;
    }
    
    if (accuracy >= 0.90) {
        std::cout << "|  - Qualite         : Excellent (>=90%)" << std::endl;
    } else if (accuracy >= 0.75) {
        std::cout << "|  - Qualite         : Bon (>=75%)" << std::endl;
    } else {
        std::cout << "|  - Qualite         : A ameliorer (<75%)" << std::endl;
    }
    std::cout << "+-" << std::endl;
}

void gaussian_clusters_train(const std::string& optimizer_choice) {
    print_separator("CLASSIFICATION CLUSTERS GAUSSIENS", '=', 80);
    std::cout << "Probleme: Classification multi-classes - Clusters gaussiens (4 classes)" << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    // Structure du réseau pour classification 4 classes
    std::map<int, int> sizes;
    sizes[0] = 2; sizes[1] = 80; sizes[2] = 60; sizes[3] = 4;  // 4 classes de sortie
    std::vector<std::string> activations = {"relu", "relu", "sigmoid"};

    // Paramètres d'entraînement
    real learning_rate; 
    int epochs; 
    int batch_size = 40; 
    int num_samples = 1600; // 400 échantillons par classe
    std::unique_ptr<Optimizer> optimizer;
    
    if (optimizer_choice == "sgd") {
        learning_rate = 0.4; 
        epochs = 600;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.008; 
        epochs = 400;
        optimizer = std::make_unique<Adam>(learning_rate);
    }

    Network dnn(sizes, activations, std::move(optimizer));
    
    // Affichage de l'architecture
    print_network_architecture(dnn);

    // Génération des données de clusters gaussiens
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0); 
    const int output_dim = sizes.at(sizes.size()-1);
    HostView2D h_inputs("h_gaussian_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_gaussian_outputs", num_samples, output_dim);
    
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + 789);
    
    // Définition des centres et variances des 4 clusters
    std::vector<std::array<real, 2>> cluster_centers = {
        {-1.5, -1.5},  // Cluster 0: bas-gauche
        { 1.5, -1.5},  // Cluster 1: bas-droite  
        {-1.5,  1.5},  // Cluster 2: haut-gauche
        { 1.5,  1.5}   // Cluster 3: haut-droite
    };
    
    std::vector<real> cluster_std = {0.4, 0.35, 0.45, 0.38}; // Écarts-types différents
    
    print_section_header("GENERATION DES CLUSTERS GAUSSIENS", '-', 70);
    std::cout << "+- Parametres de generation:" << std::endl;
    std::cout << "|  - Nombre de classes      : 4" << std::endl;
    std::cout << "|  - Echantillons par classe: " << num_samples/4 << std::endl;
    std::cout << "|  - Centres des clusters   :" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "|    * Cluster " << i << ": (" << std::fixed << std::setprecision(1) 
                  << cluster_centers[i][0] << ", " << cluster_centers[i][1] 
                  << ") σ=" << cluster_std[i] << std::endl;
    }
    std::cout << "+-" << std::endl;
    
    int samples_per_class = num_samples / 4;
    
    for (int class_id = 0; class_id < 4; ++class_id) {
        std::normal_distribution<real> normal_x(cluster_centers[class_id][0], cluster_std[class_id]);
        std::normal_distribution<real> normal_y(cluster_centers[class_id][1], cluster_std[class_id]);
        
        for (int i = 0; i < samples_per_class; ++i) {
            int sample_idx = class_id * samples_per_class + i;
            
            // Génération selon distribution gaussienne
            real x = normal_x(gen);
            real y = normal_y(gen);
            
            h_inputs(sample_idx, 0) = x;
            h_inputs(sample_idx, 1) = y;
            
            // Encodage one-hot pour les classes
            for (int j = 0; j < 4; ++j) {
                h_outputs(sample_idx, j) = (j == class_id) ? 1.0 : 0.0;
            }
        }
    }
    
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    print_section_header("ENTRAINEMENT EN COURS", '-', 70);
    std::cout << "+- Parametres:" << std::endl;
    std::cout << "|  - Epoques         : " << epochs << std::endl;
    std::cout << "|  - Taille batch    : " << batch_size << std::endl;
    std::cout << "|  - Echantillons    : " << num_samples << std::endl;
    std::cout << "|  - Taux apprentis. : " << learning_rate << std::endl;
    std::cout << "+-" << std::endl;
    std::cout << "\n+- Progression:" << std::endl;

    std::vector<int> indices(num_samples); 
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        std::shuffle(indices.begin(), indices.end(), gen);
        
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - batch_start);
            if (current_batch_size <= 0) continue;
            
            dnn.zero_accumulated_gradients();
            for (int j = 0; j < current_batch_size; ++j) {
                int sample_index = indices[batch_start + j];
                auto input_subview = Kokkos::subview(train_inputs, sample_index, Kokkos::ALL());
                auto target_subview = Kokkos::subview(train_outputs, sample_index, Kokkos::ALL());
                View1D prediction = dnn.forward(input_subview);
                total_epoch_cost += dnn.calculate_cost(prediction, target_subview);
                dnn.backward(target_subview);
            }
            dnn.update(current_batch_size);
        }
        
        real avg_cost = total_epoch_cost / num_samples;
        if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs - 1) {
            print_training_progress(epoch + 1, epochs, avg_cost, epoch == epochs - 1);
        }
        
        if (avg_cost < 1e-4) {
            std::cout << "| Convergence atteinte a l'epoque " << epoch + 1 << " !" << std::endl;
            break;
        }
    }
    std::cout << "+-" << std::endl;

    print_predictions_header("CLUSTERS GAUSSIENS - CLASSIFICATION 4 CLASSES");
    View1D prediction_result("prediction_result_gaussian", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; 
    int correct_predictions = 0;
    std::vector<int> class_correct(4, 0);
    std::vector<int> class_total(4, 0);
    
    // Matrice de confusion
    std::vector<std::vector<int>> confusion_matrix(4, std::vector<int>(4, 0));
    
    // Calcul des statistiques complètes
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); 
        Kokkos::fence();
        
        // Trouver la classe prédite (argmax)
        int predicted_class = 0;
        real max_pred = h_prediction_result(0);
        for (int c = 1; c < 4; ++c) {
            if (h_prediction_result(c) > max_pred) {
                max_pred = h_prediction_result(c);
                predicted_class = c;
            }
        }
        
        // Trouver la vraie classe
        int true_class = 0;
        for (int c = 0; c < 4; ++c) {
            if (h_outputs(i, c) > 0.5) {
                true_class = c;
                break;
            }
        }
        
        class_total[true_class]++;
        confusion_matrix[true_class][predicted_class]++;
        if (predicted_class == true_class) {
            correct_predictions++;
            class_correct[true_class]++;
        }
    }
    
    real accuracy = static_cast<real>(correct_predictions) / num_samples;
    
    std::cout << "+- Echantillons de prediction (3 premiers par classe):" << std::endl;
    std::cout << "|" << std::endl;
    std::cout << "|  Coordonnees  | Cluster | Predictions [C0,C1,C2,C3] | Pred | Status" << std::endl;
    std::cout << "| ----------------------------------------------------------------" << std::endl;
    
    // Affichage de 3 échantillons par classe
    for (int class_id = 0; class_id < 4; ++class_id) {
        for (int k = 0; k < 3; ++k) {
            int i = class_id * samples_per_class + k;
            auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
            auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
            View1D prediction = dnn.forward(input_subview);
            
            Kokkos::deep_copy(prediction_result, prediction);
            Kokkos::deep_copy(h_prediction_result, prediction_result); 
            Kokkos::fence();
            
            int predicted_class = 0;
            real max_pred = h_prediction_result(0);
            for (int c = 1; c < 4; ++c) {
                if (h_prediction_result(c) > max_pred) {
                    max_pred = h_prediction_result(c);
                    predicted_class = c;
                }
            }
            
            std::cout << "| [" << std::fixed << std::setprecision(2) << std::setw(5) << h_inputs(i,0) 
                      << "," << std::setw(5) << h_inputs(i,1) << "] |    " << class_id << "    | [" 
                      << std::setprecision(2) << std::setw(3) << h_prediction_result(0) << ","
                      << std::setw(3) << h_prediction_result(1) << "," << std::setw(3) << h_prediction_result(2) 
                      << "," << std::setw(3) << h_prediction_result(3) << "] |  " 
                      << predicted_class << "   | ";
            
            if (predicted_class == class_id) {
                std::cout << "OK";
            } else {
                std::cout << "ERR";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "|" << std::endl;
    std::cout << "+- RESULTATS FINAUX:" << std::endl;
    std::cout << "|  - Cout final moyen : " << std::scientific << std::setprecision(4) << final_total_cost / num_samples << std::endl;
    std::cout << "|  - Precision globale: " << std::fixed << std::setprecision(2) << accuracy * 100.0 << "%" << std::endl;
    std::cout << "|  - Echantillons OK  : " << correct_predictions << "/" << num_samples << std::endl;
    std::cout << "|" << std::endl;
    std::cout << "|  - Precision par cluster:" << std::endl;
    
    for (int c = 0; c < 4; ++c) {
        real class_accuracy = (class_total[c] > 0) ? static_cast<real>(class_correct[c]) / class_total[c] : 0.0;
        std::cout << "|    * Cluster " << c << "        : " << std::setprecision(1) << class_accuracy * 100.0 
                  << "% (" << class_correct[c] << "/" << class_total[c] << ")" << std::endl;
    }
    
    std::cout << "|" << std::endl;
    std::cout << "|  - Matrice de confusion:" << std::endl;
    std::cout << "|         |  Predite ->  | C0 | C1 | C2 | C3 |" << std::endl;
    std::cout << "|    -----|-------------|----|----|----|----|" << std::endl;
    for (int true_c = 0; true_c < 4; ++true_c) {
        std::cout << "|    Vraie|      C" << true_c << "     |";
        for (int pred_c = 0; pred_c < 4; ++pred_c) {
            std::cout << std::setw(3) << confusion_matrix[true_c][pred_c] << " |";
        }
        std::cout << std::endl;
    }
    
    if (accuracy >= 0.92) {
        std::cout << "|  - Qualite         : Excellent (>=92%)" << std::endl;
    } else if (accuracy >= 0.80) {
        std::cout << "|  - Qualite         : Bon (>=80%)" << std::endl;
    } else {
        std::cout << "|  - Qualite         : A ameliorer (<80%)" << std::endl;
    }
    std::cout << "+-" << std::endl;
}

void time_series_train(const std::string& optimizer_choice) {
    print_separator("PREDICTION SERIES TEMPORELLES", '=', 80);
    std::cout << "Probleme: Regression sequentielle - Prediction de series temporelles" << std::endl;
    std::cout << "Donnees: Fenetres temporelles de 10 points -> predire le 11eme point" << std::endl;
    std::cout << "Originalite: Frequences aleatoires par echantillon (pas de valeurs fixes!)" << std::endl;
    std::cout << "Optimiseur: " << optimizer_choice << std::endl;

    // Structure du réseau pour séries temporelles
    std::map<int, int> sizes;
    sizes[0] = 10; sizes[1] = 80; sizes[2] = 60; sizes[3] = 40; sizes[4] = 20; sizes[5] = 1;  // 10 entrées temporelles -> 1 prédiction
    std::vector<std::string> activations = {"relu", "relu", "relu", "relu", "linear"};

    // Paramètres d'entraînement
    real learning_rate; 
    int epochs; 
    int batch_size = 10; 
    int num_samples = 4000; // Plus d'échantillons pour la variabilité
    std::unique_ptr<Optimizer> optimizer;
    
    if (optimizer_choice == "sgd") {
        learning_rate = 0.01; 
        epochs = 1000;
        optimizer = std::make_unique<SGD>(learning_rate);
    } else { // adam default
        learning_rate = 0.005; 
        epochs = 1000;
        optimizer = std::make_unique<Adam>(learning_rate);
    }

    Network dnn(sizes, activations, std::move(optimizer));
    
    // Affichage de l'architecture
    print_network_architecture(dnn);

    // Génération des données de séries temporelles
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int input_dim = sizes.at(0); // 10 points temporels
    const int output_dim = sizes.at(sizes.size()-1); // 1 prédiction
    HostView2D h_inputs("h_timeseries_inputs", num_samples, input_dim);
    HostView2D h_outputs("h_timeseries_outputs", num_samples, output_dim);
    
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + 1001);
    std::uniform_real_distribution<real> freq1_dist(0.1, 0.8);  // Première fréquence aléatoire
    std::uniform_real_distribution<real> freq2_dist(0.2, 1.2);  // Deuxième fréquence aléatoire  
    std::uniform_real_distribution<real> amp1_dist(0.3, 1.0);   // Amplitude 1 aléatoire
    std::uniform_real_distribution<real> amp2_dist(0.2, 0.8);   // Amplitude 2 aléatoire
    std::uniform_real_distribution<real> phase_dist(0.0, 2.0 * M_PI); // Phase aléatoire
    std::uniform_real_distribution<real> noise_dist(-0.05, 0.05); // Bruit léger
    
    print_section_header("GENERATION DES SERIES TEMPORELLES", '-', 70);
    std::cout << "+- Parametres de generation (ALEATOIRES par echantillon):" << std::endl;
    std::cout << "|  - Nombre d'echantillons  : " << num_samples << std::endl;
    std::cout << "|  - Fenetre temporelle     : " << input_dim << " points" << std::endl;
    std::cout << "|  - Frequence 1            : [0.1, 0.8] (aleatoire)" << std::endl;
    std::cout << "|  - Frequence 2            : [0.2, 1.2] (aleatoire)" << std::endl;
    std::cout << "|  - Amplitude 1            : [0.3, 1.0] (aleatoire)" << std::endl;
    std::cout << "|  - Amplitude 2            : [0.2, 0.8] (aleatoire)" << std::endl;
    std::cout << "|  - Phase                  : [0, 2π] (aleatoire)" << std::endl;
    std::cout << "|  - Bruit                  : [-0.05, 0.05]" << std::endl;
    std::cout << "+-" << std::endl;
    
    // Génération des échantillons avec paramètres aléatoires
    for (int sample = 0; sample < num_samples; ++sample) {
        // Paramètres aléatoires pour cet échantillon
        real freq1 = freq1_dist(gen);
        real freq2 = freq2_dist(gen);
        real amp1 = amp1_dist(gen);
        real amp2 = amp2_dist(gen);
        real phase1 = phase_dist(gen);
        real phase2 = phase_dist(gen);
        
        // Générer une série temporelle de longueur (input_dim + 1)
        std::vector<real> time_series(input_dim + 1);
        for (int t = 0; t < input_dim + 1; ++t) {
            real base_signal = amp1 * std::sin(freq1 * t + phase1) + 
                              amp2 * std::sin(freq2 * t + phase2);
            time_series[t] = base_signal + noise_dist(gen);
        }
        
        // Les 10 premiers points sont l'entrée
        for (int i = 0; i < input_dim; ++i) {
            h_inputs(sample, i) = time_series[i];
        }
        
        // Le 11ème point est la cible à prédire
        h_outputs(sample, 0) = time_series[input_dim];
    }
    
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    print_section_header("ENTRAINEMENT EN COURS", '-', 70);
    std::cout << "+- Parametres:" << std::endl;
    std::cout << "|  - Epoques         : " << epochs << std::endl;
    std::cout << "|  - Taille batch    : " << batch_size << std::endl;
    std::cout << "|  - Echantillons    : " << num_samples << std::endl;
    std::cout << "|  - Taux apprentis. : " << learning_rate << std::endl;
    std::cout << "+-" << std::endl;
    std::cout << "\n+- Progression:" << std::endl;

    std::vector<int> indices(num_samples); 
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        std::shuffle(indices.begin(), indices.end(), gen);
        
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - batch_start);
            if (current_batch_size <= 0) continue;
            
            dnn.zero_accumulated_gradients();
            for (int j = 0; j < current_batch_size; ++j) {
                int sample_index = indices[batch_start + j];
                auto input_subview = Kokkos::subview(train_inputs, sample_index, Kokkos::ALL());
                auto target_subview = Kokkos::subview(train_outputs, sample_index, Kokkos::ALL());
                View1D prediction = dnn.forward(input_subview);
                total_epoch_cost += dnn.calculate_cost(prediction, target_subview);
                dnn.backward(target_subview);
            }
            dnn.update(current_batch_size);
        }
        
        real avg_cost = total_epoch_cost / num_samples;
        if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs - 1) {
            print_training_progress(epoch + 1, epochs, avg_cost, epoch == epochs - 1);
        }
        
        if (avg_cost < 1e-5) {
            std::cout << "| Convergence atteinte a l'epoque " << epoch + 1 << " !" << std::endl;
            break;
        }
    }
    std::cout << "+-" << std::endl;

    print_predictions_header("SERIES TEMPORELLES - PREDICTION SEQUENTIELLE");
    View1D prediction_result("prediction_result_timeseries", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; 
    real total_abs_error = 0.0;
    real max_error = 0.0;
    
    // Test sur quelques échantillons pour évaluation
    int num_test_samples = std::min(15, num_samples);
    
    std::cout << "+- Echantillons de prediction:" << std::endl;
    std::cout << "|" << std::endl;
    std::cout << "|  Sequence (10 derniers points)     | Cible  | Pred.  | Erreur | Qualite" << std::endl;
    std::cout << "| ---------------------------------------------------------------------------" << std::endl;
    
    for (int i = 0; i < num_test_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        
        real cost = dnn.calculate_cost(prediction, target_subview);
        final_total_cost += cost;
        
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); 
        Kokkos::fence();
        
        real target_value = h_outputs(i, 0);
        real predicted_value = h_prediction_result(0);
        real abs_error = std::abs(target_value - predicted_value);
        total_abs_error += abs_error;
        max_error = std::max(max_error, abs_error);
        
        // Affichage de la séquence (derniers 4 points pour économiser l'espace)
        std::cout << "| [";
        for (int t = 6; t < 10; ++t) {  // Derniers 4 points de la fenêtre
            std::cout << std::fixed << std::setprecision(2) << std::setw(5) << h_inputs(i, t);
            if (t < 9) std::cout << ",";
        }
        std::cout << "...] | " << std::setprecision(3) << std::setw(6) << target_value 
                  << " | " << std::setw(6) << predicted_value 
                  << " | " << std::setw(6) << abs_error << " | ";
        
        if (abs_error < 0.05) {
            std::cout << "Excellent";
        } else if (abs_error < 0.15) {
            std::cout << "Bon";
        } else if (abs_error < 0.3) {
            std::cout << "Moyen";
        } else {
            std::cout << "Faible";
        }
        std::cout << std::endl;
    }
    
    // Calcul des métriques sur tous les échantillons
    real total_test_error = 0.0;
    real total_test_cost = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        
        total_test_cost += dnn.calculate_cost(prediction, target_subview);
        
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); 
        Kokkos::fence();
        
        real abs_error = std::abs(h_outputs(i, 0) - h_prediction_result(0));
        total_test_error += abs_error;
    }
    
    real avg_abs_error = total_test_error / num_samples;
    real avg_cost = total_test_cost / num_samples;
    
    std::cout << "|" << std::endl;
    std::cout << "+- RESULTATS FINAUX:" << std::endl;
    std::cout << "|  - Cout final moyen     : " << std::scientific << std::setprecision(4) << avg_cost << std::endl;
    std::cout << "|  - Erreur absolue moy.  : " << std::fixed << std::setprecision(4) << avg_abs_error << std::endl;
    std::cout << "|  - Erreur absolue max   : " << std::setprecision(4) << max_error << std::endl;
    std::cout << "|  - Echantillons testes  : " << num_samples << std::endl;
    std::cout << "|" << std::endl;
    std::cout << "|  - Performance temporelle:" << std::endl;
    
    if (avg_abs_error < 0.08) {
        std::cout << "|    * Prediction        : Excellente (err < 0.08)" << std::endl;
    } else if (avg_abs_error < 0.15) {
        std::cout << "|    * Prediction        : Bonne (err < 0.15)" << std::endl;
    } else if (avg_abs_error < 0.25) {
        std::cout << "|    * Prediction        : Moyenne (err < 0.25)" << std::endl;
    } else {
        std::cout << "|    * Prediction        : A ameliorer (err >= 0.25)" << std::endl;
    }
    
    std::cout << "|    * Capacite adapt.   : Le reseau apprend sans freq. fixes!" << std::endl;
    std::cout << "|    * Robustesse        : Teste sur " << num_samples << " patterns differents" << std::endl;
    std::cout << "+-" << std::endl;
} 