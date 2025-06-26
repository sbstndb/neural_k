#include "training.hpp"
#include "optimizers.hpp"

void xor_train(const std::string& optimizer_choice) {
    std::cout << "\n--- XOR Training Example (Sparse Weights Storage) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

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

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << std::endl;

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
            std::cout << "Epoch: " << std::setw(5) << epoch + 1
                      << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost << std::endl;
        }
        if (avg_cost < 1e-4) {
             std::cout << "Convergence likely reached at epoch " << epoch + 1 << std::endl; break;
        }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    std::cout << "\n-- FINAL PREDICTIONS --" << std::endl;
    View1D prediction_result("prediction_result", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        std::cout << "Input: [" << h_xor_inputs(i,0) << "," << h_xor_inputs(i,1) << "] "
                  << "Target: " << h_xor_outputs(i,0) << " "
                  << "Prediction: " << std::fixed << std::setprecision(4) << h_prediction_result(0)
                  << " (Rounded: " << std::round(h_prediction_result(0)) << ")" << std::endl;
    }
     std::cout << "Final Average Cost: " << std::fixed << std::setprecision(8) << final_total_cost / num_samples << std::endl;
}

void sine_train(const std::string& optimizer_choice) {
    std::cout << "\n--- Sine Function Approx Training (Sparse Weights Storage) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

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

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << ", Samples: " << num_samples << std::endl;

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
             std::cout << "Epoch: " << std::setw(5) << epoch + 1 << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost << std::endl;
        }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    std::cout << "\n-- FINAL PREDICTIONS (Sample) --" << std::endl;
    View1D prediction_result("prediction_result_sine", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; int num_test_samples = std::min(num_samples, 10);
    for (int i = 0; i < num_test_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        real input_x = h_inputs(i, 0); real target_y = h_outputs(i, 0);
        std::cout << "Input x: " << std::fixed << std::setprecision(4) << input_x << " "
                  << "Target sin(x): " << std::fixed << std::setprecision(4) << target_y << " "
                  << "Prediction: " << std::fixed << std::setprecision(4) << h_prediction_result(0) << std::endl;
    }
    std::cout << "Final Average Cost (on first " << num_test_samples << " samples): " << std::fixed << std::setprecision(8) << final_total_cost / num_test_samples << std::endl;
}

void linear_sep_train(const std::string& optimizer_choice) {
    std::cout << "\n--- Linear Separation Training (Sparse Weights Storage) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;

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
    std::cout << "Generated " << count_class0 << " Class 0 and " << count_class1 << " Class 1." << std::endl;
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << ", Samples: " << num_samples << std::endl;

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
             std::cout << "Epoch: " << std::setw(4) << epoch + 1 << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost << std::endl;
        }
         if (avg_cost < 1e-3) {
             std::cout << "Convergence likely reached at epoch " << epoch + 1 << std::endl; break;
         }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    std::cout << "\n-- FINAL PREDICTIONS & ACCURACY --" << std::endl;
    View1D prediction_result("prediction_result_linear", output_dim);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; int correct_predictions = 0;
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(train_inputs, i, Kokkos::ALL());
        auto target_subview = Kokkos::subview(train_outputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        final_total_cost += dnn.calculate_cost(prediction, target_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
        real predicted_value = h_prediction_result(0); int predicted_class = std::round(predicted_value);
        int target_class = static_cast<int>(h_outputs(i, 0));
        if (predicted_class == target_class) correct_predictions++;
         if (i < 10) {
              std::cout << "Input: [" << std::fixed << std::setprecision(2) << h_inputs(i,0) << "," << h_inputs(i,1) << "] "
                        << "Target: " << target_class << " Pred: " << std::fixed << std::setprecision(3) << predicted_value
                        << " (Rounded: " << predicted_class << ")" << (predicted_class == target_class ? "" : " <-- WRONG") << std::endl;
         }
    }
    real accuracy = static_cast<real>(correct_predictions) / num_samples;
    std::cout << "Final Average Cost: " << std::fixed << std::setprecision(8) << final_total_cost / num_samples << std::endl;
    std::cout << "Final Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%" << std::endl;
} 