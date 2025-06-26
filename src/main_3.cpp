#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <random>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <numeric> // Pour std::iota (potentiellement)
#include <iomanip> // Pour std::setprecision

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp> // *** AJOUTÉ ***
#include <Kokkos_StdAlgorithms.hpp> // Pour parallel_reduce

// --- Typedefs ---
using real = float;
using View1D = Kokkos::View<real*>;
using View2D = Kokkos::View<real**>;
// using View3D = Kokkos::View<real***>; // Pas utilisé ici

// --- Forward Declarations ---
class Optimizer; // Gardé pour future extension, mais non utilisé activement ici
class Layer;
class Activation;
class Dataset;
class Network;
class BatchHandler;

// --- Classe Optimizer (placeholder) ---
class Optimizer {
public:
    real learning_rate = 0.1; // Default learning rate
    Optimizer() = default;
    virtual ~Optimizer() = default;
    // Future methods: virtual void update(Layer& layer, int batch_size) = 0;
};

// --- Classes Activation (inchangées par rapport à la version précédente) ---
class Activation {
public:
    virtual ~Activation() = default;
    virtual void apply(const View1D& input, View1D& output) const = 0;
    virtual void apply_derivative(const View1D& input_z, View1D& output_deriv) const = 0;
};

class RELU : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent(0);
        Kokkos::parallel_for("relu_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = (input(i) > 0.0f) ? input(i) : 0.0f;
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent(0);
        Kokkos::parallel_for("relu_deriv", size, KOKKOS_LAMBDA(const int i) {
            output_deriv(i) = (input_z(i) > 0.0f) ? 1.0f : 0.0f;
        });
    }
};

class SIGMOID : public Activation {
public:
    KOKKOS_INLINE_FUNCTION real scalar_sigmoid(real x) const {
        x = Kokkos::max(-30.0f, Kokkos::min(30.0f, x));
        return 1.0f / (1.0f + Kokkos::exp(-x));
    }
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent(0);
        Kokkos::parallel_for("sigmoid_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = scalar_sigmoid(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent(0);
        Kokkos::parallel_for("sigmoid_deriv", size, KOKKOS_LAMBDA(const int i) {
            real s = scalar_sigmoid(input_z(i));
            output_deriv(i) = s * (1.0f - s);
        });
    }
};

class TANH : public Activation {
public:
     KOKKOS_INLINE_FUNCTION real scalar_tanh(real x) const {
        return Kokkos::tanh(x);
    }
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent(0);
        Kokkos::parallel_for("tanh_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = scalar_tanh(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent(0);
        Kokkos::parallel_for("tanh_deriv", size, KOKKOS_LAMBDA(const int i) {
            real t = scalar_tanh(input_z(i));
            output_deriv(i) = 1.0f - t * t;
        });
    }
};

// Helper pour créer des activations par type (inchangé)
std::unique_ptr<Activation> create_activation(const std::string& type) {
    if (type == "relu") return std::make_unique<RELU>();
    if (type == "sigmoid") return std::make_unique<SIGMOID>();
    if (type == "tanh") return std::make_unique<TANH>();
    throw std::runtime_error("Unknown activation type: " + type);
}


// --- Classe Layer (Modifiée) ---
class Layer {
public:
    int input_size;
    int layer_size;
    std::unique_ptr<Activation> activation;
    Optimizer optimizer; // Gardé pour le futur

    // Vues Kokkos pour les paramètres et états
    View2D weights;
    View1D biases;
    View1D z;     // Sortie avant activation
    View1D a;     // Sortie après activation (activation(z))

    // Vues Kokkos pour la rétropropagation (gradients)
    View1D delta;            // Erreur rétropropagée (δ) pour cette couche
    View1D d_biases;         // Gradient instantané des biais (∂Cost/∂b = δ)
    View2D d_weights;        // Gradient instantané des poids (∂Cost/∂W = δ * a_prev^T)
    View1D d_biases_sum;     // Somme des gradients des biais sur le batch
    View2D d_weights_sum;    // Somme des gradients des poids sur le batch

    View1D tmp_deriv;        // Stockage temporaire pour la dérivée de l'activation f'(z)


    // Constructeur pour couche cachée/sortie
    Layer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        input_size(_input_size),
        layer_size(_layer_size),
        activation(std::move(act_func))
    {
        if (input_size <= 0 || layer_size <= 0) {
             throw std::runtime_error("Layer input/output sizes must be positive.");
        }
        // Allocation des vues de paramètres et d'état
        weights = View2D("weights", layer_size, input_size);
        biases = View1D("biases", layer_size);
        z = View1D("z", layer_size);
        a = View1D("a", layer_size);

        // Allocation des vues pour les gradients
        delta = View1D("delta", layer_size);
        d_biases = View1D("d_biases", layer_size);
        d_weights = View2D("d_weights", layer_size, input_size);
        d_biases_sum = View1D("d_biases_sum", layer_size);
        d_weights_sum = View2D("d_weights_sum", layer_size, input_size);
        tmp_deriv = View1D("tmp_deriv", layer_size); // Pour stocker f'(z)


        // Initialisation aléatoire des poids et biais
        Kokkos::Random_XorShift64_Pool<> rand_pool(std::time(0) + reinterpret_cast<uintptr_t>(this));
        Kokkos::fill_random(weights, rand_pool, static_cast<real>(-0.2), static_cast<real>(0.2)); // Ajusté l'intervalle
        Kokkos::fill_random(biases, rand_pool, static_cast<real>(-0.1), static_cast<real>(0.1));

        // Initialisation des autres vues à zéro
        zero_accumulated_gradients(); // Initialise les _sum à 0
        Kokkos::deep_copy(z, 0.0);
        Kokkos::deep_copy(a, 0.0);
        Kokkos::deep_copy(delta, 0.0);
        Kokkos::deep_copy(d_biases, 0.0);
        Kokkos::deep_copy(d_weights, 0.0);
        Kokkos::deep_copy(tmp_deriv, 0.0);

        //std::cout << "  Layer created: " << input_size << " -> " << layer_size << std::endl;
    }

    // Constructeur spécifique pour InputLayer (pas de poids/biais/gradient)
    Layer(int _layer_size) :
        input_size(0), // Indique InputLayer
        layer_size(_layer_size),
        activation(nullptr)
        // Ne pas allouer weights, biases, z, delta, gradients...
    {
         if (layer_size <= 0) {
             throw std::runtime_error("Input layer size must be positive.");
        }
        a = View1D("input_a", layer_size); // 'a' est l'entrée
        Kokkos::deep_copy(a, 0.0);
         //std::cout << "  Input Layer created: size " << layer_size << std::endl;
    }

    // --- Méthodes ajoutées ---

    // Calcule la sortie de la couche
    virtual void forward(const View1D& prev_layer_a) {
        if (!activation || input_size == 0) return; // Ne rien faire pour InputLayer

        // 1. Calculer z = W * prev_layer_a + b
        // z = W * prev_layer_a (beta=0.0 pour écraser l'ancien z)
        KokkosBlas::gemv("N", 1.0, weights, prev_layer_a, 0.0, z);

        // z = z + b (ajout des biais)
        Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) {
            z(i) += biases(i);
        });

        // 2. Calculer a = activation(z)
        activation->apply(z, a);
    }

     // Calcule les gradients pour les couches cachées
    virtual void compute_gradients(const Layer& next_layer, const View1D& prev_layer_a) {
         if (!activation || input_size == 0) return; // Ne rien faire pour InputLayer

        // 1. Calculer la dérivée de l'activation f'(z) et la stocker
        activation->apply_derivative(z, tmp_deriv);

        // 2. Calculer delta de cette couche : δ_l = (W_{l+1}^T * δ_{l+1}) .* f'(z_l)
        //    delta = W_{l+1}^T * δ_{l+1} (beta=0.0 pour écraser l'ancien delta)
        KokkosBlas::gemv("T", 1.0, next_layer.weights, next_layer.delta, 0.0, delta);

        //    delta = delta .* f'(z_l) (multiplication élément par élément)
         Kokkos::parallel_for("hadamard_delta_deriv", layer_size, KOKKOS_LAMBDA(int i) {
            delta(i) *= tmp_deriv(i);
        });

        // 3. Calculer les gradients instantanés pour ce sample
        //    d_biases = delta
        Kokkos::deep_copy(d_biases, delta);

        //    d_weights = delta * prev_layer_a^T (produit externe)
        Kokkos::parallel_for("compute_d_weights", layer_size, KOKKOS_LAMBDA(int i) {
            for (int j = 0; j < input_size; ++j) {
                d_weights(i, j) = delta(i) * prev_layer_a(j);
            }
        });

        // 4. Accumuler les gradients (pour Batch Gradient Descent)
        Kokkos::parallel_for("accumulate_gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                d_weights_sum(i, j) += d_weights(i, j);
        });
         Kokkos::parallel_for("accumulate_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             d_biases_sum(i) += d_biases(i);
        });
    }


    // Met à jour les poids et biais (appelé après un batch)
    virtual void update_weights(real learning_rate, int batch_size) {
        if (input_size == 0) return; // Pas de poids/biais pour InputLayer

        real scale = learning_rate / static_cast<real>(batch_size);

        // Mettre à jour les poids
        Kokkos::parallel_for("update_weights", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                weights(i, j) -= scale * d_weights_sum(i, j);
        });

        // Mettre à jour les biais
        Kokkos::parallel_for("update_biases", layer_size, KOKKOS_LAMBDA(int i) {
            biases(i) -= scale * d_biases_sum(i);
        });
    }

    // Remet à zéro les gradients accumulés (appelé avant chaque batch)
    void zero_accumulated_gradients() {
         if (input_size == 0) return;
         Kokkos::deep_copy(d_biases_sum, 0.0);
         Kokkos::deep_copy(d_weights_sum, 0.0);
    }

    // Afficher poids et biais (pour débogage)
    void show() const {
        if (input_size == 0) {
            std::cout << "Input Layer (size " << layer_size << ")" << std::endl;
            return;
        }
        std::cout << "Layer (" << input_size << " -> " << layer_size << "):" << std::endl;
        auto h_weights = Kokkos::create_mirror_view(weights);
        auto h_biases = Kokkos::create_mirror_view(biases);
        Kokkos::deep_copy(h_weights, weights);
        Kokkos::deep_copy(h_biases, biases);
        Kokkos::fence(); // Assurer que la copie est terminée

        std::cout << "  Weights (sample):" << std::endl;
        for(int i=0; i< std::min(layer_size, 5) ; ++i) { // Affiche les 5 premières lignes max
             std::cout << "    [";
             for(int j=0; j< std::min(input_size, 5); ++j) { // Affiche les 5 premières colonnes max
                 std::cout << std::fixed << std::setprecision(3) << h_weights(i,j) << " ";
             }
             if (input_size > 5) std::cout << "...";
             std::cout << "]" << std::endl;
        }
         if (layer_size > 5) std::cout << "    ..." << std::endl;

        std::cout << "  Biases (sample): [";
        for(int i=0; i< std::min(layer_size, 10); ++i) { // Affiche les 10 premiers biais max
            std::cout << std::fixed << std::setprecision(3) << h_biases(i) << " ";
        }
        if (layer_size > 10) std::cout << "...";
        std::cout << "]" << std::endl;
    }

    // S'assurer que Layer est déplaçable (pour std::vector)
    Layer(Layer&& other) = default;
    Layer& operator=(Layer&& other) = default;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    virtual ~Layer() = default;
};

// --- Classe InputLayer (simplifiée) ---
class InputLayer : public Layer {
public:
    InputLayer(int _layer_size) : Layer(_layer_size) {}

    // Méthode pour définir l'entrée du réseau
    void set_input(const View1D& input_data) {
         if (input_data.extent(0) != layer_size) {
             throw std::runtime_error("Input data size mismatch for InputLayer.");
         }
         Kokkos::deep_copy(a, input_data); // Copie les données d'entrée dans le 'a' de InputLayer
     }
};

// --- Classe OutputLayer (avec calcul de gradient spécifique) ---
class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        Layer(_input_size, _layer_size, std::move(act_func)) {}

    // Calcule les gradients pour la couche de sortie (diffère par le calcul initial de delta)
    // Prend la cible en argument
    void compute_gradients(const View1D& target, const View1D& prev_layer_a) {
         if (!activation) return;

        // 1. Calculer la dérivée de l'activation f'(z)
        activation->apply_derivative(z, tmp_deriv);

        // 2. Calculer delta pour la couche de sortie : δ_L = (a_L - y) .* f'(z_L)
        //    (Suppose une fonction de coût MSE: dCost/da = a - y)
        Kokkos::parallel_for("compute_output_delta", layer_size, KOKKOS_LAMBDA(int i) {
            delta(i) = (a(i) - target(i)) * tmp_deriv(i);
        });

       // 3. Calculer les gradients instantanés (identique à Layer)
        Kokkos::deep_copy(d_biases, delta);
        Kokkos::parallel_for("compute_output_d_weights", layer_size, KOKKOS_LAMBDA(int i) {
            for (int j = 0; j < input_size; ++j) {
                d_weights(i, j) = delta(i) * prev_layer_a(j);
            }
        });

        // 4. Accumuler les gradients (identique à Layer)
        Kokkos::parallel_for("accumulate_output_gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                d_weights_sum(i, j) += d_weights(i, j);
        });
         Kokkos::parallel_for("accumulate_output_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             d_biases_sum(i) += d_biases(i);
        });
    }
};


// --- Classes Dataset / BatchHandler (Placeholders inchangés) ---
class Dataset {};
class BatchHandler {};


// --- Classe Network (Modifiée) ---
class Network {
public:
    std::map<int, int> layer_sizes;
    InputLayer input_layer;
    std::vector<Layer> hidden_layers;
    OutputLayer output_layer;
    Dataset dataset;
    BatchHandler batch_handler;

    // Constructeur (inchangé par rapport à la version précédente)
    Network(const std::map<int, int>& _layer_sizes, const std::vector<std::string>& activation_types) :
        layer_sizes(_layer_sizes),
        input_layer(get_size(0)),
        output_layer(get_size(num_layers() - 2), get_size(num_layers() - 1), create_activation(activation_types.back()))
    {
        //std::cout << "Creating Network..." << std::endl;
        if (layer_sizes.size() < 2) {
            throw std::runtime_error("Network must have at least an input and output layer.");
        }
        if (activation_types.size() != layer_sizes.size() -1) {
             throw std::runtime_error("Number of activation types must match number of hidden + output layers.");
        }
        int num_hidden_layers = layer_sizes.size() - 2;
        hidden_layers.reserve(num_hidden_layers);
        //std::cout << "  Number of hidden layers: " << num_hidden_layers << std::endl;
        for (int i = 0; i < num_hidden_layers; ++i) {
            int current_layer_idx = i + 1;
            int in_size = get_size(current_layer_idx - 1);
            int out_size = get_size(current_layer_idx);
            std::string act_type = activation_types[i];
            hidden_layers.emplace_back(in_size, out_size, create_activation(act_type));
            //std::cout << "  Hidden Layer " << i+1 << " added." << std::endl;
        }
        //std::cout << "Network creation complete." << std::endl;
    }

    // Helper (inchangés)
    int get_size(int layer_index) const { /* ... */ return layer_sizes.at(layer_index); }
    int num_layers() const { /* ... */ return layer_sizes.size(); }


    // --- Méthodes ajoutées / modifiées ---

    // Effectue la passe avant (prédiction)
    View1D forward(const View1D& input_data) {
        // 1. Définir l'entrée dans la couche d'input
        input_layer.set_input(input_data);

        // 2. Propager à travers les couches cachées
        View1D current_a = input_layer.a; // Commence avec l'activation de la couche d'input
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.forward(current_a);
            current_a = hidden_layer.a; // Met à jour pour la couche suivante
        }

        // 3. Propager à travers la couche de sortie
        output_layer.forward(current_a);

        // 4. Retourner l'activation de la couche de sortie
        return output_layer.a;
    }

    // Effectue la passe arrière (calcul des gradients) pour un sample
    void backward(const View1D& target) {
        // 1. Calculer les gradients pour la couche de sortie
        const View1D& prev_a_output = hidden_layers.empty() ? input_layer.a : hidden_layers.back().a;
        output_layer.compute_gradients(target, prev_a_output);

        // 2. Rétropropager à travers les couches cachées (de la dernière à la première)
        Layer* next_layer_ptr = &output_layer; // Commence avec la couche de sortie comme "next"
        for (int i = hidden_layers.size() - 1; i >= 0; --i) {
            Layer& current_layer = hidden_layers[i];
            // Activation de la couche *précédente* à current_layer
            const View1D& prev_a = (i == 0) ? input_layer.a : hidden_layers[i - 1].a;
            current_layer.compute_gradients(*next_layer_ptr, prev_a);
            next_layer_ptr = &current_layer; // Met à jour pour l'itération suivante
        }
        // Note: InputLayer n'a pas de gradients à calculer.
    }

     // Met à jour les poids de toutes les couches après un batch
    void update(real learning_rate, int batch_size) {
        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for update.");
        }
        // Mettre à jour les couches cachées
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.update_weights(learning_rate, batch_size);
        }
        // Mettre à jour la couche de sortie
        output_layer.update_weights(learning_rate, batch_size);
    }

    // Remet à zéro les gradients accumulés de toutes les couches
    void zero_accumulated_gradients() {
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.zero_accumulated_gradients();
        }
        output_layer.zero_accumulated_gradients();
    }

    // Calcule le coût (erreur quadratique moyenne pour l'instant) pour un sample
    real calculate_cost(const View1D& prediction, const View1D& target) {
        int output_size = prediction.extent(0);
        if (target.extent(0) != output_size) {
            throw std::runtime_error("Prediction and target size mismatch for cost calculation.");
        }
        real squared_error_sum = 0.0;

        // Utiliser parallel_reduce pour sommer les erreurs au carré
        Kokkos::parallel_reduce("compute_cost", output_size, KOKKOS_LAMBDA (int i, real& lsum) {
            real diff = prediction(i) - target(i);
            lsum += diff * diff;
        }, squared_error_sum);

        Kokkos::fence(); // S'assurer que la réduction est terminée
        return 0.5 * squared_error_sum;
    }

    // Affiche les informations des couches
    void show() const {
         std::cout << "--- Network Structure ---" << std::endl;
         input_layer.show();
         int i = 1;
         for (const auto& layer : hidden_layers) {
              std::cout << "\n--- Hidden Layer " << i++ << " ---" << std::endl;
              layer.show();
         }
         std::cout << "\n--- Output Layer ---" << std::endl;
         output_layer.show();
         std::cout << "-------------------------" << std::endl;
    }


    // Déplacement/Copie (inchangés)
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    ~Network() {
        //std::cout << "Destroying Network..." << std::endl;
    }
};

// --- Fonction d'entraînement XOR (adaptée) ---
void xor_train() {
    std::cout << "\n--- XOR Training Example ---" << std::endl;

    // Structure du réseau
    std::map<int, int> sizes;
    sizes[0] = 2; // Input
    sizes[1] = 3; // Hidden 1
    // sizes[2] = 2; // Hidden 2 (comme dans le code 1)
    sizes[2] = 1; // Output (directement)

    //std::vector<std::string> activations = {"tanh", "tanh", "tanh"}; // H1, H2, Out (comme code 1)
    std::vector<std::string> activations = {"tanh", "tanh"}; // H1, Out (structure simplifiée)


    Network dnn(sizes, activations);

    // Données XOR
    // Utilisation de Kokkos Views pour les données pour éviter les copies Host <-> Device constantes
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    using HostView1D = Kokkos::View<real*, Kokkos::HostSpace>;

    const int num_samples = 4; // XOR standard
    const int input_dim = sizes[0];
    const int output_dim = sizes[sizes.size()-1];

    HostView2D h_xor_inputs("h_xor_inputs", num_samples, input_dim);
    HostView2D h_xor_outputs("h_xor_outputs", num_samples, output_dim);

    h_xor_inputs(0, 0) = 0.0; h_xor_inputs(0, 1) = 0.0; h_xor_outputs(0, 0) = 0.0; // tanh -> 0
    h_xor_inputs(1, 0) = 1.0; h_xor_inputs(1, 1) = 0.0; h_xor_outputs(1, 0) = 1.0; // tanh -> 1
    h_xor_inputs(2, 0) = 0.0; h_xor_inputs(2, 1) = 1.0; h_xor_outputs(2, 0) = 1.0; // tanh -> 1
    h_xor_inputs(3, 0) = 1.0; h_xor_inputs(3, 1) = 1.0; h_xor_outputs(3, 0) = 0.0; // tanh -> 0

    // Si activation tanh, viser -1 et 1 pourrait être mieux, mais gardons 0 et 1 pour l'instant
    // h_xor_outputs(0, 0) = -1.0;
    // h_xor_outputs(1, 0) = 1.0;
    // h_xor_outputs(2, 0) = 1.0;
    // h_xor_outputs(3, 0) = -1.0;

    // Copier les données sur le device par défaut
    auto xor_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_inputs);
    auto xor_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_outputs);

    // Paramètres d'entraînement
    int epochs = 100000;
    real learning_rate = 0.5;
    int batch_size = num_samples; // Batch Gradient Descent

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << "Epochs: " << epochs << ", Learning Rate: " << learning_rate << ", Batch Size: " << batch_size << std::endl;

    View1D current_input("current_input", input_dim);
    View1D current_target("current_target", output_dim);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        dnn.zero_accumulated_gradients(); // Important: remettre à zéro avant le batch

        // Itérer sur les samples du batch
        for (int i = 0; i < num_samples; ++i) {
            // Extraire le i-ème sample (subview)
            auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
            auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());

            // Copier dans les vues temporaires si nécessaire (ou utiliser subview directement si compatible)
            // Kokkos::deep_copy(current_input, input_subview);
            // Kokkos::deep_copy(current_target, target_subview);

            // 1. Forward pass
            View1D prediction = dnn.forward(input_subview); // Utilise directement le subview

            // 2. Calculer le coût pour ce sample (optionnel pendant l'entraînement batch)
            real sample_cost = dnn.calculate_cost(prediction, target_subview);
            total_epoch_cost += sample_cost;

            // 3. Backward pass (accumule les gradients)
            dnn.backward(target_subview);
        }

        // 4. Mettre à jour les poids après avoir vu tous les samples du batch
        dnn.update(learning_rate, batch_size);

        // Afficher le coût moyen de l'époque
        if ((epoch + 1) % 500 == 0 || epoch == 0) { // Affiche toutes les 500 époques
            std::cout << "Epoch: " << std::setw(5) << epoch + 1
                      << ", Average Cost: " << std::fixed << std::setprecision(6)
                      << total_epoch_cost / num_samples << std::endl;
        }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    // Afficher les poids finaux
    dnn.show();

    // Tester les prédictions
    std::cout << "\n-- PREDICTIONS --" << std::endl;
    View1D prediction_result = Kokkos::create_mirror_view(dnn.output_layer.a); // Vue sur l'hôte pour afficher
    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
        dnn.forward(input_subview); // Recalcule la sortie avec les poids finaux
        Kokkos::deep_copy(prediction_result, dnn.output_layer.a); // Copie le résultat sur l'hôte
        Kokkos::fence(); // Assurer la fin de la copie

        std::cout << "Input:  [" << h_xor_inputs(i, 0) << ", " << h_xor_inputs(i, 1) << "]"
                  << ", Target: [" << h_xor_outputs(i, 0) << "]"
                  << ", Predicted: [" << std::fixed << std::setprecision(4) << prediction_result(0) << "]" << std::endl;
    }
     std::cout << "-----------------" << std::endl;
}


// --- Fonction Main (inchangée) ---
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    { // Scope pour Kokkos
        std::cout << "--- Neural Network Library v2 ---" << std::endl;

        try {
            xor_train(); // Lance l'entraînement XOR
            // p_train(); // Pourrait être adapté de la même manière
        } catch (const std::exception& e) {
            std::cerr << "Runtime Error: " << e.what() << std::endl;
            Kokkos::finalize();
            return 1;
        }

        std::cout << "\n--- Test Complete ---" << std::endl;
    } // Fin du scope Kokkos
    Kokkos::finalize();
    return 0;
}
