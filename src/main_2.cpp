#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <random>
#include <vector> // Utiliser std::vector pour les couches
#include <map>
#include <string>
#include <memory> // Pour std::unique_ptr
#include <stdexcept> // Pour std::runtime_error
#include <functional> // Pas forcément nécessaire ici si on utilise des classes virtuelles

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
// #include <KokkosBlas.hpp> // Pas utilisé dans cette version mais potentiellement utile

// --- Typedefs ---
using real = float;
using View1D = Kokkos::View<real*>;
using View2D = Kokkos::View<real**>;
// using View3D = Kokkos::View<real***>; // Renommé car View3D était mal défini

// --- Forward Declarations ---
class Optimizer;
class Layer;
class Activation;
class Dataset;
class Network;
class BatchHandler;

// --- Classe Optimizer (placeholder) ---
class Optimizer {
public:
    // TODO: Implement optimizer logic (e.g., SGD, Adam)
    // Needs members like learning rate, possibly momentum terms, etc.
    // Needs methods like `update(weights, gradients)`
    real learning_rate = 0.01; // Example
    Optimizer() = default;
    // Add virtual destructor if inheritance is planned
    virtual ~Optimizer() = default;
};

// --- Classes Activation ---
class Activation {
public:
    virtual ~Activation() = default; // Important: Virtual destructor for base class

    // Applique la fonction d'activation
    virtual void apply(const View1D& input, View1D& output) const = 0;

    // Applique la dérivée de la fonction d'activation (souvent par rapport à l'entrée *avant* activation, z)
    virtual void apply_derivative(const View1D& input_z, View1D& output_deriv) const = 0;
};

class RELU : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent(0);
        Kokkos::parallel_for("relu_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = (input(i) > 0.0) ? input(i) : 0.0;
        });
    }

    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent(0);
        Kokkos::parallel_for("relu_deriv", size, KOKKOS_LAMBDA(const int i) {
            output_deriv(i) = (input_z(i) > 0.0) ? 1.0 : 0.0;
        });
    }
};

class SIGMOID : public Activation {
public:
    // Fonction scalaire pour la réutiliser facilement
    KOKKOS_INLINE_FUNCTION real scalar_sigmoid(real x) const {
         // Limiter l'argument de exp pour éviter overflow/underflow
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
        // Utiliser Kokkos::tanh pour la compatibilité potentielle GPU intrinsics
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

// Helper pour créer des activations par type
std::unique_ptr<Activation> create_activation(const std::string& type) {
    if (type == "relu") return std::make_unique<RELU>();
    if (type == "sigmoid") return std::make_unique<SIGMOID>();
    if (type == "tanh") return std::make_unique<TANH>();
    // Ajouter 'linear' si besoin
    throw std::runtime_error("Unknown activation type: " + type);
}


// --- Classe Layer ---
class Layer {
public:
    int input_size;
    int layer_size;
    std::unique_ptr<Activation> activation; // Utilisation de pointeur intelligent
    Optimizer optimizer; // Chaque couche pourrait avoir son propre état d'optimiseur si nécessaire

    View2D weights;
    View1D biases;
    View1D z;     // Sortie avant activation (w*x + b)
    View1D a;     // Sortie après activation (activation(z))
    View1D tmp; // Pour usage temporaire (ex: stocker dérivée de l'activation)

    // Constructeur pour couche cachée/sortie
    Layer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        input_size(_input_size),
        layer_size(_layer_size),
        activation(std::move(act_func)) // Prend possession du pointeur
    {
        if (input_size <= 0 || layer_size <= 0) {
             throw std::runtime_error("Layer input/output sizes must be positive.");
        }
        weights = View2D("weights", layer_size, input_size);
        biases = View1D("biases", layer_size);
        z = View1D("z", layer_size);
        a = View1D("a", layer_size); // a a la taille de la couche
        tmp = View1D("tmp", layer_size);

        // Initialisation (exemple simple : aléatoire) - utiliser Kokkos::Random
        Kokkos::Random_XorShift64_Pool<> rand_pool(std::time(0) + reinterpret_cast<uintptr_t>(this)); // Seed unique par couche
        Kokkos::fill_random(weights, rand_pool, static_cast<real>(-0.1), static_cast<real>(0.1));
        Kokkos::fill_random(biases, rand_pool, static_cast<real>(-0.1), static_cast<real>(0.1));
        // Le reste (z, a, tmp) n'a pas besoin d'être initialisé ici, sera calculé.
        Kokkos::deep_copy(z, 0.0);
        Kokkos::deep_copy(a, 0.0);
        Kokkos::deep_copy(tmp, 0.0);
        std::cout << "  Layer created: " << input_size << " -> " << layer_size << std::endl;
    }

    // Constructeur spécifique pour InputLayer (pas de poids/biais, a = input)
    Layer(int _layer_size) :
        input_size(0), // Indique que c'est une couche d'entrée
        layer_size(_layer_size),
        activation(nullptr) // Pas d'activation pour la couche d'entrée typiquement
        // Ne pas allouer weights/biases/z/tmp car non nécessaires
    {
         if (layer_size <= 0) {
             throw std::runtime_error("Input layer size must be positive.");
        }
        // 'a' représente directement l'entrée fournie au réseau
        a = View1D("input_a", layer_size);
        Kokkos::deep_copy(a, 0.0);
         std::cout << "  Input Layer created: size " << layer_size << std::endl;
    }

     // Méthodes potentielles (à implémenter)
    // virtual void forward(const View1D& prev_layer_a) { /*...*/ }
    // virtual void backward(/*...*/) { /*...*/ }
    // virtual void update_weights(/*...*/) { /*...*/ }

    // S'assurer que Layer est déplaçable (pour std::vector)
    Layer(Layer&& other) = default;
    Layer& operator=(Layer&& other) = default;

    // Supprimer la copie pour éviter les problèmes avec les unique_ptr et Kokkos Views
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    virtual ~Layer() = default; // Important si on prévoit d'hériter (InputLayer/OutputLayer)
};

// Pas besoin de classes InputLayer/OutputLayer séparées si Layer gère le cas input_size=0
// et si la logique spécifique (ex: fonction de coût) est gérée par Network.
// On garde la structure pour l'instant si souhaité.

class InputLayer : public Layer {
public:
    InputLayer(int _layer_size) : Layer(_layer_size) {} // Utilise le constructeur spécial de Layer

    void set_input(const View1D& input_data) {
         if (input_data.extent(0) != layer_size) {
             throw std::runtime_error("Input data size mismatch for InputLayer.");
         }
         Kokkos::deep_copy(a, input_data);
     }
};

// OutputLayer pourrait avoir une logique spécifique pour le calcul de coût/erreur
class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        Layer(_input_size, _layer_size, std::move(act_func)) {}

    // Ajouter des méthodes spécifiques si nécessaire, ex: calcul de l'erreur
    // real calculate_error(const View1D& target) { /* ... */ }
};


// --- Classe Dataset (placeholder) ---
class Dataset {
    // TODO: Load/manage training/testing data
};

// --- Classe BatchHandler (placeholder) ---
class BatchHandler {
    // TODO: Create mini-batches from Dataset
};


// --- Classe Network ---
class Network {
public:
    std::map<int, int> layer_sizes; // Stocke la configuration [0: input_size, 1: hidden1_size, ..., N: output_size]
    InputLayer input_layer;
    std::vector<Layer> hidden_layers; // Utilise std::vector<Layer>
    OutputLayer output_layer;
    Dataset dataset; // Placeholder
    BatchHandler batch_handler; // Placeholder

    // Constructeur principal
    Network(const std::map<int, int>& _layer_sizes, const std::vector<std::string>& activation_types) :
        layer_sizes(_layer_sizes),
        input_layer(get_size(0)) // Construit InputLayer
        // Initialisation de output_layer nécessite le dernier type d'activation
        , output_layer(get_size(num_layers() - 2), get_size(num_layers() - 1), create_activation(activation_types.back()))
    {
        std::cout << "Creating Network..." << std::endl;
        if (layer_sizes.size() < 2) {
            throw std::runtime_error("Network must have at least an input and output layer.");
        }
        if (activation_types.size() != layer_sizes.size() -1) {
             throw std::runtime_error("Number of activation types must match number of hidden + output layers.");
        }

        int num_hidden_layers = layer_sizes.size() - 2;
        hidden_layers.reserve(num_hidden_layers); // Pré-allouer la mémoire pour le vecteur

        std::cout << "  Number of hidden layers: " << num_hidden_layers << std::endl;

        // Construire les couches cachées
        for (int i = 0; i < num_hidden_layers; ++i) {
            int current_layer_idx = i + 1; // Indice dans la map layer_sizes (0=input, 1=hidden1, ...)
            int in_size = get_size(current_layer_idx - 1);
            int out_size = get_size(current_layer_idx);
            std::string act_type = activation_types[i]; // activation pour cette couche cachée

            // Utiliser emplace_back pour construire l'objet Layer directement dans le vecteur
            hidden_layers.emplace_back(in_size, out_size, create_activation(act_type));
            std::cout << "  Hidden Layer " << i+1 << " added." << std::endl;
        }
         std::cout << "Network creation complete." << std::endl;
    }

    // Helper pour obtenir la taille d'une couche depuis la map
    int get_size(int layer_index) const {
        try {
            return layer_sizes.at(layer_index);
        } catch (const std::out_of_range& oor) {
            throw std::runtime_error("Invalid layer index requested: " + std::to_string(layer_index));
        }
    }

    // Helper pour obtenir le nombre total de couches (input + hidden + output)
     int num_layers() const {
        return layer_sizes.size();
    }


    // TODO: Implémenter forward, backward, train, etc.
    // View1D predict(const View1D& input) { /* ... */ }
    // void train(/* ... */) { /* ... */ }

     // Déplacement autorisé (pour pouvoir retourner un Network d'une fonction par ex.)
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;

    // Copie supprimée car contient des unique_ptr et des Kokkos Views
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

     ~Network() {
         std::cout << "Destroying Network..." << std::endl;
         // Les unique_ptr dans Layer seront automatiquement détruits
         // Les Kokkos Views dans Layer seront automatiquement décrémentés/libérés
         // Le std::vector<Layer> sera automatiquement détruit
     }

};

// --- Fonction Main ---
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    { // Scope pour Kokkos
        std::cout << "--- Neural Network Test ---" << std::endl;

        try {
            // Définir la structure du réseau
            std::map<int, int> sizes;
            sizes[0] = 2; // Couche d'entrée : 2 neurones
            sizes[1] = 4; // 1ère couche cachée : 4 neurones
            sizes[2] = 3; // 2ème couche cachée : 3 neurones
            sizes[3] = 1; // Couche de sortie : 1 neurone

            // Définir les activations pour les couches cachées et la sortie
            std::vector<std::string> activations = {"tanh", "tanh", "sigmoid"}; // Hidden1, Hidden2, Output

             if (sizes.size() - 1 != activations.size()) {
                 std::cerr << "Error: Mismatch between number of layers and activations provided." << std::endl;
                 Kokkos::finalize();
                 return 1;
             }

            // Créer le réseau
            Network network(sizes, activations);

            std::cout << "Network object created successfully." << std::endl;

            // --- Ici, on pourrait ajouter des tests ---
            // Ex: Créer une donnée d'entrée et faire une prédiction (quand forward sera implémenté)
             View1D sample_input("sample_input", sizes[0]);
             Kokkos::deep_copy(sample_input, 1.0); // Mettre une valeur exemple
            // View1D prediction = network.predict(sample_input); // Appel futur
            // Kokkos::fence(); // S'assurer que le calcul est terminé avant d'accéder

            // network.train(...) // Appel futur

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }

        std::cout << "--- End of Test ---" << std::endl;
    } // Fin du scope Kokkos
    Kokkos::finalize();
    return 0;
}
