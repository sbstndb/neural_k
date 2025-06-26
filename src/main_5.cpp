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
#include <KokkosBlas.hpp>
#include <Kokkos_StdAlgorithms.hpp> // Pour parallel_reduce

// --- Typedefs ---
using real = float;
using View1D = Kokkos::View<real*>;
using View2D = Kokkos::View<real**>;
// using View3D = Kokkos::View<real***>; // Pas utilisé ici

// --- Forward Declarations ---
class Optimizer; // Base class
class SGD;       // Concrete optimizer
class Layer;
class Activation;
class Dataset;
class Network;
class BatchHandler;

// --- Classe Optimizer ---
class Optimizer {
public:
    real learning_rate;

    Optimizer(real lr = 0.1) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    // Method to update layer parameters based on accumulated gradients
    // Takes non-const views for weights/biases (to modify them)
    // Takes const views for gradients (only reads them)
    virtual void update(View2D weights, View1D biases,
                        const View2D& accumulated_d_weights, const View1D& accumulated_d_biases,
                        int batch_size) = 0;

    // Allow changing learning rate
    void set_learning_rate(real lr) { learning_rate = lr; }
    real get_learning_rate() const { return learning_rate; }

    virtual std::string get_info() const {
        return "Optimizer (Base Class), LR: " + std::to_string(learning_rate);
    }

};

// --- Concrete Optimizer: SGD ---
class SGD : public Optimizer {
public:
    SGD(real lr = 0.1) : Optimizer(lr) {}

    void update(View2D weights, View1D biases,
                const View2D& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override {

        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for SGD update.");
        }
        if (weights.extent(0) == 0) {
            return; // Nothing to update (e.g., could happen if called on input layer data)
        }

        const real scale = learning_rate / static_cast<real>(batch_size);
        const int layer_size = biases.extent(0);
        const int input_size = weights.extent(1); // Assuming weights is layer_size x input_size

        // Update weights: weights -= learning_rate/batch_size * accumulated_d_weights
        Kokkos::parallel_for("sgd_update_weights",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                weights(i, j) -= scale * accumulated_d_weights(i, j);
        });

        // Update biases: biases -= learning_rate/batch_size * accumulated_d_biases
        Kokkos::parallel_for("sgd_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
            biases(i) -= scale * accumulated_d_biases(i);
        });
    }
};

// --- Concrete Optimizer: RMSprop ---
class RMSprop : public Optimizer {
public:
    real rho;     // Decay rate (beta)
    real epsilon; // Small value to prevent division by zero

    // State storage: Map parameter address to its squared gradient average
    std::map<const real*, View2D> s_weights; // State for weights
    std::map<const real*, View1D> s_biases;  // State for biases

    RMSprop(real lr = 0.001, real rho_ = 0.9, real eps = 1e-7)
        : Optimizer(lr), rho(rho_), epsilon(eps) {}

    void update(View2D weights, View1D biases,
                const View2D& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override {

        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for RMSprop update.");
        }
        if (weights.extent(0) == 0 || biases.extent(0) == 0) {
            return;
        }

        const real scale = 1.0f / static_cast<real>(batch_size);
        const int layer_size = biases.extent(0);
        const int input_size = weights.extent(1);

        // --- Update Weights ---
        const real* w_ptr = weights.data();
        if (s_weights.find(w_ptr) == s_weights.end()) {
            // Initialize state for these weights if not present
            s_weights[w_ptr] = View2D("rmsprop_s_w", layer_size, input_size);
            Kokkos::deep_copy(s_weights.at(w_ptr), 0.0);
        }
        View2D s_w = s_weights.at(w_ptr); // Get the state view

        Kokkos::parallel_for("rmsprop_update_weights",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                real grad = scale * accumulated_d_weights(i, j);
                // Update squared gradient average: s = rho * s + (1-rho) * grad^2
                s_w(i, j) = rho * s_w(i, j) + (1.0f - rho) * grad * grad;
                // Update weights: w = w - lr * grad / (sqrt(s) + epsilon)
                weights(i, j) -= learning_rate * grad / (Kokkos::sqrt(s_w(i, j)) + epsilon);
        });

        // --- Update Biases ---
        const real* b_ptr = biases.data();
        if (s_biases.find(b_ptr) == s_biases.end()) {
            // Initialize state for these biases if not present
            s_biases[b_ptr] = View1D("rmsprop_s_b", layer_size);
            Kokkos::deep_copy(s_biases.at(b_ptr), 0.0);
        }
        View1D s_b = s_biases.at(b_ptr); // Get the state view

        Kokkos::parallel_for("rmsprop_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
            real grad = scale * accumulated_d_biases(i);
            // Update squared gradient average: s = rho * s + (1-rho) * grad^2
            s_b(i) = rho * s_b(i) + (1.0f - rho) * grad * grad;
            // Update biases: b = b - lr * grad / (sqrt(s) + epsilon)
            biases(i) -= learning_rate * grad / (Kokkos::sqrt(s_b(i)) + epsilon);
        });
    }

    std::string get_info() const override {
        return "RMSprop, LR: " + std::to_string(learning_rate) +
               ", Rho: " + std::to_string(rho) +
               ", Epsilon: " + std::to_string(epsilon);
    }
};

// --- Add other optimizers here later (e.g., Adam, RMSprop) ---
// class Adam : public Optimizer { ... };



// --- Concrete Optimizer: Adam ---
class Adam : public Optimizer {
public:
    real beta1;   // Decay rate for first moment
    real beta2;   // Decay rate for second moment
    real epsilon; // Small value for numerical stability
    long long t;  // Timestep counter (use long long to avoid overflow)

    // State storage: Map parameter address to its moments
    // Using a struct to hold both moments for a given parameter set
    struct AdamState1D {
        View1D m; // First moment
        View1D v; // Second moment (uncentered variance)
    };
     struct AdamState2D {
        View2D m; // First moment
        View2D v; // Second moment (uncentered variance)
    };

    std::map<const real*, AdamState2D> state_weights; // State for weights
    std::map<const real*, AdamState1D> state_biases;  // State for biases

    Adam(real lr = 0.001, real b1 = 0.9, real b2 = 0.999, real eps = 1e-7)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    // Override update method
    void update(View2D weights, View1D biases,
                const View2D& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override {

        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for Adam update.");
        }
         if (weights.extent(0) == 0 || biases.extent(0) == 0) {
            return;
        }

        // Increment timestep t. IMPORTANT: This assumes update() is called once per effective batch update.
        t++;

        const real scale = 1.0f / static_cast<real>(batch_size);
        const int layer_size = biases.extent(0);
        const int input_size = weights.extent(1);

        // Bias correction terms (calculate once per update)
        // Use Kokkos::pow for potentially large t
        // Need to ensure t doesn't make pow result 0 or inf prematurely.
        // Clamping t might be needed in very long runs, but usually okay.
        real beta1_t = Kokkos::pow(beta1, static_cast<real>(t));
        real beta2_t = Kokkos::pow(beta2, static_cast<real>(t));

        // Avoid division by zero if beta^t becomes 1 (unlikely with float but good practice)
        // Note: bias correction should use 1.0 - beta^t, which approaches 1.0.
        // If t is very large, beta^t -> 0, correction factor -> 1.0 (no correction needed)
        // If t is 1, correction factor is 1/(1-beta)
        real m_hat_factor = (1.0f - beta1_t > 1e-8f) ? 1.0f / (1.0f - beta1_t) : 1.0f / 1e-8f;
        real v_hat_factor = (1.0f - beta2_t > 1e-8f) ? 1.0f / (1.0f - beta2_t) : 1.0f / 1e-8f;


        // --- Update Weights ---
        const real* w_ptr = weights.data();
        if (state_weights.find(w_ptr) == state_weights.end()) {
            // Initialize state for these weights if not present
            AdamState2D new_state;
            new_state.m = View2D("adam_m_w", layer_size, input_size);
            new_state.v = View2D("adam_v_w", layer_size, input_size);
            Kokkos::deep_copy(new_state.m, 0.0);
            Kokkos::deep_copy(new_state.v, 0.0);
            state_weights[w_ptr] = new_state; // Use map's operator[] which default constructs if key not found
        }
        AdamState2D& current_w_state = state_weights.at(w_ptr); // Get references to state views

        Kokkos::parallel_for("adam_update_weights",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                real grad = scale * accumulated_d_weights(i, j);

                // Update biased first moment estimate: m = beta1 * m + (1-beta1) * grad
                current_w_state.m(i, j) = beta1 * current_w_state.m(i, j) + (1.0f - beta1) * grad;

                // Update biased second raw moment estimate: v = beta2 * v + (1-beta2) * grad^2
                current_w_state.v(i, j) = beta2 * current_w_state.v(i, j) + (1.0f - beta2) * grad * grad;

                // Compute bias-corrected moment estimates
                real m_hat = current_w_state.m(i, j) * m_hat_factor;
                real v_hat = current_w_state.v(i, j) * v_hat_factor;

                // Update weights: w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
                weights(i, j) -= learning_rate * m_hat / (Kokkos::sqrt(v_hat) + epsilon);
        });

        // --- Update Biases ---
        const real* b_ptr = biases.data();
        if (state_biases.find(b_ptr) == state_biases.end()) {
            // Initialize state for these biases if not present
            AdamState1D new_state;
            new_state.m = View1D("adam_m_b", layer_size);
            new_state.v = View1D("adam_v_b", layer_size);
            Kokkos::deep_copy(new_state.m, 0.0);
            Kokkos::deep_copy(new_state.v, 0.0);
            state_biases[b_ptr] = new_state;
        }
        AdamState1D& current_b_state = state_biases.at(b_ptr); // Get references to state views

        Kokkos::parallel_for("adam_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
            real grad = scale * accumulated_d_biases(i);

            // Update biased first moment estimate: m = beta1 * m + (1-beta1) * grad
            current_b_state.m(i) = beta1 * current_b_state.m(i) + (1.0f - beta1) * grad;

            // Update biased second raw moment estimate: v = beta2 * v + (1-beta2) * grad^2
            current_b_state.v(i) = beta2 * current_b_state.v(i) + (1.0f - beta2) * grad * grad;

            // Compute bias-corrected moment estimates
            real m_hat = current_b_state.m(i) * m_hat_factor;
            real v_hat = current_b_state.v(i) * v_hat_factor;

            // Update biases: b = b - lr * m_hat / (sqrt(v_hat) + epsilon)
            biases(i) -= learning_rate * m_hat / (Kokkos::sqrt(v_hat) + epsilon);
        });
    }

    // Optional: Reset timestep (e.g., if reusing optimizer for a new training run)
    void reset_timestep() {
        t = 0;
        // Note: This does *not* clear the moment estimates (m, v).
        // If you need to fully reset, you'd clear the state maps too.
        // state_weights.clear();
        // state_biases.clear();
    }

     std::string get_info() const override {
        return "Adam, LR: " + std::to_string(learning_rate) +
               ", Beta1: " + std::to_string(beta1) +
               ", Beta2: " + std::to_string(beta2) +
               ", Epsilon: " + std::to_string(epsilon) +
               ", t: " + std::to_string(t);
    }
};



// --- Classes Activation (inchangées) ---
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
    // Optimizer optimizer; // *** SUPPRIMÉ ***

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
        // Utilisation de l'initialisation de Xavier/Glorot (approximative pour uniform)
        real limit = std::sqrt(6.0f / (input_size + layer_size));
        Kokkos::fill_random(weights, rand_pool, static_cast<real>(-limit), static_cast<real>(limit));
        // Kokkos::fill_random(weights, rand_pool, static_cast<real>(-0.2), static_cast<real>(0.2)); // Ancienne initialisation
        Kokkos::deep_copy(biases, 0.0); // Souvent initialisé à 0
        // Kokkos::fill_random(biases, rand_pool, static_cast<real>(-0.1), static_cast<real>(0.1)); // Ancienne initialisation


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

    // --- Méthodes ---

    // Calcule la sortie de la couche (inchangée)
    virtual void forward(const View1D& prev_layer_a) {
        if (!activation || input_size == 0) return; // Ne rien faire pour InputLayer

        // 1. Calculer z = W * prev_layer_a + b
        KokkosBlas::gemv("N", 1.0, weights, prev_layer_a, 0.0, z);
        Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) {
            z(i) += biases(i);
        });

        // 2. Calculer a = activation(z)
        activation->apply(z, a);
    }

     // Calcule les gradients pour les couches cachées (inchangée)
    virtual void compute_gradients(const Layer& next_layer, const View1D& prev_layer_a) {
         if (!activation || input_size == 0) return; // Ne rien faire pour InputLayer

        // 1. Calculer la dérivée de l'activation f'(z) et la stocker
        activation->apply_derivative(z, tmp_deriv);

        // 2. Calculer delta de cette couche : δ_l = (W_{l+1}^T * δ_{l+1}) .* f'(z_l)
        KokkosBlas::gemv("T", 1.0, next_layer.weights, next_layer.delta, 0.0, delta);
         Kokkos::parallel_for("hadamard_delta_deriv", layer_size, KOKKOS_LAMBDA(int i) {
            delta(i) *= tmp_deriv(i);
        });

        // 3. Calculer les gradients instantanés pour ce sample
        Kokkos::deep_copy(d_biases, delta);
        Kokkos::parallel_for("compute_d_weights", layer_size, KOKKOS_LAMBDA(int i) {
            for (int j = 0; j < input_size; ++j) {
                d_weights(i, j) = delta(i) * prev_layer_a(j);
            }
        });

        // 4. Accumuler les gradients (pour Batch Gradient Descent)
        Kokkos::parallel_for("accumulate_gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
            KOKKOS_LAMBDA (const int i, const int j) {
                // Utiliser l'opérateur atomique pour éviter les race conditions si exécuté en parallèle sur le même batch plus tard
                 // Pour SGD simple sur un seul thread par sample, l'atomique n'est pas nécessaire ici, mais bonne pratique
                 // Kokkos::atomic_add(&d_weights_sum(i, j), d_weights(i,j)); // Si besoin d'atomicité
                 d_weights_sum(i, j) += d_weights(i, j);
        });
         Kokkos::parallel_for("accumulate_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             // Kokkos::atomic_add(&d_biases_sum(i), d_biases(i)); // Si besoin d'atomicité
             d_biases_sum(i) += d_biases(i);
        });
    }


    // Met à jour les poids et biais (appelé après un batch)
    // *** SUPPRIMÉ - La logique est maintenant dans Optimizer::update ***
    /*
    virtual void update_weights(real learning_rate, int batch_size) {
        if (input_size == 0) return; // Pas de poids/biais pour InputLayer
        // ... ancienne logique ...
    }
    */

    // Remet à zéro les gradients accumulés (inchangée)
    void zero_accumulated_gradients() {
         if (input_size == 0) return;
         Kokkos::deep_copy(d_biases_sum, 0.0);
         Kokkos::deep_copy(d_weights_sum, 0.0);
    }

    // Afficher poids et biais (inchangée)
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
        for(int i=0; i< std::min(layer_size, 5) ; ++i) {
             std::cout << "    [";
             for(int j=0; j< std::min(input_size, 5); ++j) {
                 std::cout << std::fixed << std::setprecision(3) << h_weights(i,j) << " ";
             }
             if (input_size > 5) std::cout << "...";
             std::cout << "]" << std::endl;
        }
         if (layer_size > 5) std::cout << "    ..." << std::endl;

        std::cout << "  Biases (sample): [";
        for(int i=0; i< std::min(layer_size, 10); ++i) {
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

// --- Classe InputLayer (inchangée) ---
class InputLayer : public Layer {
public:
    InputLayer(int _layer_size) : Layer(_layer_size) {}

    void set_input(const View1D& input_data) {
         if (input_data.extent(0) != layer_size) {
             throw std::runtime_error("Input data size mismatch for InputLayer.");
         }
         Kokkos::deep_copy(a, input_data);
     }
};

// --- Classe OutputLayer (compute_gradients inchangée, update_weights supprimée) ---
class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        Layer(_input_size, _layer_size, std::move(act_func)) {}

    // Calcule les gradients pour la couche de sortie (inchangée)
    void compute_gradients(const View1D& target, const View1D& prev_layer_a) {
         if (!activation) return;

        // 1. Calculer la dérivée de l'activation f'(z)
        activation->apply_derivative(z, tmp_deriv);

        // 2. Calculer delta pour la couche de sortie : δ_L = (a_L - y) .* f'(z_L)
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
    std::unique_ptr<Optimizer> optimizer; // *** AJOUTÉ ***
    Dataset dataset; // Placeholder
    BatchHandler batch_handler; // Placeholder

    // Constructeur (modifié pour initialiser l'optimizer)
    Network(const std::map<int, int>& _layer_sizes,
            const std::vector<std::string>& activation_types,
//          std::unique_ptr<Optimizer> opt = std::make_unique<SGD>(0.1)) // Default SGD optimizer
            std::unique_ptr<Optimizer> opt = std::make_unique<RMSprop>(0.1, 0.00001)) // Default SGD optimizer									 // 
        : layer_sizes(_layer_sizes),
          input_layer(get_size(0)),
          output_layer(get_size(num_layers() - 2), get_size(num_layers() - 1), create_activation(activation_types.back())),
          optimizer(std::move(opt)) // Prend possession de l'optimizer fourni ou du défaut
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
        for (int i = 0; i < num_hidden_layers; ++i) {
            int current_layer_idx = i + 1;
            int in_size = get_size(current_layer_idx - 1);
            int out_size = get_size(current_layer_idx);
            std::string act_type = activation_types[i];
            hidden_layers.emplace_back(in_size, out_size, create_activation(act_type));
        }
         if (!optimizer) {
             throw std::runtime_error("Optimizer cannot be null after network construction.");
         }
        //std::cout << "Network creation complete." << std::endl;
    }

    // Helper (inchangés)
    int get_size(int layer_index) const { return layer_sizes.at(layer_index); }
    int num_layers() const { return layer_sizes.size(); }


    // --- Méthodes modifiées ---

    // Effectue la passe avant (prédiction) - inchangée
    View1D forward(const View1D& input_data) {
        input_layer.set_input(input_data);
        View1D current_a = input_layer.a;
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.forward(current_a);
            current_a = hidden_layer.a;
        }
        output_layer.forward(current_a);
        return output_layer.a;
    }

    // Effectue la passe arrière (calcul des gradients) pour un sample - inchangée
    void backward(const View1D& target) {
        const View1D& prev_a_output = hidden_layers.empty() ? input_layer.a : hidden_layers.back().a;
        output_layer.compute_gradients(target, prev_a_output);

        Layer* next_layer_ptr = &output_layer;
        for (int i = hidden_layers.size() - 1; i >= 0; --i) {
            Layer& current_layer = hidden_layers[i];
            const View1D& prev_a = (i == 0) ? input_layer.a : hidden_layers[i - 1].a;
            current_layer.compute_gradients(*next_layer_ptr, prev_a);
            next_layer_ptr = &current_layer;
        }
    }

     // Met à jour les poids de toutes les couches après un batch (utilise l'optimizer)
    void update(int batch_size) { // Ne prend plus learning_rate en argument
        if (!optimizer) {
             throw std::runtime_error("Optimizer not set in Network.");
        }
         if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for update.");
        }

        // Mettre à jour les couches cachées
        for (Layer& layer : hidden_layers) {
            // L'optimizer a besoin des poids/biais (non-const) et des gradients (const)
            optimizer->update(layer.weights, layer.biases,
                              layer.d_weights_sum, layer.d_biases_sum,
                              batch_size);
        }
        // Mettre à jour la couche de sortie
        optimizer->update(output_layer.weights, output_layer.biases,
                          output_layer.d_weights_sum, output_layer.d_biases_sum,
                          batch_size);
    }

    // Remet à zéro les gradients accumulés de toutes les couches (inchangée)
    void zero_accumulated_gradients() {
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.zero_accumulated_gradients();
        }
        output_layer.zero_accumulated_gradients();
    }

    // Calcule le coût (inchangée)
    real calculate_cost(const View1D& prediction, const View1D& target) {
        int output_size = prediction.extent(0);
        if (target.extent(0) != output_size) {
            throw std::runtime_error("Prediction and target size mismatch for cost calculation.");
        }
        real squared_error_sum = 0.0;
        Kokkos::parallel_reduce("compute_cost", output_size, KOKKOS_LAMBDA (int i, real& lsum) {
            real diff = prediction(i) - target(i);
            lsum += diff * diff;
        }, squared_error_sum);
        Kokkos::fence();
        return 0.5 * squared_error_sum;
    }

    // Méthodes pour gérer l'optimizer
    void set_optimizer(std::unique_ptr<Optimizer> opt) {
        if (!opt) {
            throw std::runtime_error("Cannot set a null optimizer.");
        }
        optimizer = std::move(opt);
    }

    Optimizer* get_optimizer() const {
        return optimizer.get();
    }

    void set_learning_rate(real lr) {
        if (!optimizer) {
            throw std::runtime_error("Optimizer not set, cannot set learning rate.");
        }
        optimizer->set_learning_rate(lr);
    }

     real get_learning_rate() const {
         if (!optimizer) {
             // Ou retourner une valeur par défaut / lancer une exception
             return 0.0;
         }
        return optimizer->get_learning_rate();
    }


    // Affiche les informations des couches (inchangée)
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
         std::cout << "--- Optimizer Info ---" << std::endl;
         if(optimizer) {
            // Pourrait ajouter un `virtual std::string get_info() const` à Optimizer
            std::cout << "  Type: " << typeid(*optimizer).name() << std::endl; // Nom mangled
            std::cout << "  Learning Rate: " << optimizer->get_learning_rate() << std::endl;
         } else {
            std::cout << "  Optimizer: Not set" << std::endl;
         }
         std::cout << "-------------------------" << std::endl;
    }


    // Déplacement/Copie (inchangés)
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    virtual ~Network() { // virtual car on pourrait en hériter
        //std::cout << "Destroying Network..." << std::endl;
    }
};

// --- Fonction d'entraînement XOR (adaptée) ---
void xor_train() {
    std::cout << "\n--- XOR Training Example ---" << std::endl;

    // Structure du réseau
    std::map<int, int> sizes;
    sizes[0] = 2; // Input
    sizes[1] = 4; // Hidden 1 (Augmenté un peu la capacité)
    sizes[2] = 1; // Output

    std::vector<std::string> activations = {"tanh", "tanh"}; // H1, Out

    // Paramètres d'entraînement
    real learning_rate = 0.1; // Taux d'apprentissage
    int epochs = 50000;       // Nombre d'époques (peut nécessiter ajustement)
    int batch_size = 4;       // Batch Gradient Descent (toutes les données)

    // Créer l'optimizer SGD
    auto sgd_optimizer = std::make_unique<SGD>(learning_rate);

    // Créer le réseau en lui passant l'optimizer
    Network dnn(sizes, activations, std::move(sgd_optimizer));

    // Alternative: créer le réseau avec le défaut, puis le configurer
    // Network dnn(sizes, activations); // Utilise SGD(0.1) par défaut
    // dnn.set_learning_rate(learning_rate); // Définit le taux d'apprentissage désiré

    // Données XOR (inchangé)
    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    const int num_samples = 4;
    const int input_dim = sizes[0];
    const int output_dim = sizes[sizes.size()-1];
    HostView2D h_xor_inputs("h_xor_inputs", num_samples, input_dim);
    HostView2D h_xor_outputs("h_xor_outputs", num_samples, output_dim);
    h_xor_inputs(0, 0) = 0.0; h_xor_inputs(0, 1) = 0.0; h_xor_outputs(0, 0) = 0.0;
    h_xor_inputs(1, 0) = 1.0; h_xor_inputs(1, 1) = 0.0; h_xor_outputs(1, 0) = 1.0;
    h_xor_inputs(2, 0) = 0.0; h_xor_inputs(2, 1) = 1.0; h_xor_outputs(2, 0) = 1.0;
    h_xor_inputs(3, 0) = 1.0; h_xor_inputs(3, 1) = 1.0; h_xor_outputs(3, 0) = 0.0;
    auto xor_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_inputs);
    auto xor_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_xor_outputs);

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << "Epochs: " << epochs << ", Learning Rate: " << dnn.get_learning_rate() << ", Batch Size: " << batch_size << std::endl;


    for (int epoch = 0; epoch < epochs; ++epoch) {
        real total_epoch_cost = 0.0;
        dnn.zero_accumulated_gradients(); // Important: remettre à zéro avant le batch

        // Itérer sur les samples du batch
        for (int i = 0; i < num_samples; ++i) {
            auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
            auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());

            // 1. Forward pass
            View1D prediction = dnn.forward(input_subview);

            // 2. Calculer le coût pour ce sample
            real sample_cost = dnn.calculate_cost(prediction, target_subview);
            total_epoch_cost += sample_cost;

            // 3. Backward pass (accumule les gradients)
            dnn.backward(target_subview);
        }

        // 4. Mettre à jour les poids après avoir vu tous les samples du batch
        //    Utilise l'optimizer interne et son learning rate.
        dnn.update(batch_size); // *** MODIFIÉ ***

        // Afficher le coût moyen de l'époque
        if ((epoch + 1) % 5000 == 0 || epoch == 0) { // Affichage moins fréquent
            std::cout << "Epoch: " << std::setw(6) << epoch + 1
                      << ", Average Cost: " << std::fixed << std::setprecision(8)
                      << total_epoch_cost / num_samples << std::endl;
        }

        // Optionnel: Stopper si le coût est suffisamment bas
        if (total_epoch_cost / num_samples < 1e-4) {
             std::cout << "Convergence reached at epoch " << epoch + 1 << std::endl;
             break;
        }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    // Afficher la structure et les poids finaux
    dnn.show();

    // Tester les prédictions (inchangé)
    std::cout << "\n-- PREDICTIONS --" << std::endl;
    View1D prediction_result("prediction_result", output_dim); // Sur device
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result); // Miroir sur host

    for (int i = 0; i < num_samples; ++i) {
        auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
        dnn.forward(input_subview); // Recalcule la sortie avec les poids finaux
        Kokkos::deep_copy(prediction_result, dnn.output_layer.a); // Copie le résultat (device)
        Kokkos::deep_copy(h_prediction_result, prediction_result); // Copie vers host
        Kokkos::fence(); // Assurer la fin des copies

        std::cout << "Input:  [" << h_xor_inputs(i, 0) << ", " << h_xor_inputs(i, 1) << "]"
                  << ", Target: [" << h_xor_outputs(i, 0) << "]"
                  << ", Predicted: [" << std::fixed << std::setprecision(4) << h_prediction_result(0) << "]"
                  << " (Raw output)" << std::endl;
    }
     std::cout << "-----------------" << std::endl;
}


// --- Fonction Main (inchangée) ---
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    { // Scope pour Kokkos
        std::cout << "--- Neural Network Library v3 (with Optimizer) ---" << std::endl;

        try {
            xor_train(); // Lance l'entraînement XOR
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
