#include <iostream>
#include <limits>
#include <cmath> // Pour sin, M_PI
#include <fstream>
#include <filesystem>
#include <random>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <numeric> // Pour std::iota
#include <iomanip> // Pour std::setprecision
#include <typeinfo> // Pour typeid dans Network::show
#include <chrono>  // Pour seed random

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp> // Pour gemv (biases) et peut-être temporairement
#include <Kokkos_StdAlgorithms.hpp> // Pour parallel_reduce

// --- NOUVEAU: Includes KokkosKernels ---
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosKernels_Handle.hpp> // Nécessaire pour certaines opérations, bien que spmv puisse souvent l'inférer
#include <KokkosSparse_Utils.hpp> // Pour des utilitaires potentiels (comme la conversion, si disponible publiquement, sinon on l'implémente)


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Typedefs ---
using real = float;
using View1D = Kokkos::View<real*>;
using View2D = Kokkos::View<real**>;
// using View3D = Kokkos::View<real***>; // Pas utilisé ici

// --- NOUVEAU: Typedef pour la matrice éparse ---
// Utilise 'int' pour les ordinaux/offsets par défaut dans KokkosKernels
// Assurez-vous que la taille des couches ne dépasse pas les limites de 'int'
using OrdinalType = int;
using SizeType    = size_t; // Souvent size_t pour row_map
using DeviceType  = Kokkos::DefaultExecutionSpace;
using SparseMatrix = KokkosSparse::CrsMatrix<real, OrdinalType, DeviceType, void, SizeType>;
using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
    SizeType, OrdinalType, real,
    DeviceType, DeviceType, DeviceType>; // Handle pour SpMV si nécessaire explicitement

// --- Forward Declarations ---
class Optimizer;
class SGD;
class Adam;
class Layer;
class Activation;
class Dataset;
class Network;
class BatchHandler;


// --- NOUVEAU: Fonctions utilitaires Dense <-> Sparse ---

// Convertit une View2D dense en CrsMatrix éparse
// NOTE: Cette implémentation suppose que la matrice dense d'entrée *est* dense
// et crée une structure CRS correspondante (inefficace si la matrice était déjà éparse).
// C'est utile pour l'initialisation où l'on part d'une Glorot/Xavier dense.
SparseMatrix dense_to_sparse(const View2D& dense_matrix) {
    const size_t numRows = dense_matrix.extent(0);
    const size_t numCols = dense_matrix.extent(1);

    if (numRows == 0 || numCols == 0) {
        // Retourner une matrice vide valide
        return SparseMatrix("empty_sparse", numRows, numCols, 0, nullptr, nullptr, nullptr);
    }

    // Créer une copie Host pour accéder aux données
    auto h_dense_matrix = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dense_matrix);
    Kokkos::fence(); // Assurer la fin de la copie

    // --- Calculer la structure CRS (en supposant dense) ---
    const size_t nnz = numRows * numCols; // Nombre d'éléments non nuls (tous)

    // Allouer les vues pour la structure CRS sur l'hôte
    Kokkos::View<SizeType*, Kokkos::HostSpace> h_row_map("h_row_map", numRows + 1);
    Kokkos::View<OrdinalType*, Kokkos::HostSpace> h_col_idx("h_col_idx", nnz);
    Kokkos::View<real*, Kokkos::HostSpace> h_values("h_values", nnz);

    size_t nz_count = 0;
    h_row_map(0) = 0;
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            h_values(nz_count) = h_dense_matrix(i, j);
            h_col_idx(nz_count) = static_cast<OrdinalType>(j); // Cast en OrdinalType
            nz_count++;
        }
        h_row_map(i + 1) = nz_count;
    }

    // --- Copier la structure CRS vers le Device ---
    auto d_row_map = Kokkos::create_mirror_view_and_copy(DeviceType(), h_row_map);
    auto d_col_idx = Kokkos::create_mirror_view_and_copy(DeviceType(), h_col_idx);
    auto d_values = Kokkos::create_mirror_view_and_copy(DeviceType(), h_values);
    Kokkos::fence(); // Assurer la fin des copies

    // Créer la CrsMatrix
    SparseMatrix sparse_mat("sparse_matrix", numRows, numCols, nnz, d_values, d_row_map, d_col_idx);
    return sparse_mat;
}

// Convertit une CrsMatrix éparse en View2D dense
View2D sparse_to_dense(const SparseMatrix& sparse_matrix) {
    const size_t numRows = sparse_matrix.numRows();
    const size_t numCols = sparse_matrix.numCols();

    // Allouer la matrice dense sur le device, initialisée à 0
    View2D dense_matrix("dense_matrix", numRows, numCols);
    Kokkos::deep_copy(dense_matrix, 0.0);

    // Si la matrice éparse est vide, retourner la matrice dense de zéros
    if (sparse_matrix.nnz() == 0) {
        return dense_matrix;
    }

    // Obtenir les vues de la structure CRS (sur le device)
    auto row_map = sparse_matrix.graph.row_map;
    auto col_idx = sparse_matrix.graph.entries;
    auto values = sparse_matrix.values;

    // Kernel pour remplir la matrice dense à partir de la structure éparse
    Kokkos::parallel_for("sparse_to_dense_fill", numRows, KOKKOS_LAMBDA(const OrdinalType i) {
        // Utiliser SizeType pour l'indexation de row_map
        const SizeType row_start = row_map(i);
        const SizeType row_end = row_map(i + 1);
        for (SizeType k = row_start; k < row_end; ++k) {
            const OrdinalType j = col_idx(k);
            if (j >= 0 && j < static_cast<OrdinalType>(numCols)) { // Vérification des bornes
                 dense_matrix(i, j) = values(k);
            }
        }
    });
    Kokkos::fence(); // Assurer la fin du kernel

    return dense_matrix;
}


// --- Classe Optimizer (Base) ---
class Optimizer {
public:
    real learning_rate;

    Optimizer(real lr) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    // Méthode de mise à jour adaptée: prend SparseMatrix pour les poids
    // mais garde View2D pour les gradients accumulés (compromis)
    virtual void update(SparseMatrix& weights, View1D biases, // weights est maintenant SparseMatrix
                        const View2D& accumulated_d_weights, // Gardé dense
                        const View1D& accumulated_d_biases,
                        int batch_size) = 0;

    void set_learning_rate(real lr) { learning_rate = lr; }
    real get_learning_rate() const { return learning_rate; }

    virtual std::string get_info() const {
        return "Optimizer(LR=" + std::to_string(learning_rate) + ")";
    }
};

// --- Concrete Optimizer: SGD (adapté avec conversion) ---
class SGD : public Optimizer {
public:
    SGD(real lr = 0.1) : Optimizer(lr) {}

    void update(SparseMatrix& sparse_weights, View1D biases,
                const View2D& accumulated_d_weights, const View1D& accumulated_d_biases,
                int batch_size) override {

        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for SGD update.");
        }
        // Vérifier si les biais ont une taille > 0. Pour les poids, on ne met à jour que s'il y a des lignes/colonnes.
         if (sparse_weights.numRows() == 0 && biases.extent(0) == 0) {
             return; // Rien à mettre à jour
         }
         // Dimensions des gradients denses
         const int layer_size = accumulated_d_weights.extent_int(0);
         const int input_size = accumulated_d_weights.extent_int(1);

        // --- Mise à jour des poids (via conversion Dense <-> Sparse) ---
         if (sparse_weights.numRows() > 0 && sparse_weights.numCols() > 0) {
             if (static_cast<size_t>(layer_size) != sparse_weights.numRows() || static_cast<size_t>(input_size) != sparse_weights.numCols()) {
                  throw std::runtime_error("SGD update: Dimension mismatch between sparse weights and dense gradients.");
             }

             // 1. Convertir sparse_weights -> dense_weights temporaire
             View2D temp_dense_weights = sparse_to_dense(sparse_weights);

             // 2. Effectuer la mise à jour sur la matrice dense
             const real scale_w = learning_rate / static_cast<real>(batch_size);
             Kokkos::parallel_for("sgd_update_dense_weights_temp",
                 Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {layer_size, input_size}),
                 KOKKOS_LAMBDA (const int i, const int j) {
                     temp_dense_weights(i, j) -= scale_w * accumulated_d_weights(i, j);
             });
             Kokkos::fence();

             // 3. Reconvertir dense_weights temporaire -> sparse_weights
             // L'opérateur d'affectation de CrsMatrix devrait gérer cela (ou recréer)
             sparse_weights = dense_to_sparse(temp_dense_weights); // Réassigne la nouvelle matrice éparse
         }


        // --- Mise à jour des biais (reste inchangée car View1D) ---
        if (biases.extent(0) > 0) {
             if (biases.extent_int(0) != layer_size) {
                  throw std::runtime_error("SGD update: Dimension mismatch for biases.");
             }
             const real scale_b = learning_rate / static_cast<real>(batch_size);
            Kokkos::parallel_for("sgd_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
                biases(i) -= scale_b * accumulated_d_biases(i);
            });
        }
         Kokkos::fence(); // S'assurer que les mises à jour sont terminées avant de continuer
    }

    std::string get_info() const override {
        return "SGD(LR=" + std::to_string(learning_rate) + ")";
    }
};


// --- Concrete Optimizer: Adam (adapté avec conversion) ---
class Adam : public Optimizer {
public:
    real beta1;
    real beta2;
    real epsilon;

private:
    // Structure pour l'état (m, v, t) - m et v sont maintenant potentiellement épars
    struct ParameterState {
        View1D m_1d; // Pour les biais (dense)
        View1D v_1d; // Pour les biais (dense)
        SparseMatrix m_2d; // Pour les poids (épars)
        SparseMatrix v_2d; // Pour les poids (épars)
        long long t = 0;

        // Constructeur pour l'état des biais (dense)
        ParameterState(size_t size) : t(0) {
            if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
                 throw std::runtime_error("Adam state bias size exceeds limits.");
            }
            m_1d = View1D("adam_m1d", size);
            v_1d = View1D("adam_v1d", size);
            Kokkos::deep_copy(m_1d, 0.0);
            Kokkos::deep_copy(v_1d, 0.0);
            // Initialise m_2d/v_2d comme vides mais valides
            m_2d = SparseMatrix("adam_m2d_empty", 0, 0, static_cast<SizeType>(0));
            v_2d = SparseMatrix("adam_v2d_empty", 0, 0, static_cast<SizeType>(0));
        }
        // Constructeur pour l'état des poids (épars)
        ParameterState(size_t rows, size_t cols) : t(0) {
             if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) || cols > static_cast<size_t>(std::numeric_limits<int>::max())) {
                 throw std::runtime_error("Adam state weight dimensions exceed limits.");
            }
            // Initialise comme des matrices éparses ZEROS (0 nnz)
            m_2d = SparseMatrix("adam_m2d", rows, cols, static_cast<SizeType>(0)); // 0 non-zeros initialement
            v_2d = SparseMatrix("adam_v2d", rows, cols, static_cast<SizeType>(0)); // 0 non-zeros initialement
             // Initialise m_1d/v_1d comme vides mais valides
             m_1d = View1D("adam_m1d_empty", 0);
             v_1d = View1D("adam_v1d_empty", 0);
        }
         ParameterState() = default;
         // Attention: CrsMatrix n'est pas trivialement déplaçable/copiable comme View.
         // Il faut s'assurer que la map gère correctement les objets CrsMatrix.
         // L'utilisation de try_emplace et la non-copie/déplacement après insertion devrait être sûr.
         ParameterState(ParameterState&&) = delete; // Interdire move pour simplicité (peut être réactivé avec soin)
         ParameterState& operator=(ParameterState&&) = delete; // Interdire move assignment
         ParameterState(const ParameterState&) = delete; // Interdire copy
         ParameterState& operator=(const ParameterState&) = delete; // Interdire copy assignment
    };

    // Maps pour stocker l'état: clé = pointeur vers les données des poids/biais.
    // NOTE: La clé pointe vers l'objet CrsMatrix ou View1D. Sa stabilité est cruciale.
    std::map<const real*, ParameterState> state_map_1d; // Pour les biais (clé: biases.data())
    std::map<const SparseMatrix*, ParameterState> state_map_2d; // Pour les poids (clé: &sparse_weights)

public:
    Adam(real lr = 0.001, real b1 = 0.9, real b2 = 0.999, real eps = 1e-8)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps) {
        if (lr <= 0 || b1 < 0 || b1 >= 1 || b2 < 0 || b2 >= 1 || eps <= 0) {
            throw std::runtime_error("Invalid Adam hyperparameters.");
        }
    }

    void update(SparseMatrix& sparse_weights, View1D biases, // weights est SparseMatrix
                const View2D& accumulated_d_weights,       // Gardé dense
                const View1D& accumulated_d_biases,
                int batch_size) override {

        if (batch_size <= 0) {
             throw std::runtime_error("Batch size must be positive for Adam update.");
        }
        // Dimensions à partir des gradients (denses) et des paramètres
         const int layer_size = accumulated_d_weights.extent_int(0);
         const int input_size = accumulated_d_weights.extent_int(1);

        // --- Mise à jour des poids (via conversion Dense <-> Sparse) ---
        if (sparse_weights.numRows() > 0 && sparse_weights.numCols() > 0) {
             if (static_cast<size_t>(layer_size) != sparse_weights.numRows() || static_cast<size_t>(input_size) != sparse_weights.numCols()) {
                  throw std::runtime_error("Adam update: Dimension mismatch between sparse weights and dense gradients.");
             }

            // Get or create state for weights (clé = adresse de l'objet CrsMatrix)
            auto it_w = state_map_2d.find(&sparse_weights);
            if (it_w == state_map_2d.end()) {
                auto result = state_map_2d.try_emplace(&sparse_weights, sparse_weights.numRows(), sparse_weights.numCols());
                 if (!result.second) {
                     throw std::runtime_error("Failed to insert Adam state for weights.");
                 }
                 it_w = result.first;
                 //std::cout << "Adam: Initialized sparse state for weights at " << &sparse_weights << std::endl;
            }
            ParameterState& state_w = it_w->second;
            state_w.t++;

            // Convertir les matrices éparses (poids, m, v) en denses temporaires
            View2D temp_dense_weights = sparse_to_dense(sparse_weights);
            View2D temp_dense_m = sparse_to_dense(state_w.m_2d);
            View2D temp_dense_v = sparse_to_dense(state_w.v_2d);

            // Calculer les termes de correction de biais (host)
            const double beta1_pow_t = std::pow(static_cast<double>(beta1), static_cast<double>(state_w.t));
            const double beta2_pow_t = std::pow(static_cast<double>(beta2), static_cast<double>(state_w.t));
            const real bias_correction1 = 1.0f / (1.0f - static_cast<real>(beta1_pow_t));
            const real bias_correction2 = 1.0f / (1.0f - static_cast<real>(beta2_pow_t));
            const real scale = 1.0 / static_cast<real>(batch_size);

            // Effectuer la mise à jour Adam sur les matrices denses temporaires
            real lr = learning_rate;
            real b1 = beta1;
            real b2 = beta2;
            real eps = epsilon;
            Kokkos::parallel_for("adam_update_dense_weights_temp",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
                KOKKOS_LAMBDA (const int i, const int j) {
                    real grad = scale * accumulated_d_weights(i, j);
                    // Update biased moments (using temp_dense_m, temp_dense_v)
                    temp_dense_m(i, j) = b1 * temp_dense_m(i, j) + (1.0f - b1) * grad;
                    temp_dense_v(i, j) = b2 * temp_dense_v(i, j) + (1.0f - b2) * grad * grad;
                    // Compute bias-corrected moments
                    real m_hat = temp_dense_m(i, j) * bias_correction1;
                    real v_hat = temp_dense_v(i, j) * bias_correction2;
                    // Update weights (using temp_dense_weights)
                    temp_dense_weights(i, j) -= lr * m_hat / (Kokkos::sqrt(v_hat) + eps);
            });
            Kokkos::fence();

            // Reconvertir les matrices denses temporaires en éparses
            sparse_weights = dense_to_sparse(temp_dense_weights);
            state_w.m_2d = dense_to_sparse(temp_dense_m);
            state_w.v_2d = dense_to_sparse(temp_dense_v);

        } // End weights update

        // --- Mise à jour des biais (reste dense, inchangé) ---
        if (biases.extent(0) > 0) {
            if (biases.extent_int(0) != layer_size) {
                 throw std::runtime_error("Adam update: Dimension mismatch for biases.");
             }
             // Get or create state for biases (clé = pointeur de données de la View)
             auto it_b = state_map_1d.find(biases.data());
             if (it_b == state_map_1d.end()) {
                 auto result = state_map_1d.try_emplace(biases.data(), biases.extent(0));
                 if (!result.second) {
                     throw std::runtime_error("Failed to insert Adam state for biases.");
                 }
                 it_b = result.first;
                  //std::cout << "Adam: Initialized dense state for biases at " << biases.data() << std::endl;
             }
             ParameterState& state_b = it_b->second;
             state_b.t++;

             const double beta1_pow_t = std::pow(static_cast<double>(beta1), static_cast<double>(state_b.t));
             const double beta2_pow_t = std::pow(static_cast<double>(beta2), static_cast<double>(state_b.t));
             const real bias_correction1 = 1.0f / (1.0f - static_cast<real>(beta1_pow_t));
             const real bias_correction2 = 1.0f / (1.0f - static_cast<real>(beta2_pow_t));
             const real scale = 1.0 / static_cast<real>(batch_size);

             real lr = learning_rate;
             real b1 = beta1;
             real b2 = beta2;
             real eps = epsilon;
             View1D m = state_b.m_1d; // Références directes (dense)
             View1D v = state_b.v_1d; // Références directes (dense)

             Kokkos::parallel_for("adam_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
                 real grad = scale * accumulated_d_biases(i);
                 m(i) = b1 * m(i) + (1.0f - b1) * grad;
                 v(i) = b2 * v(i) + (1.0f - b2) * grad * grad;
                 real m_hat = m(i) * bias_correction1;
                 real v_hat = v(i) * bias_correction2;
                 biases(i) -= lr * m_hat / (Kokkos::sqrt(v_hat) + eps);
             });
        } // End biases update
         Kokkos::fence(); // Attendre la fin des mises à jour
    } // End update method

     std::string get_info() const override {
         return "Adam(LR=" + std::to_string(learning_rate) +
                ", beta1=" + std::to_string(beta1) +
                ", beta2=" + std::to_string(beta2) +
                ", epsilon=" + std::to_string(epsilon) + ")";
     }
     virtual ~Adam() override = default;
};


// --- Classes Activation (inchangées) ---
// ... (RELU, SIGMOID, TANH, LinearActivation, create_activation restent identiques)
class Activation {
public:
    virtual ~Activation() = default;
    virtual void apply(const View1D& input, View1D& output) const = 0;
    virtual void apply_derivative(const View1D& input_z, View1D& output_deriv) const = 0;
};

class LinearActivation : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override {
        Kokkos::deep_copy(output, input);
    }
    void apply_derivative(const View1D& /*input_z*/, View1D& output_deriv) const override {
        Kokkos::deep_copy(output_deriv, 1.0f);
    }
};

class RELU : public Activation {
public:
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent_int(0);
        Kokkos::parallel_for("relu_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = (input(i) > 0.0f) ? input(i) : 0.0f;
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent_int(0);
        Kokkos::parallel_for("relu_deriv", size, KOKKOS_LAMBDA(const int i) {
            output_deriv(i) = (input_z(i) > 0.0f) ? 1.0f : 0.0f;
        });
    }
};

class SIGMOID : public Activation {
public:
    KOKKOS_INLINE_FUNCTION real scalar_sigmoid(real x) const {
        x = Kokkos::max(-30.0f, Kokkos::min(30.0f, x)); // Clamping
        return 1.0f / (1.0f + Kokkos::exp(-x));
    }
    void apply(const View1D& input, View1D& output) const override {
        const int size = input.extent_int(0);
        Kokkos::parallel_for("sigmoid_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = scalar_sigmoid(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent_int(0);
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
        const int size = input.extent_int(0);
        Kokkos::parallel_for("tanh_apply", size, KOKKOS_LAMBDA(const int i) {
            output(i) = scalar_tanh(input(i));
        });
    }
    void apply_derivative(const View1D& input_z, View1D& output_deriv) const override {
        const int size = input_z.extent_int(0);
        Kokkos::parallel_for("tanh_deriv", size, KOKKOS_LAMBDA(const int i) {
            real t = scalar_tanh(input_z(i));
            output_deriv(i) = 1.0f - t * t;
        });
    }
};

std::unique_ptr<Activation> create_activation(const std::string& type) {
    if (type == "relu") return std::make_unique<RELU>();
    if (type == "sigmoid") return std::make_unique<SIGMOID>();
    if (type == "tanh") return std::make_unique<TANH>();
    if (type == "linear") return std::make_unique<LinearActivation>();
    throw std::runtime_error("Unknown activation type: " + type);
}


// --- Classe Layer (adaptée pour sparse_weights) ---
class Layer {
public:
    int input_size;
    int layer_size;
    std::unique_ptr<Activation> activation;

    // --- Modifié: Utilisation de SparseMatrix pour les poids ---
    SparseMatrix sparse_weights; // [layer_size x input_size]

    // --- Conservés (Denses) ---
    View1D biases;         // [layer_size]
    View1D z;              // [layer_size] - Sortie avant activation
    View1D a;              // [layer_size] - Sortie après activation

    // --- Backpropagation (Denses) ---
    View1D delta;          // [layer_size] - Erreur propagée (δ)
    View1D d_biases;       // [layer_size] - Gradient instantané biais (pour UN sample)
    View1D d_biases_sum;   // [layer_size] - Somme gradients biais (batch)
    // View2D d_weights;   // Supprimé - Calculé et accumulé directement
    View2D d_weights_sum;  // [layer_size x input_size] - Somme gradients poids (batch) - RESTE DENSE

    View1D tmp_deriv;      // [layer_size] - Stockage temporaire dérivée activation f'(z)


    // Constructeur (adapté pour initialiser sparse_weights)
    Layer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        input_size(_input_size),
        layer_size(_layer_size),
        activation(std::move(act_func))
    {
        if (input_size < 0 || layer_size <= 0) {
             throw std::runtime_error("Layer sizes must be non-negative, layer_size > 0.");
        }

        // Allouer les Views 1D (biais, états, gradients biais)
        biases = View1D("biases", layer_size);
        z = View1D("z", layer_size);
        delta = View1D("delta", layer_size);
        d_biases = View1D("d_biases", layer_size); // Instantané
        d_biases_sum = View1D("d_biases_sum", layer_size); // Accumulé
        tmp_deriv = View1D("tmp_deriv", layer_size);

        // Initialiser biais à 0 et autres Views 1D
        Kokkos::deep_copy(biases, 0.0);
        Kokkos::deep_copy(z, 0.0);
        Kokkos::deep_copy(delta, 0.0);
        Kokkos::deep_copy(d_biases, 0.0);
        Kokkos::deep_copy(d_biases_sum, 0.0);
        Kokkos::deep_copy(tmp_deriv, 0.0);

        // Gérer les poids (sparse_weights) et gradients poids (d_weights_sum - dense)
        if (input_size > 0) {
            // --- Initialisation de sparse_weights via Dense -> Sparse ---
            // 1. Créer une matrice dense temporaire
            View2D temp_dense_weights("temp_dense_weights", layer_size, input_size);
            // 2. Initialiser la matrice dense (Xavier/Glorot)
            Kokkos::Random_XorShift64_Pool<> rand_pool(std::chrono::high_resolution_clock::now().time_since_epoch().count() + reinterpret_cast<uintptr_t>(this));
            real limit = (input_size + layer_size > 0) ? std::sqrt(6.0f / (input_size + layer_size)) : 1.0f;
            Kokkos::fill_random(temp_dense_weights, rand_pool, static_cast<real>(-limit), static_cast<real>(limit));
            // 3. Convertir la matrice dense en sparse_weights
            sparse_weights = dense_to_sparse(temp_dense_weights);
            // La matrice dense temporaire est détruite ici

            // --- Initialisation de d_weights_sum (dense) ---
            d_weights_sum = View2D("d_weights_sum", layer_size, input_size);
            Kokkos::deep_copy(d_weights_sum, 0.0); // Initialiser à 0

        } else {
             // Si input_size est 0, créer des matrices/vues vides mais valides
             sparse_weights = SparseMatrix("weights_empty", layer_size, 0, static_cast<SizeType>(0)); // Note: layer_size != 0
             d_weights_sum = View2D("d_weights_sum_empty", layer_size, 0);
        }

        // 'a' est toujours nécessaire (sortie de la couche)
        a = View1D("a", layer_size);
        Kokkos::deep_copy(a, 0.0);
    }

    // Constructeur pour InputLayer (inchangé conceptuellement)
    Layer(int _layer_size) :
        Layer(0, _layer_size, nullptr) {}

    // --- Méthodes (adaptées) ---

    // Forward pass (utilise SpMV)
    virtual void forward(const View1D& prev_layer_a) {
        if (input_size == 0) {
             return; // InputLayer géré ailleurs
         }
         if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("Forward pass: prev_layer_a size mismatch.");
         }
         if (sparse_weights.numRows() != static_cast<size_t>(layer_size) || sparse_weights.numCols() != static_cast<size_t>(input_size)) {
             throw std::runtime_error("Forward pass: sparse_weights dimensions mismatch.");
         }

        // 1. z = W * prev_layer_a + b
        // Utiliser KokkosSparse::spmv
        // "N" = No transpose
        KokkosSparse::spmv("N", 1.0, sparse_weights, prev_layer_a, 0.0, z); // z = 1.0 * W * prev_a + 0.0 * z

        // Ajouter les biais (inchangé)
        if (biases.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Bias dimension mismatch in forward pass.");
        }
        Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) {
            z(i) += biases(i);
        });

        // 2. a = activation(z) (inchangé)
        if (a.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Activation output 'a' dimension mismatch.");
        }
        if (activation) {
            activation->apply(z, a);
        } else {
            Kokkos::deep_copy(a, z); // Linear activation
        }
         Kokkos::fence(); // Assurer la fin des calculs forward
    }

    // Compute gradients (adapte SpMV^T et accumulation directe dans d_weights_sum dense)
    virtual void compute_gradients(const Layer& next_layer, const View1D& prev_layer_a) {
         if (input_size == 0) return; // InputLayer

         // Vérifications de dimensions (sparse_weights de la couche suivante)
          if (next_layer.sparse_weights.numCols() != static_cast<size_t>(layer_size)) {
               throw std::runtime_error("Dimension mismatch (next_layer.W_cols vs layer_size).");
           }
           if (next_layer.delta.extent(0) != next_layer.sparse_weights.numRows()) {
               throw std::runtime_error("Dimension mismatch (next_layer.delta vs next_layer.W_rows).");
           }
          if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("Dimension mismatch for prev_layer_a.");
         }
          if (delta.extent(0) != static_cast<size_t>(layer_size) || tmp_deriv.extent(0) != static_cast<size_t>(layer_size)) {
              throw std::runtime_error("Internal dimension mismatch for delta or tmp_deriv.");
          }
           if (d_weights_sum.extent(0) != static_cast<size_t>(layer_size) || d_weights_sum.extent(1) != static_cast<size_t>(input_size)) {
               throw std::runtime_error("d_weights_sum dimension mismatch.");
           }


        // 1. Calculer f'(z) -> tmp_deriv (inchangé)
        if (activation) {
             activation->apply_derivative(z, tmp_deriv);
        } else {
            Kokkos::deep_copy(tmp_deriv, 1.0f);
        }

        // 2. Calculer delta: δ_l = (W_{l+1}^T * δ_{l+1}) .* f'(z_l)
        View1D delta_prop("delta_prop", layer_size);
        // Utiliser SpMV avec Transpose "T" sur sparse_weights de la couche suivante
        KokkosSparse::spmv("T", 1.0, next_layer.sparse_weights, next_layer.delta, 0.0, delta_prop);

        Kokkos::parallel_for("hadamard_delta_deriv", layer_size, KOKKOS_LAMBDA(int i) {
            delta(i) = delta_prop(i) * tmp_deriv(i);
        });

        // 3. Calculer gradient biais instantané -> d_biases (inchangé)
        if (d_biases.extent(0) != delta.extent(0)) throw std::runtime_error("d_biases/delta size mismatch.");
        Kokkos::deep_copy(d_biases, delta);

        // 4. Calculer gradient poids instantané (produit externe) ET l'accumuler dans d_weights_sum (dense)
        // dW = delta * prev_layer_a^T
        // d_weights_sum += dW (via atomic add)
        Kokkos::parallel_for("compute_accumulate_d_weights",
             Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
             KOKKOS_LAMBDA(const int i, const int j) {
                 real dw_ij = delta(i) * prev_layer_a(j); // Calcul du gradient instantané
                 Kokkos::atomic_add(&d_weights_sum(i, j), dw_ij); // Accumulation directe
        });

        // 5. Accumuler gradient biais (inchangé)
         if (d_biases_sum.extent(0) != d_biases.extent(0)) {
             throw std::runtime_error("Accumulated d_biases dimension mismatch.");
         }
         Kokkos::parallel_for("accumulate_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
         });
         Kokkos::fence(); // Assurer la fin des calculs de gradients
    }

    // Zero out accumulated gradients (d_weights_sum reste dense)
    void zero_accumulated_gradients() {
         if (input_size == 0 && layer_size == 0) return; // Couche vide
         if (d_biases_sum.data() != nullptr) Kokkos::deep_copy(d_biases_sum, 0.0);
         // d_weights_sum est dense
         if (d_weights_sum.data() != nullptr) Kokkos::deep_copy(d_weights_sum, 0.0);
    }

    // Afficher les infos de la couche (adapté pour sparse_weights)
    void show() const {
        if (input_size == 0 && activation == nullptr) {
            std::cout << "Input Layer (size " << layer_size << ")" << std::endl;
            return;
        }
        std::cout << "Layer (" << input_size << " -> " << layer_size << "):";
        if (activation) {
             try {
                 std::string act_name = typeid(*activation).name();
                 size_t last_colon = act_name.find_last_of("::");
                 if(last_colon != std::string::npos) act_name = act_name.substr(last_colon+1);
                 std::cout << " Activation: " << act_name;
             } catch (const std::exception&) {
                 std::cout << " Activation: [Unknown]";
             }
         } else {
             std::cout << " Activation: Linear (Implicit)";
         }
         std::cout << std::endl;

        // Afficher infos sur sparse_weights
         std::cout << "  Sparse Weights: "
                   << sparse_weights.numRows() << "x" << sparse_weights.numCols()
                   << ", NNZ (Non-Zeros): " << sparse_weights.nnz()
                   << ", Sparsity: " << std::fixed << std::setprecision(4)
                   << (sparse_weights.numRows()*sparse_weights.numCols() > 0 ?
                       100.0 * static_cast<double>(sparse_weights.nnz()) / (sparse_weights.numRows() * sparse_weights.numCols()) : 0.0)
                   << "%" << std::endl;

        // Optionnel: Afficher un petit bout converti en dense (peut être lent)
        /*
        const int max_print_rows = 3;
        const int max_print_cols = 3;
        if (sparse_weights.numRows() > 0 && sparse_weights.numCols() > 0 && sparse_weights.nnz() > 0) {
            try {
                 View2D h_weights_dense = sparse_to_dense(sparse_weights); // Conversion potentiellement coûteuse
                 auto h_weights_mirror = Kokkos::create_mirror_view(h_weights_dense);
                 Kokkos::deep_copy(h_weights_mirror, h_weights_dense);
                 Kokkos::fence();
                 std::cout << "  Weights (Dense Sample):" << std::endl;
                 for(int i=0; i< std::min((int)h_weights_mirror.extent(0), max_print_rows) ; ++i) {
                      std::cout << "    [";
                      for(int j=0; j< std::min((int)h_weights_mirror.extent(1), max_print_cols); ++j) {
                          std::cout << std::fixed << std::setprecision(3) << h_weights_mirror(i,j) << " ";
                      }
                      if (h_weights_mirror.extent(1) > max_print_cols) std::cout << "...";
                      std::cout << "]" << std::endl;
                 }
                  if (h_weights_mirror.extent(0) > max_print_rows) std::cout << "    ..." << std::endl;
            } catch (const std::exception& e) {
                std::cout << "  Weights (Dense Sample): Error during conversion/display - " << e.what() << std::endl;
            }
        }
        */

        // Afficher les biais (inchangé)
        if (biases.data() != nullptr && biases.extent(0) > 0) {
            auto h_biases = Kokkos::create_mirror_view(biases);
            Kokkos::deep_copy(h_biases, biases);
            Kokkos::fence();
            const int max_print_biases = 10;
            std::cout << "  Biases (" << biases.extent(0) << ", showing sample): [";
            for(int i=0; i< std::min((int)biases.extent(0), max_print_biases); ++i) {
                std::cout << std::fixed << std::setprecision(3) << h_biases(i) << " ";
            }
            if (biases.extent(0) > max_print_biases) std::cout << "...";
            std::cout << "]" << std::endl;
         } else {
             std::cout << "  Biases: N/A" << std::endl;
         }
    }

    // Gérer move/copy: Désactiver la copie car CrsMatrix n'est pas trivialement copiable.
    // Le déplacement par défaut peut fonctionner mais soyons prudents et désactivons-le aussi
    // pour l'instant pour éviter des problèmes potentiels avec les pointeurs dans la map Adam.
    // On utilisera emplace_back dans Network.
    Layer(Layer&& other) = delete; // default;
    Layer& operator=(Layer&& other) = delete; // default;
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    virtual ~Layer() = default;
};


// --- Classe InputLayer (hérite de Layer, pas de changement majeur nécessaire) ---
class InputLayer : public Layer {
public:
    InputLayer(int _layer_size) : Layer(_layer_size) {} // Appelle Layer(0, _layer_size, nullptr)

    void set_input(const View1D& input_data) {
         if (input_data.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Input data size mismatch for InputLayer.");
         }
         if (a.data() == nullptr || a.extent(0) != static_cast<size_t>(layer_size)) {
              throw std::runtime_error("InputLayer 'a' view is not correctly initialized.");
         }
         Kokkos::deep_copy(a, input_data);
     }

     // Overrides pour clarifier qu'elles ne font rien
     void forward(const View1D& /*prev_layer_a*/) override {}
     void compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) override {}
};


// --- Classe OutputLayer (hérite de Layer, adaptation de compute_gradients) ---
class OutputLayer : public Layer {
public:
    OutputLayer(int _input_size, int _layer_size, std::unique_ptr<Activation> act_func) :
        Layer(_input_size, _layer_size, std::move(act_func)) {}

    // Calcul des gradients spécifique à la couche de sortie (cible y)
    // Accumule directement dans d_weights_sum (dense)
    void compute_gradients(const View1D& target, const View1D& prev_layer_a) {
        // Vérifications
         if (target.extent(0) != static_cast<size_t>(layer_size)) {
             throw std::runtime_error("Target size mismatch for OutputLayer gradient.");
         }
          if (prev_layer_a.extent(0) != static_cast<size_t>(input_size)) {
             throw std::runtime_error("prev_layer_a size mismatch in OutputLayer gradient.");
         }
          if (delta.extent(0) != static_cast<size_t>(layer_size) ||
              a.extent(0) != static_cast<size_t>(layer_size) ||
              tmp_deriv.extent(0) != static_cast<size_t>(layer_size)) {
              throw std::runtime_error("Internal dimension mismatch in OutputLayer gradient.");
          }
          if (d_weights_sum.extent(0) != static_cast<size_t>(layer_size) || d_weights_sum.extent(1) != static_cast<size_t>(input_size)) {
               throw std::runtime_error("OutputLayer d_weights_sum dimension mismatch.");
           }

        // 1. Calculer f'(z_L) -> tmp_deriv (inchangé)
        if (activation) {
            activation->apply_derivative(z, tmp_deriv);
        } else {
            Kokkos::deep_copy(tmp_deriv, 1.0f); // Linear
        }

        // 2. Calculer delta_L = (a_L - y) .* f'(z_L) (inchangé)
        Kokkos::parallel_for("compute_output_delta", layer_size, KOKKOS_LAMBDA(int i) {
            delta(i) = (a(i) - target(i)) * tmp_deriv(i);
        });

       // 3. Calculer gradient biais instantané -> d_biases (inchangé)
        if (d_biases.extent(0) != delta.extent(0)) throw std::runtime_error("OutputLayer d_biases/delta mismatch.");
        Kokkos::deep_copy(d_biases, delta);

        // 4. Calculer gradient poids instantané (produit externe) ET l'accumuler dans d_weights_sum (dense)
        Kokkos::parallel_for("compute_accumulate_output_d_weights",
             Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {layer_size, input_size}),
             KOKKOS_LAMBDA(const int i, const int j) {
                 real dw_ij = delta(i) * prev_layer_a(j);
                 Kokkos::atomic_add(&d_weights_sum(i, j), dw_ij);
        });

        // 5. Accumuler gradient biais (inchangé)
         if (d_biases_sum.extent(0) != d_biases.extent(0)) {
             throw std::runtime_error("OutputLayer accumulated d_biases mismatch.");
         }
         Kokkos::parallel_for("accumulate_output_bias_gradients", layer_size, KOKKOS_LAMBDA(int i) {
             Kokkos::atomic_add(&d_biases_sum(i), d_biases(i));
        });
         Kokkos::fence();
    }

    // Override pour éviter l'appel accidentel à la version couche cachée
     void compute_gradients(const Layer& /*next_layer*/, const View1D& /*prev_layer_a*/) override {
         throw std::logic_error("OutputLayer::compute_gradients should be called with target, not next_layer.");
     }
};


// --- Classes Dataset / BatchHandler (Placeholders inchangés) ---
class Dataset {};
class BatchHandler {};


// --- Classe Network (adaptée pour gérer les Layers modifiés) ---
class Network {
public:
    std::map<int, int> layer_sizes_map;
    InputLayer input_layer;
    std::vector<Layer> hidden_layers; // Utilise emplace_back pour construire Layer
    OutputLayer output_layer;
    std::unique_ptr<Optimizer> optimizer;
    Dataset dataset; // Placeholder
    BatchHandler batch_handler; // Placeholder

    // Constructeur (utilise emplace_back pour hidden_layers)
    Network(const std::map<int, int>& _layer_sizes_map,
            const std::vector<std::string>& activation_types,
            std::unique_ptr<Optimizer> opt)
        : layer_sizes_map(_layer_sizes_map),
          input_layer(get_size(0)),
          // Crée OutputLayer directement (avant hidden pour avoir les tailles)
          output_layer(get_size(num_layers() - 2), get_size(num_layers() - 1), create_activation(activation_types.back())),
          optimizer(std::move(opt))
    {
        if (layer_sizes_map.size() < 2) {
            throw std::runtime_error("Network must have at least input and output layers.");
        }
        if (activation_types.size() != layer_sizes_map.size() - 1) {
             throw std::runtime_error("Mismatch between activation types and number of layers.");
        }
        if (!optimizer) {
             throw std::runtime_error("Optimizer must be provided.");
        }

        // Créer les couches cachées avec emplace_back
        int num_hidden_layers = layer_sizes_map.size() - 2;
        hidden_layers.reserve(num_hidden_layers);
        for (int i = 0; i < num_hidden_layers; ++i) {
            int current_layer_idx_in_map = i + 1;
            int in_size = get_size(current_layer_idx_in_map - 1);
            int out_size = get_size(current_layer_idx_in_map);
            std::string act_type = activation_types[i];
            // Construit l'objet Layer directement dans le vecteur
            hidden_layers.emplace_back(in_size, out_size, create_activation(act_type));
        }
    }

    // Fonctions get_size, num_layers (inchangées)
    int get_size(int layer_index) const { /* ... comme avant ... */
        try { return layer_sizes_map.at(layer_index); }
        catch (const std::out_of_range&) { throw std::out_of_range("Layer index " + std::to_string(layer_index) + " not found."); }
    }
    int num_layers() const { return layer_sizes_map.size(); }


    // Forward pass (inchangé logiquement)
    View1D forward(const View1D& input_data) {
        input_layer.set_input(input_data);
        const View1D* current_a = &input_layer.a;
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.forward(*current_a);
            current_a = &hidden_layer.a;
        }
        output_layer.forward(*current_a);
        return output_layer.a;
    }

    // Backward pass (inchangé logiquement)
    void backward(const View1D& target) {
        const View1D& prev_a_output = hidden_layers.empty() ? input_layer.a : hidden_layers.back().a;
        output_layer.compute_gradients(target, prev_a_output); // Utilise la version spécifique de OutputLayer

        Layer* next_layer_ptr = &output_layer;
        for (int i = hidden_layers.size() - 1; i >= 0; --i) {
            Layer& current_layer = hidden_layers[i];
            const View1D& prev_a = (i == 0) ? input_layer.a : hidden_layers[i - 1].a;
            current_layer.compute_gradients(*next_layer_ptr, prev_a); // Utilise la version de Layer/OutputLayer
            next_layer_ptr = &current_layer;
        }
    }

     // Update (passe sparse_weights et d_weights_sum dense à l'optimizer)
    void update(int batch_size) {
        if (!optimizer) throw std::runtime_error("Optimizer not set.");
        if (batch_size <= 0) throw std::runtime_error("Batch size must be positive.");

        // Mettre à jour couches cachées
        for (Layer& layer : hidden_layers) {
            if (layer.input_size > 0) { // Seulement si elle a des poids
                // Passe sparse_weights et le d_weights_sum (dense)
                optimizer->update(layer.sparse_weights, layer.biases,
                                  layer.d_weights_sum, layer.d_biases_sum,
                                  batch_size);
            }
        }
        // Mettre à jour couche de sortie
         if (output_layer.input_size > 0) {
            optimizer->update(output_layer.sparse_weights, output_layer.biases,
                              output_layer.d_weights_sum, output_layer.d_biases_sum,
                              batch_size);
         }
    }

    // Zero accumulated gradients (inchangé logiquement)
    void zero_accumulated_gradients() {
        for (Layer& hidden_layer : hidden_layers) {
            hidden_layer.zero_accumulated_gradients();
        }
        output_layer.zero_accumulated_gradients();
    }

    // Calculate Cost (inchangé)
    real calculate_cost(const View1D& prediction, const View1D& target) {
         // ... comme avant ...
        int output_size = prediction.extent_int(0);
        if (target.extent_int(0) != output_size) {
             throw std::runtime_error("Prediction and target size mismatch for cost.");
        }
        if (output_size == 0) return 0.0;
        real squared_error_sum = 0.0;
        Kokkos::parallel_reduce("compute_cost", output_size, KOKKOS_LAMBDA (int i, real& lsum) {
            real diff = prediction(i) - target(i);
            lsum += diff * diff;
        }, squared_error_sum);
        Kokkos::fence();
        return 0.5 * squared_error_sum;
     }


    // --- Optimizer Management (inchangé) ---
    void set_optimizer(std::unique_ptr<Optimizer> opt) { /* ... */ if (!opt) throw std::runtime_error("Cannot set null optimizer."); optimizer = std::move(opt); }
    Optimizer* get_optimizer() const { return optimizer.get(); }
    void set_learning_rate(real lr) { /* ... */ if (!optimizer) throw std::runtime_error("Optimizer not set."); optimizer->set_learning_rate(lr); }
    real get_learning_rate() const { /* ... */ if (!optimizer) return 0.0; return optimizer->get_learning_rate(); }

    // --- Display Network Info (utilise Layer::show adapté) ---
    void show() const {
         std::cout << "\n--- Network Structure (Sparse Weights) ---" << std::endl;
         input_layer.show();
         int i = 1;
         for (const auto& layer : hidden_layers) { // const ref est ok car show() est const
              std::cout << "\n--- Hidden Layer " << i++ << " ---" << std::endl;
              layer.show();
         }
         std::cout << "\n--- Output Layer ---" << std::endl;
         output_layer.show();
         std::cout << "\n--- Optimizer Info ---" << std::endl;
         if(optimizer) { std::cout << "  " << optimizer->get_info() << std::endl; }
         else { std::cout << "  Optimizer: Not set" << std::endl; }
         std::cout << "------------------------------------------" << std::endl;
    }

    // --- Move/Copy Semantics ---
    // Désactiver explicitement pour éviter problèmes avec CrsMatrix/Optimizer state map
    Network(Network&&) = delete;
    Network& operator=(Network&&) = delete;
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    virtual ~Network() = default;
};


// --- Fonctions d'entraînement (xor_train, sine_train, linear_sep_train) ---
// Elles devraient fonctionner sans modification majeure car l'interface de Network
// (forward, backward, update, zero_gradients, calculate_cost) est restée la même.
// La différence sera interne (utilisation de SpMV, conversions dans l'optimizer).
// ... (Les fonctions xor_train, sine_train, linear_sep_train restent identiques à l'original) ...
void xor_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- XOR Training Example (Sparse Weights Storage) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;
    std::cout << "Note: Weight updates involve inefficient Dense<->Sparse conversions." << std::endl;

    std::map<int, int> sizes;
    sizes[0] = 2; sizes[1] = 8; sizes[2] = 4; sizes[3] = 1;
    std::vector<std::string> activations = {"relu", "relu", "sigmoid"};
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
    // dnn.show(); // Affiche maintenant les infos de sparsité

    using HostView2D = Kokkos::View<real**, Kokkos::HostSpace>;
    HostView2D h_xor_inputs("h_xor_inputs", 4, 2);
    HostView2D h_xor_outputs("h_xor_outputs", 4, 1);
    // ... (remplissage données XOR comme avant) ...
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
        for (int i = 0; i < 4; ++i) {
            auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
            auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
            dnn.forward(input_subview);
            dnn.backward(target_subview); // Accumule gradients (dans vues denses)
        }
        dnn.update(batch_size); // Met à jour poids (avec conversions internes)

        // Calcul coût (optionnel, comme avant)
         if ((epoch + 1) % (epochs / 20) == 0 || epoch == 0 || epoch == epochs -1) {
             real current_total_cost = 0.0;
             for (int i = 0; i < 4; ++i) {
                 auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
                 auto target_subview = Kokkos::subview(xor_outputs, i, Kokkos::ALL());
                 View1D prediction = dnn.forward(input_subview);
                 current_total_cost += dnn.calculate_cost(prediction, target_subview);
             }
             real avg_cost = current_total_cost / 4.0;
             std::cout << "Epoch: " << std::setw(5) << epoch + 1
                       << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost << std::endl;
             if (avg_cost < 1e-4 && epoch > 10) { // Early stop possible
                  std::cout << "Convergence likely reached at epoch " << epoch + 1 << std::endl;
                  break;
             }
         }
    }
    std::cout << "-- TRAINING END --" << std::endl;

    // Afficher prédictions finales (inchangé)
    std::cout << "\n-- FINAL PREDICTIONS --" << std::endl;
    View1D prediction_result("prediction_result", 1);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    for (int i = 0; i < 4; ++i) {
        // ... (calcul et affichage comme avant) ...
        auto input_subview = Kokkos::subview(xor_inputs, i, Kokkos::ALL());
        View1D prediction = dnn.forward(input_subview);
        Kokkos::deep_copy(prediction_result, prediction);
        Kokkos::deep_copy(h_prediction_result, prediction_result); Kokkos::fence();
         std::cout << "Input: [" << h_xor_inputs(i,0) << "," << h_xor_inputs(i,1) << "] "
                   << "Target: " << h_xor_outputs(i,0) << " "
                   << "Prediction: " << std::fixed << std::setprecision(4) << h_prediction_result(0)
                   << " (Rounded: " << std::round(h_prediction_result(0)) << ")" << std::endl;
    }
    // dnn.show(); // Pour voir la structure finale et la sparsité (si elle a changé)
}

// ... Les fonctions sine_train et linear_sep_train restent également identiques en termes d'appels à Network ...
//     Il suffit de copier/coller leur corps ici si nécessaire.
//     Ajoutons juste un message indiquant l'utilisation du stockage épars.

void sine_train(const std::string& optimizer_choice = "adam") {
     std::cout << "\n--- Sine Function Approximation Training Example (Sparse Weights Storage) ---" << std::endl;
     std::cout << "Using Optimizer: " << optimizer_choice << std::endl;
     std::cout << "Note: Weight updates involve inefficient Dense<->Sparse conversions." << std::endl;
    // ... (Reste du code de sine_train identique) ...
    std::map<int, int> sizes;
    sizes[0] = 1; sizes[1] = 64; sizes[2] = 64; sizes[3] = 1;
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
    HostView2D h_inputs("h_sine_inputs", num_samples, 1);
    HostView2D h_outputs("h_sine_outputs", num_samples, 1);
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<real> distrib(-M_PI, M_PI);
    for(int i=0; i < num_samples; ++i) {
        real x = distrib(gen);
        h_inputs(i, 0) = x;
        h_outputs(i, 0) = std::sin(x);
    }
    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << ", Samples: " << num_samples << std::endl;
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
             std::cout << "Epoch: " << std::setw(5) << epoch + 1
                       << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost << std::endl;
        }
         if (avg_cost < 1e-3 && epoch > 50) {
             std::cout << "Good convergence likely reached at epoch " << epoch + 1 << std::endl;
             // break;
         }
    }
    std::cout << "-- TRAINING END --" << std::endl;
    // ... (Test prédictions finales comme avant) ...
     std::cout << "\n-- FINAL PREDICTIONS (Sample) --" << std::endl;
    View1D prediction_result("prediction_result_sine", 1);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0;
    int num_test_samples = std::min(num_samples, 10);
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
    std::cout << "Final Average Cost (on first " << num_test_samples << " samples): "
              << std::fixed << std::setprecision(8) << final_total_cost / num_test_samples << std::endl;
}


void linear_sep_train(const std::string& optimizer_choice = "adam") {
    std::cout << "\n--- Linear Separation Training Example (Sparse Weights Storage) ---" << std::endl;
    std::cout << "Using Optimizer: " << optimizer_choice << std::endl;
    std::cout << "Note: Weight updates involve inefficient Dense<->Sparse conversions (though less impactful here)." << std::endl;
    // ... (Reste du code de linear_sep_train identique) ...
    std::map<int, int> sizes;
    sizes[0] = 2; sizes[1] = 1; // Pas de couche cachée
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
    HostView2D h_inputs("h_linear_inputs", num_samples, 2);
    HostView2D h_outputs("h_linear_outputs", num_samples, 1);
    std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count() + 123);
    std::uniform_real_distribution<real> distrib(-1.0, 1.0);
    real margin = 0.1; int count_class0 = 0; int count_class1 = 0;
    for(int i=0; i < num_samples; ++i) { /* ... (Génération données comme avant) ... */
        real x = distrib(gen); real y = distrib(gen);
        h_inputs(i, 0) = x; h_inputs(i, 1) = y;
        if (y < x - margin) { h_outputs(i, 0) = 0.0; count_class0++; }
        else if (y > x + margin) { h_outputs(i, 0) = 1.0; count_class1++; }
        else { i--; continue; }
    }
     std::cout << "Generated " << count_class0 << " Class 0 and " << count_class1 << " Class 1 samples." << std::endl;

    auto train_inputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_inputs);
    auto train_outputs = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_outputs);

    std::cout << "-- TRAINING START --" << std::endl;
    std::cout << dnn.get_optimizer()->get_info() << std::endl;
    std::cout << "Epochs: " << epochs << ", Batch Size: " << batch_size << ", Samples: " << num_samples << std::endl;
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) { /* ... (Boucle training comme avant) ... */
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
              std::cout << "Epoch: " << std::setw(4) << epoch + 1
                        << ", Avg Cost: " << std::fixed << std::setprecision(8) << avg_cost << std::endl;
         }
          if (avg_cost < 1e-3 && epoch > 20) {
              std::cout << "Convergence likely reached at epoch " << epoch + 1 << std::endl;
              break;
          }
    }
    std::cout << "-- TRAINING END --" << std::endl;
    // ... (Test final et accuracy comme avant) ...
    std::cout << "\n-- FINAL PREDICTIONS & ACCURACY --" << std::endl;
    View1D prediction_result("prediction_result_linear", 1);
    auto h_prediction_result = Kokkos::create_mirror_view(prediction_result);
    real final_total_cost = 0.0; int correct_predictions = 0;
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
        if (i < 10) { /* ... (Affichage exemples) ... */
            std::cout << "Input: [" << std::fixed << std::setprecision(2) << h_inputs(i,0) << "," << h_inputs(i,1) << "] "
                      << "Target: " << target_class << " "
                      << "Pred: " << std::fixed << std::setprecision(3) << predicted_value
                      << " (Rounded: " << predicted_class << ")"
                      << (predicted_class == target_class ? "" : " <-- WRONG") << std::endl;
        }
    }
    real accuracy = static_cast<real>(correct_predictions) / num_samples;
    std::cout << "Final Average Cost: " << std::fixed << std::setprecision(8) << final_total_cost / num_samples << std::endl;
    std::cout << "Final Accuracy: " << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%" << std::endl;
}


// --- Main (inchangé) ---
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        try {
            std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;

            xor_train("adam");
            std::cout << "\n---------------------------\n" << std::endl;
//            sine_train("adam"); // Peut être lent avec les conversions
            std::cout << "\n---------------------------\n" << std::endl;
//            linear_sep_train("adam"); // Moins affecté par les conversions car petite matrice

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            Kokkos::finalize(); // Assurer la finalisation en cas d'erreur
            return 1;
        } catch (...) {
            std::cerr << "An unknown error occurred." << std::endl;
            Kokkos::finalize(); // Assurer la finalisation
            return 1;
        }
    }
    Kokkos::finalize();
    return 0;
}
