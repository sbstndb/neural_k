# Composants techniques

## Activations (`activations.*`)

### Interface commune

```cpp
class Activation {
public:
    virtual void apply(const View1D& input, View1D& output) const = 0;
    virtual void apply_derivative(const View1D& input_z, View1D& output_deriv) const = 0;
};
```

### Implémentations

#### RELU
```cpp
// Forward: f(x) = max(0, x)
Kokkos::parallel_for("relu_apply", size, KOKKOS_LAMBDA(const int i) {
    output(i) = (input(i) > 0.0f) ? input(i) : 0.0f;
});

// Derivative: f'(x) = x > 0 ? 1 : 0
Kokkos::parallel_for("relu_deriv", size, KOKKOS_LAMBDA(const int i) {
    output_deriv(i) = (input_z(i) > 0.0f) ? 1.0f : 0.0f;
});
```

#### SIGMOID
```cpp
// Forward: f(x) = 1 / (1 + e^(-x))
KOKKOS_INLINE_FUNCTION real scalar_sigmoid(real x) const {
    x = Kokkos::max(-30.0f, Kokkos::min(30.0f, x)); // Éviter overflow
    return 1.0f / (1.0f + Kokkos::exp(-x));
}

// Derivative: f'(x) = f(x) * (1 - f(x))
real s = scalar_sigmoid(input_z(i));
output_deriv(i) = s * (1.0f - s);
```

## Optimiseurs (`optimizers.*`)

### Classe de base

```cpp
class Optimizer {
public:
    real learning_rate;
    
    virtual void update(SparseMatrixType& weights, View1D biases,
                        const SparseMatrixType& accumulated_d_weights, 
                        const View1D& accumulated_d_biases,
                        int batch_size) = 0;
};
```

### SGD

**Algorithme simple** :
```
w = w - (learning_rate / batch_size) * gradient
```

**Implémentation parallèle** :
```cpp
const real scale = learning_rate / static_cast<real>(batch_size);

// Mise à jour des poids (sparse)
Kokkos::parallel_for("sgd_update_weights_sparse", nnz, KOKKOS_LAMBDA(const int k) {
    w_vals(k) -= scale * dw_vals(k);
});

// Mise à jour des biais (dense)
Kokkos::parallel_for("sgd_update_biases", layer_size, KOKKOS_LAMBDA(int i) {
    biases(i) -= scale * accumulated_d_biases(i);
});
```

### Adam

**Innovation : Moments denses pour matrices creuses**

#### Structure de données
```cpp
struct ParameterState {
    View1D m_1d, v_1d;           // Pour biais (1D)
    Kokkos::View<real**> m_2d, v_2d;  // Pour poids (2D dense)
    long long t = 0;             // Timestep
};
```

#### Algorithme
```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
w_t = w_{t-1} - α * m̂_t / (√v̂_t + ε)
```

#### Implémentation clé
```cpp
// Itération par structure sparse, mais stockage dense des moments
Kokkos::parallel_for("adam_update_weights_sparse", layer_size, 
KOKKOS_LAMBDA (const int i) {
    const auto row_start = graph.row_map(i);
    const auto row_end = graph.row_map(i+1);
    for (auto k = row_start; k < row_end; ++k) {
        const int j = graph.entries(k);
        
        // Gradient pour cet élément sparse
        real grad = scale * dw_vals(k);
        
        // Mise à jour moments denses à la position (i,j)
        m(i, j) = b1 * m(i, j) + (1.0f - b1) * grad;
        v(i, j) = b2 * v(i, j) + (1.0f - b2) * grad * grad;
        
        // Correction de biais et mise à jour
        real m_hat = m(i, j) * bias_correction1;
        real v_hat = v(i, j) * bias_correction2;
        w_vals(k) -= lr * m_hat / (Kokkos::sqrt(v_hat) + eps);
    }
});
```

## Couches (`layers.*`)

### Structure des données

```cpp
class Layer {
    // Paramètres
    SparseMatrixType weights;     // [layer_size x input_size]
    View1D biases;               // [layer_size]
    
    // États forward
    View1D z;                    // Pré-activation
    View1D a;                    // Post-activation
    
    // États backward  
    View1D delta;                // Erreur propagée
    View1D d_biases;             // Gradient instantané biais
    SparseMatrixType d_weights;  // Gradient instantané poids
    
    // Accumulation
    View1D d_biases_sum;         // Somme gradients biais
    SparseMatrixType d_weights_sum; // Somme gradients poids
};
```

### Forward pass

```cpp
void Layer::forward(const View1D& prev_layer_a) {
    // 1. Produit matrice-vecteur sparse
    KokkosSparse::spmv("N", 1.0, weights, prev_layer_a, 0.0, z);
    
    // 2. Ajout des biais
    Kokkos::parallel_for("add_biases", layer_size, KOKKOS_LAMBDA(int i) {
        z(i) += biases(i);
    });
    
    // 3. Application de l'activation
    if (activation) {
        activation->apply(z, a);
    } else {
        Kokkos::deep_copy(a, z);
    }
}
```

### Backward pass

#### Couche cachée
```cpp
void Layer::compute_gradients(const Layer& next_layer, const View1D& prev_layer_a) {
    // 1. Calcul dérivée activation
    activation->apply_derivative(z, tmp_deriv);
    
    // 2. Propagation erreur (transpose SpMV)
    KokkosSparse::spmv("T", 1.0, next_layer.weights, next_layer.delta, 0.0, delta_prop);
    
    // 3. Produit de Hadamard
    Kokkos::parallel_for("hadamard_delta_deriv", layer_size, KOKKOS_LAMBDA(int i) {
        delta(i) = delta_prop(i) * tmp_deriv(i);
    });
    
    // 4. Calcul gradients poids (produit extérieur sparse)
    Kokkos::parallel_for("compute_d_weights_sparse", layer_size, KOKKOS_LAMBDA(const int i) {
        const auto row_start = graph.row_map(i);
        const auto row_end = graph.row_map(i+1);
        for (auto k = row_start; k < row_end; ++k) {
            const int j = graph.entries(k);
            dw_vals(k) = delta(i) * prev_layer_a(j);
        }
    });
}
```

#### Couche de sortie
```cpp
void OutputLayer::compute_gradients(const View1D& target, const View1D& prev_layer_a) {
    // 1. Dérivée activation
    activation->apply_derivative(z, tmp_deriv);
    
    // 2. Delta de sortie : (prédiction - cible) * f'(z)
    Kokkos::parallel_for("compute_output_delta", layer_size, KOKKOS_LAMBDA(int i) {
        delta(i) = (a(i) - target(i)) * tmp_deriv(i);
    });
    
    // 3. Même calcul de gradients que couche cachée
    // ...
}
```

## Réseau (`network.*`)

### Orchestration des passes

```cpp
View1D Network::forward(const View1D& input_data) {
    // 1. Initialiser couche d'entrée
    input_layer.set_input(input_data);
    
    // 2. Propager à travers couches cachées
    const View1D* current_a = &input_layer.a;
    for (Layer& hidden_layer : hidden_layers) {
        hidden_layer.forward(*current_a);
        current_a = &hidden_layer.a;
    }
    
    // 3. Couche de sortie
    output_layer.forward(*current_a);
    return output_layer.a;
}

void Network::backward(const View1D& target) {
    // 1. Commencer par la sortie
    const View1D& prev_a_output = hidden_layers.empty() ? 
        input_layer.a : hidden_layers.back().a;
    output_layer.compute_gradients(target, prev_a_output);
    
    // 2. Propager vers l'arrière
    Layer* next_layer_ptr = &output_layer;
    for (int i = hidden_layers.size() - 1; i >= 0; --i) {
        Layer& current_layer = hidden_layers[i];
        const View1D& prev_a = (i == 0) ? 
            input_layer.a : hidden_layers[i - 1].a;
        current_layer.compute_gradients(*next_layer_ptr, prev_a);
        next_layer_ptr = &current_layer;
    }
}
```

### Gestion mémoire

**Matrices creuses** : Structure CRS (Compressed Row Storage)
- `row_map` : Offsets des débuts de lignes
- `entries` : Indices des colonnes  
- `values` : Valeurs non-nulles

**Initialisation dense** :
```cpp
// Tous les éléments sont présents initialement
size_t nnz = layer_size * input_size;
for (int i = 0; i < layer_size; ++i) {
    for (int j = 0; j < input_size; ++j) {
        entries[i * input_size + j] = j;
    }
}
```

**Avantages** :
- Interface unifiée sparse/dense
- Optimisations KokkosSparse
- Extensibilité future pour vraie sparsité 