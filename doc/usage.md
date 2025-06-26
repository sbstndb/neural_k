# Guide d'utilisation

## Création d'un réseau de neurones

### 1. Configuration de base

```cpp
#include "network.hpp"
#include "optimizers.hpp"

// Définir l'architecture
std::map<int, int> sizes;
sizes[0] = 2;    // Couche d'entrée : 2 inputs
sizes[1] = 10;   // Couche cachée : 10 neurones  
sizes[2] = 1;    // Couche de sortie : 1 output

// Définir les activations (une par couche non-input)
std::vector<std::string> activations = {"relu", "sigmoid"};

// Créer l'optimiseur
auto optimizer = std::make_unique<Adam>(0.001); // learning rate = 0.001

// Construire le réseau
Network network(sizes, activations, std::move(optimizer));
```

### 2. Fonctions d'activation disponibles

| Activation | Description | Usage typique |
|------------|-------------|---------------|
| `"linear"` | Identité (f(x) = x) | Régression, couche finale |
| `"relu"` | ReLU (f(x) = max(0,x)) | Couches cachées |
| `"sigmoid"` | Sigmoïde (f(x) = 1/(1+e^-x)) | Classification binaire |
| `"tanh"` | Tangente hyperbolique | Couches cachées |

### 3. Optimiseurs disponibles

#### SGD (Stochastic Gradient Descent)
```cpp
auto sgd = std::make_unique<SGD>(0.1);  // learning rate
```

#### Adam (Adaptive Moment Estimation)
```cpp
auto adam = std::make_unique<Adam>(
    0.001,  // learning rate
    0.9,    // beta1 (moment decay)
    0.999,  // beta2 (variance decay) 
    1e-8    // epsilon (numerical stability)
);
```

## Entraînement d'un réseau

### Processus d'entraînement standard

```cpp
// 1. Préparer les données
View1D input("input", input_size);
View1D target("target", output_size);
// ... remplir les données ...

// 2. Boucle d'entraînement
int epochs = 1000;
int batch_size = 32;

for (int epoch = 0; epoch < epochs; ++epoch) {
    network.zero_accumulated_gradients();
    
    // Forward + backward pour chaque échantillon du batch
    for (int i = 0; i < batch_size; ++i) {
        // Récupérer input et target pour l'échantillon i
        View1D prediction = network.forward(input);
        network.backward(target);
    }
    
    // Mise à jour des poids
    network.update(batch_size);
    
    // Optionnel : calculer et afficher le coût
    View1D final_pred = network.forward(input);
    real cost = network.calculate_cost(final_pred, target);
    std::cout << "Epoch " << epoch << ", Cost: " << cost << std::endl;
}
```

### Gestion des données Kokkos

#### Création de vues
```cpp
// Sur host (CPU)
Kokkos::View<real**, Kokkos::HostSpace> h_data("host_data", num_samples, input_dim);

// Remplir les données
for (int i = 0; i < num_samples; ++i) {
    for (int j = 0; j < input_dim; ++j) {
        h_data(i, j) = /* valeur */;
    }
}

// Transférer vers device (GPU si disponible)
auto device_data = Kokkos::create_mirror_view_and_copy(
    Kokkos::DefaultExecutionSpace(), h_data
);
```

#### Accès aux sous-vues
```cpp
// Extraire un échantillon
auto sample = Kokkos::subview(device_data, i, Kokkos::ALL());
View1D prediction = network.forward(sample);
```

## Évaluation et prédiction

### Prédiction simple
```cpp
View1D input("test_input", input_size);
// ... remplir input ...

View1D prediction = network.forward(input);

// Récupérer le résultat sur host
auto h_pred = Kokkos::create_mirror_view(prediction);
Kokkos::deep_copy(h_pred, prediction);
Kokkos::fence();

std::cout << "Prédiction: " << h_pred(0) << std::endl;
```

### Évaluation sur dataset
```cpp
real total_cost = 0.0;
int num_test_samples = /* ... */;

for (int i = 0; i < num_test_samples; ++i) {
    auto test_input = Kokkos::subview(test_data, i, Kokkos::ALL());
    auto test_target = Kokkos::subview(test_labels, i, Kokkos::ALL());
    
    View1D pred = network.forward(test_input);
    total_cost += network.calculate_cost(pred, test_target);
}

real avg_cost = total_cost / num_test_samples;
std::cout << "Coût moyen test: " << avg_cost << std::endl;
```

## Configuration avancée

### Modification des hyperparamètres
```cpp
// Changer le learning rate pendant l'entraînement
network.set_learning_rate(0.0001);

// Ou remplacer complètement l'optimiseur
auto new_optimizer = std::make_unique<SGD>(0.01);
network.set_optimizer(std::move(new_optimizer));
```

### Inspection du réseau
```cpp
// Afficher la structure du réseau
network.show();

// Accéder aux informations de l'optimiseur
std::cout << network.get_optimizer()->get_info() << std::endl;
```

## Gestion d'erreurs communes

### Problèmes de dimensions
```cpp
// ❌ Erreur : tailles incompatibles
std::map<int, int> sizes;
sizes[0] = 2;
sizes[1] = 10;
std::vector<std::string> activations = {"relu", "sigmoid", "tanh"}; // Trop d'activations !

// ✅ Correct : nb_activations = nb_couches - 1
std::vector<std::string> activations = {"relu"}; // Une seule activation pour 2→10→1
```

### Problèmes de données
```cpp
// ❌ Dimension d'entrée incorrecte
View1D wrong_input("input", 5); // Réseau attend 2 inputs
View1D pred = network.forward(wrong_input); // Erreur !

// ✅ Dimension correcte
View1D correct_input("input", 2);
View1D pred = network.forward(correct_input); // OK
```

### Optimisation mémoire
```cpp
// Réutiliser les vues quand possible
View1D reusable_input("input", input_size);
View1D reusable_target("target", output_size);

for (int sample = 0; sample < num_samples; ++sample) {
    // Copier les données dans les vues réutilisables
    // plutôt que créer de nouvelles vues à chaque fois
}
``` 