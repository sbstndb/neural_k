# Architecture du système

Ce document explique l'architecture globale de Neural K. Vous y découvrirez comment les différents modules s'enchaînent, depuis la définition des types jusqu'au point d'entrée.

## Vue d'ensemble

Neural K utilise une architecture modulaire basée sur Kokkos pour maximiser les performances sur différentes plateformes (CPU, GPU, etc.).

## Diagramme de dépendances

```
types.hpp (Types Kokkos)
    ↓
activations.hpp/.cpp (Fonctions d'activation)
    ↓
optimizers.hpp/.cpp (SGD, Adam)
    ↓  
layers.hpp/.cpp (Layer, InputLayer, OutputLayer)
    ↓
network.hpp/.cpp (Network principal)
    ↓
training.hpp/.cpp (Fonctions d'entraînement)
    ↓
main.cpp (Point d'entrée)
```

## Composants principaux

### 1. Types fondamentaux (`types.hpp`)

```cpp
using real = float;                    // Type numérique principal
using View1D = Kokkos::View<real*>;   // Vecteurs Kokkos
using SparseMatrixType = KokkosSparse::CrsMatrix<...>; // Matrices creuses
```

**Rôle** : Définit tous les types Kokkos utilisés dans le projet, centralisant les dépendances.

### 2. Fonctions d'activation (`activations.*`)

- **Classes** : `RELU`, `SIGMOID`, `TANH`, `LinearActivation`
- **Interface** : `Activation` avec méthodes `apply()` et `apply_derivative()`
- **Parallélisme** : Utilisation de `Kokkos::parallel_for` avec `KOKKOS_LAMBDA`

### 3. Optimiseurs (`optimizers.*`)

#### SGD (Gradient Descent Stochastique)
- Mise à jour simple : `w = w - lr * gradient`
- Support natif des matrices creuses

#### Adam (Adaptive Moment Estimation)
- **Innovation** : Moments denses pour matrices creuses
- Stockage des moments m et v en format dense 2D
- Mise à jour par structure sparse pour efficacité

### 4. Couches (`layers.*`)

#### Classe `Layer` (base)
```cpp
class Layer {
    SparseMatrixType weights;        // Poids (sparse)
    View1D biases;                   // Biais (dense)
    View1D z, a;                     // Activations
    SparseMatrixType d_weights_sum;  // Gradients accumulés
    // ...
};
```

#### Spécialisations
- **`InputLayer`** : Pas de poids, seulement des activations
- **`OutputLayer`** : Calcul de gradients spécialisé avec targets

### 5. Réseau (`network.*`)

La classe `Network` orchestre tous les composants :
- Gestion des couches (input, hidden, output)
- Forward/backward pass
- Interface avec l'optimiseur
- Calcul de coût

## Design patterns utilisés

### 1. **Strategy Pattern** (Optimiseurs)
- Interface `Optimizer` commune
- Implémentations `SGD` et `Adam` interchangeables
- Injection de dépendance via `std::unique_ptr<Optimizer>`

### 2. **Template Method** (Couches)
- Classe `Layer` avec méthodes virtuelles
- Spécialisations pour behavior spécifique (`InputLayer`, `OutputLayer`)

### 3. **Factory Pattern** (Activations)
```cpp
std::unique_ptr<Activation> create_activation(const std::string& type);
```

## Flux d'entraînement

Le cycle complet d'entraînement est implémenté dans `training.cpp` :
1. Génération ou chargement des données dans des vues Kokkos host puis copie vers le device.
2. Boucle `generic_train_network` qui :
   • fait un passage *forward* puis *backward* pour chaque échantillon d'un batch,
   • accumule les gradients dans chaque couche (`d_*_sum`),
   • déclenche `network.update()` pour appliquer l'optimiseur choisi.
3. Affichage simplifié de la progression toutes les 50 épochs via `print_progress`.
4. Critère d'arrêt : coût moyen `< convergence_threshold` défini dans la configuration.

La configuration (tailles, activations, epochs, batch size, etc.) est décrite dans des structs `TrainingConfig` spécifiques à chaque exemple (XOR, sinus, etc.).

## Gestion mémoire et performance

### Matrices creuses
- **Structure** : CRS (Compressed Row Storage) via KokkosSparse
- **Initialisation** : Poids initialisés densément puis convertis en CSR via `auto_convert_to_sparse`
- **Avantage** : Interface unifiée sparse/dense future

### Vues Kokkos
- **Zero-copy** : Partage de données entre host/device
- **RAII** : Gestion automatique de la mémoire
- **Portabilité** : Même code CPU/GPU

### Optimisations parallèles
- `Kokkos::parallel_for` pour operations élément par élément
- `Kokkos::parallel_reduce` pour les réductions (coût)
- `KokkosSparse::spmv` pour produits matrice-vecteur

## Extensibilité

L'architecture permet facilement :

1. **Nouvelles activations** : Hériter de `Activation`
2. **Nouveaux optimiseurs** : Hériter de `Optimizer` 
3. **Nouveaux types de couches** : Hériter de `Layer`
4. **Nouveaux problèmes** : Ajouter dans `training.cpp`

## Compromis de design

### Avantages
- ✅ Modularité et extensibilité
- ✅ Performance avec Kokkos
- ✅ Interface simple et intuitive
- ✅ Support matrices creuses

### Limitations actuelles
- ⚠️ Pas encore de quantization ni compression poids binaire
- ⚠️ Adam stocke moments denses (overhead mémoire)
- ⚠️ Pas de support dropout/batch norm
- ⚠️ Un seul type de perte (MSE)

### Sparsité & Pruning

Neural K prend désormais en charge des stratégies de sparsité **dynamiques** et de **pruning** :

* Détection/Conversion automatique : `is_sparse()`, `auto_convert_to_sparse()`
* Sparsité par seuil fixe : `apply_threshold_sparsity()` (+ version _silent_)
* Seuils adaptatifs : `compute_adaptive_threshold()`, `apply_adaptive_sparsity()`
* Masques permanents & régularisation L1 : `create_sparsity_masks()`, `apply_l1_regularization()`
* Pruning avancé : structurel, progressif, sensibilité, importance, regrowth (`apply_structural_pruning()`, `apply_progressive_pruning()`, `apply_sensitivity_pruning()`, `apply_importance_based_pruning()`, `apply_pruning_with_regrowth()`)
* Statistiques détaillées : `sparsity_report()`, `compute_sparsity_stats()`

Ces outils fonctionnent directement sur le format CSR et sont compatibles avec toutes les opérations KokkosSparse. 

## Aller plus loin

Pour approfondir l'implémentation, référez-vous aux sources listées dans le diagramme de dépendances. Le document `components.md` dresse l'inventaire API tandis que `usage.md` illustre pas-à-pas l'entraînement. Enfin, `examples.md` montre des cas réels d'application. 