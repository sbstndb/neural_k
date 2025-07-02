# Neural K - Réseau de Neurones avec Matrices Creuses Kokkos

## Vue d'ensemble

**Neural K** est une implémentation moderne de réseaux de neurones utilisant des matrices creuses avec la bibliothèque Kokkos pour des performances optimales sur architectures parallèles (CPU, GPU, etc.).

### Caractéristiques principales

- 🚀 **Performance** : Utilisation de matrices creuses KokkosSparse pour l'efficacité mémoire
- 🔄 **Parallélisme** : Exécution parallèle avec Kokkos (CPU/GPU)
- 🧮 **Optimiseurs** : SGD et Adam avec support des matrices creuses
- 🌲 **Sparsité & Pruning avancés** : Seuils adaptatifs, masques permanents et stratégies de pruning structurel/regrowth
- 🎯 **Flexibilité** : Architecture modulaire et extensible
- 📊 **Exemples** : XOR, approximation de fonctions sinusoïdales, séparation linéaire

## Structure du projet

```
neural_k/
├── src/                    # Code source
│   ├── types.hpp          # Types et typedefs Kokkos
│   ├── activations.*      # Fonctions d'activation
│   ├── optimizers.*       # Algorithmes d'optimisation
│   ├── layers.*           # Couches du réseau
│   ├── network.*          # Classe Network principale
│   ├── training.*         # Fonctions d'entraînement
│   └── main.cpp           # Point d'entrée
├── doc/                   # Documentation
└── CMakeLists.txt         # Configuration build
```

## Démarrage rapide

### Prérequis

- CMake 3.20+
- Compilateur C++17
- Kokkos et KokkosKernels

### Compilation

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Exécution

```bash
./nk
```

## Exemples d'utilisation

Le programme inclut trois exemples d'entraînement :

1. **XOR** : Problème classique non-linéaire
2. **Sinus** : Approximation de fonction continue  
3. **Séparation linéaire** : Classification binaire

## Documentation détaillée

- [Architecture](architecture.md) - Structure et design du code
- [Guide d'utilisation](usage.md) - Comment utiliser la bibliothèque
- [Composants](components.md) - Détails des composants principaux
- [Exemples](examples.md) - Explications des exemples d'entraînement

## Performances

- **Matrices creuses** : Réduction significative de l'usage mémoire
- **Kokkos** : Parallélisation automatique sur CPU/GPU
- **Optimiseurs avancés** : Adam avec moments denses pour matrices creuses

## Contribution

Le code est organisé de manière modulaire pour faciliter les extensions :
- Nouvelles fonctions d'activation
- Nouveaux optimiseurs
- Nouveaux types de couches
- Nouveaux problèmes d'entraînement 