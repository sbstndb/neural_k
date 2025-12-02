# Neural K - Réseau de Neurones avec Kokkos

Neural K est un projet de réseau de neurones minimaliste exploitant la puissance de [Kokkos](https://github.com/kokkos) pour la portabilité multi-backend (CPU/GPU). Son objectif est d'offrir une première prise en main de la bibliothèque Kokkos ainsi que des bases de l'IA.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         NEURAL K                                │
├─────────────────────────────────────────────────────────────────┤
│  InputLayer ──▶ HiddenLayer(s) ──▶ OutputLayer                  │
│      │              │                   │                       │
│   set_input()    forward()          forward()                   │
│      │         z = W·a + b         z = W·a + b                  │
│      ▼         a = σ(z)            a = σ(z)                     │
│    [a]            [a]                [pred]                     │
└─────────────────────────────────────────────────────────────────┘
```

### Composants principaux

| Fichier | Description |
|---------|-------------|
| `types.hpp` | Types Kokkos : `View1D`, `SparseMatrixType` (CSR) |
| `layers.hpp/cpp` | Classes `InputLayer`, `Layer`, `OutputLayer` |
| `activations.hpp/cpp` | ReLU, Sigmoid, Tanh, Linear |
| `network.hpp/cpp` | Orchestration forward/backward pass |
| `optimizers.hpp/cpp` | SGD et Adam |
| `training.hpp/cpp` | Boucles d'entraînement génériques |
| `sparsity_*.hpp/cpp` | Stratégies de pruning et sparsité |

### Flux de données

**Forward Pass** : Les données traversent le réseau couche par couche
```
Input ──▶ SpMV(W·a) + b ──▶ Activation(z) ──▶ Output
```

**Backward Pass** : Rétropropagation des gradients
```
δ_output = (pred - target) · σ'(z)
δ_hidden = (W_next^T · δ_next) · σ'(z)
∇W = δ · a_prev^T   (outer product sparse)
```

### Matrices creuses (CSR)

Les poids sont stockés au format **Compressed Row Storage** via KokkosSparse :
```
row_map  : offsets des lignes
entries  : indices colonnes
values   : valeurs non-nulles
```

### Optimiseurs

| Optimiseur | Formule |
|------------|---------|
| **SGD** | `w = w - lr · ∇w` |
| **Adam** | `w = w - lr · m̂/(√v̂ + ε)` avec moments adaptatifs |

### Sparsité et Pruning

Plusieurs stratégies disponibles : threshold, L1 regularization, structural pruning, sensitivity-based, progressive, adaptive...

## Fonctionnalités clés

- API C++ moderne et épurée reposant sur Kokkos
- Support natif des matrices creuses via KokkosSparse (CSR)
- Optimiseurs intégrés : SGD et Adam
- Système de sparsité/pruning extensible
- Portabilité CPU/GPU (OpenMP, CUDA, HIP)
- Exemples complets d'entraînement dans `doc/examples.md`

 From the [Kokkos](https://github.com/kokkos) github repo : 
> The Kokkos C++ Performance Portability Ecosystem is a production level solution for writing modern C++ applications in a hardware agnostic way.
Thanks to Kokkos, you can compile this code for multiple backends like OpenMP, CUDA, HIP.


**Disclaimer:** Please note that this project is a work in progress and may contain errors or programming oversights due to its experimental nature. Your understanding and feedback are appreciated as we continue to develop and refine this code.

# Compilation
Compile the code with the following commands : 
```
git submodule update --init --recursive
mkdir build && cd build && cmake ..
make -j 
```

# Usage
*You can currently launch the executable named `nk` with the following command :
```
./nk
```
This will run the executable with the default parameters.

# Prerequisites
The code use Kokkos as a performance portability library. Then, you must have it to compile the project.
I suggest you to install it and read the documentation for further understanding. 

It is possible to easily install `Kokkos` through the HPC [`spack`](https://github.com/spack) package manager. I suggest you to create a new environment : 
```
spack env create kokkos
spack env activate kokkos
spack install kokkos // here you can specify your backend like OpenMP, pthread, CUDA, HIP, ... please read the doc
spack load kokkos
```

# Todo 
### To-Do List

- [x] Provide the MVP (Minimal Viable Product)
- [ ] Support flexible network architectures
- [ ] Enable GPU-accelerated Execution (CUDA/HIP)
- [ ] Design user-friendly API
- [ ] Add tests
- [ ] Add documentation
- [ ] Implement batch training strategies

