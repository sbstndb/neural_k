# Neural K - Neural Network with Kokkos

Neural K is a minimalist neural network project leveraging the power of [Kokkos](https://github.com/kokkos) for multi-backend portability (CPU/GPU). Its goal is to provide a first hands-on experience with the Kokkos library and AI fundamentals.

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

### Main Components

| File | Description |
|------|-------------|
| `types.hpp` | Kokkos types: `View1D`, `SparseMatrixType` (CSR) |
| `layers.hpp/cpp` | `InputLayer`, `Layer`, `OutputLayer` classes |
| `activations.hpp/cpp` | ReLU, Sigmoid, Tanh, Linear |
| `network.hpp/cpp` | Forward/backward pass orchestration |
| `optimizers.hpp/cpp` | SGD and Adam |
| `training.hpp/cpp` | Generic training loops |
| `sparsity_*.hpp/cpp` | Pruning and sparsity strategies |

### Data Flow

**Forward Pass**: Data flows through the network layer by layer
```
Input ──▶ SpMV(W·a) + b ──▶ Activation(z) ──▶ Output
```

**Backward Pass**: Gradient backpropagation
```
δ_output = (pred - target) · σ'(z)
δ_hidden = (W_next^T · δ_next) · σ'(z)
∇W = δ · a_prev^T   (outer product sparse)
```

### Sparse Matrices (CSR)

Weights are stored in **Compressed Row Storage** format via KokkosSparse:
```
row_map  : row offsets
entries  : column indices
values   : non-zero values
```

### Optimizers

| Optimizer | Formula |
|-----------|---------|
| **SGD** | `w = w - lr · ∇w` |
| **Adam** | `w = w - lr · m̂/(√v̂ + ε)` with adaptive moments |

### Sparsity and Pruning

Several strategies available: threshold, L1 regularization, structural pruning, sensitivity-based, progressive, adaptive...

## Key Features

- Modern and clean C++ API built on Kokkos
- Native sparse matrix support via KokkosSparse (CSR)
- Built-in optimizers: SGD and Adam
- Extensible sparsity/pruning system
- CPU/GPU portability (OpenMP, CUDA, HIP)
- Complete training examples in `doc/examples.md`

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

