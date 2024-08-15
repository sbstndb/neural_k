# Introduction
 Here is a simple Neural Network project. 
 The aim of this project is to have a first look at the Kokkos library and AI science.

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
*You can currently launch the executable named `NN` with the following command :
```
./GS
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

- [ ] Provide the MVP (Minimal Viable Product)

