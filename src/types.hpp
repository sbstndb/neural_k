#pragma once

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
#include <numeric>
#include <iomanip>
#include <typeinfo>
#include <chrono>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosKernels_Handle.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Typedefs ---
using real = float;
using View1D = Kokkos::View<real*>;

// *** TYPEDEFS POUR SPARSE MATRIX ***
using Scalar = real;
using Ordinal = int;
using Offset = size_t;
using Device = Kokkos::DefaultExecutionSpace;
using Layout = Kokkos::LayoutLeft;
using SparseMatrixType = KokkosSparse::CrsMatrix<Scalar, Ordinal, Device, void, Offset>;
using GraphType = typename SparseMatrixType::staticcrsgraph_type;
using ValuesType = typename SparseMatrixType::values_type;
using RowMapType = typename GraphType::row_map_type::non_const_type;
using EntriesType = typename GraphType::entries_type::non_const_type; 