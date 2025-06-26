#include "types.hpp"
#include "training.hpp"

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        try {
            std::cout << "Kokkos execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
            std::cout << "*** NOTE: Using KokkosSparse::CrsMatrix for weights storage (initially dense structure). ***" << std::endl;

            xor_train("adam");
            std::cout << "\n---------------------------\n" << std::endl;

            std::cout << "\n---------------------------\n" << std::endl;
            sine_train("adam");

            std::cout << "\n---------------------------\n" << std::endl;
            linear_sep_train("adam");

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            Kokkos::finalize();
            return 1;
        } catch (...) {
            std::cerr << "An unknown error occurred." << std::endl;
            Kokkos::finalize();
             return 1;
        }
    }
    Kokkos::finalize();
    return 0;
} 