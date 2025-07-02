#include "types.hpp"
#include "training.hpp"

// Fonction utilitaire pour l'affichage principal
void print_main_header() {
    std::cout << std::endl;
    std::string line(80, '=');
    std::cout << line << std::endl;
    std::cout << "|                     NEURAL NETWORK KOKKOS (SPARSE)                      |" << std::endl;
    std::cout << "|                        Bibliotheque de Reseaux de Neurones              |" << std::endl;
    std::cout << "|                             avec Matrices Creuses                       |" << std::endl;
    std::cout << line << std::endl;
    std::cout << std::endl;
}

void print_system_info() {
    std::cout << "Informations systeme:" << std::endl;
    std::cout << "   - Espace d'execution Kokkos : " << Kokkos::DefaultExecutionSpace::name() << std::endl;
    std::cout << "   - Stockage des poids        : KokkosSparse::CrsMatrix (structure dense initialement)" << std::endl;
    std::cout << "   - Parallelisation           : Activee via Kokkos" << std::endl;
    std::cout << std::endl;
}

void print_demo_separator(int demo_num, const std::string& title) {
    std::string line(80, '-');
    std::cout << line << std::endl;
    std::cout << "                          DEMONSTRATION " << demo_num << " - " << title << std::endl;
    std::cout << line << std::endl;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        try {
            print_main_header();
            print_system_info();

            // Démonstration 1: XOR
            print_demo_separator(1, "XOR");
            xor_train("adam");

            // Démonstration 2: Sinus  
            print_demo_separator(2, "SINUS");
            sine_train("adam");

            // Démonstration 3: Séparation linéaire
            print_demo_separator(3, "SEPARATION LINEAIRE");
            linear_sep_train("adam");

            // Démonstration 4: Classification spirales
            print_demo_separator(4, "SPIRALES MULTI-CLASSES");
            spiral_train("adam");

            // Démonstration 5: Classification clusters gaussiens
            print_demo_separator(5, "CLUSTERS GAUSSIENS");
            gaussian_clusters_train("adam");

            // Démonstration 6: Prédiction séries temporelles 
            print_demo_separator(6, "SERIES TEMPORELLES");
            time_series_train("adam");

            // Démonstration 7: NOUVELLE - Test de sparsité
            print_demo_separator(7, "SPARSITÉ POST-ENTRAÎNEMENT");
            test_sparsity_xor("adam", 0.01); // Seuil de 1%

            // Démonstration 8: NOUVELLE - Test de sparsité sur sinus
            print_demo_separator(8, "SPARSITÉ SINUS - RÉGRESSION");
            test_sparsity_sine("adam", 0.005); // Seuil plus fin pour régression

        } catch (const std::exception& e) {
            std::cerr << "\nERREUR CRITIQUE:" << std::endl;
            std::cerr << "   " << e.what() << std::endl;
            std::cerr << "\nVerifiez votre installation Kokkos et recompilez." << std::endl;
            Kokkos::finalize();
            return 1;
        } catch (...) {
            std::cerr << "\nERREUR INCONNUE DETECTEE" << std::endl;
            std::cerr << "\nContactez le support technique." << std::endl;
            Kokkos::finalize();
             return 1;
        }
    }
    Kokkos::finalize();
    return 0;
} 
