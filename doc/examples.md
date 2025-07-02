# Exemples d'entraînement
Cette section regroupe plusieurs scénarios afin de démontrer la flexibilité de Neural K sur des tâches de classification et de régression.

Les exemples inclus :
- XOR (classification binaire)
- Approximation de sin(x) (régression)
- Séparation linéaire (classification linéaire)

Chaque exemple suit la même structure : génération de données, définition du réseau, entraînement puis évaluation.

## 1. XOR - Problème non-linéaire classique

### Description du problème

La fonction XOR (OU exclusif) est un problème de classification binaire non-linéairement séparable :

| Input A | Input B | Output |
|---------|---------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Défis** :
- Impossible à résoudre avec un seul perceptron
- Nécessite au moins une couche cachée
- Benchmark classique pour tester les réseaux de neurones

### Architecture utilisée

```cpp
std::map<int, int> sizes;
sizes[0] = 2;    // 2 inputs (A, B)
sizes[1] = 80;   // 80 neurones cachés (couche 1)
sizes[2] = 40;   // 40 neurones cachés (couche 2)  
sizes[3] = 1;    // 1 output (XOR result)

std::vector<std::string> activations = {"relu", "relu", "sigmoid"};
```

**Justifications** :
- **ReLU** dans les couches cachées : Évite le problème des gradients évanescents
- **Sigmoid** en sortie : Output entre 0 et 1 pour classification binaire
- **Architecture profonde** : 2 couches cachées pour capturer la non-linéarité

### Hyperparamètres

| Optimiseur | Learning Rate | Epochs | Batch Size |
|------------|---------------|--------|------------|
| Adam | 0.01 | 800 | 4 (tous les échantillons) |
| SGD | 0.8 | 1500 | 4 |

### Résultats attendus

- **Convergence** : Coût < 1e-4 après ~400-600 epochs (Adam)
- **Précision** : 100% sur les 4 échantillons
- **Prédictions** : Valeurs proches de 0 ou 1

## 2. Approximation de fonction sinus

### Description du problème

Approximer la fonction `sin(x)` sur l'intervalle `[-π, π]` :
- **Input** : Valeur x ∈ [-π, π]
- **Target** : sin(x) ∈ [-1, 1]
- **Type** : Problème de régression continue

### Architecture utilisée

```cpp
std::map<int, int> sizes;
sizes[0] = 1;    // 1 input (x)
sizes[1] = 32;   // 32 neurones cachés (couche 1)
sizes[2] = 32;   // 32 neurones cachés (couche 2)
sizes[3] = 1;    // 1 output (sin(x))

std::vector<std::string> activations = {"relu", "relu", "linear"};
```

**Justifications** :
- **ReLU** dans les couches cachées : Bonne approximation des fonctions non-linéaires
- **Linear** en sortie : Output non-borné pour régression
- **Taille modérée** : 32 neurones suffisants pour capturer les oscillations

### Génération des données

```cpp
std::uniform_real_distribution<real> distrib(-M_PI, M_PI);
for(int i = 0; i < num_samples; ++i) {
    real x = distrib(gen);
    h_inputs(i, 0) = x;
    h_outputs(i, 0) = std::sin(x);
}
```

- **Échantillons** : 8192 points aléatoires
- **Distribution** : Uniforme sur [-π, π]
- **Mélange** : Shuffle des indices pour chaque epoch

### Hyperparamètres

| Optimiseur | Learning Rate | Epochs | Batch Size | Échantillons |
|------------|---------------|--------|------------|--------------|
| Adam | 0.001 | 800 | 16 | 8192 |
| SGD | 0.02 | 500 | 16 | 8192 |

### Évaluation

```cpp
// Échantillons de test
for (int i = 0; i < 10; ++i) {
    real input_x = h_inputs(i, 0);
    real target_y = std::sin(input_x);
    View1D prediction = network.forward(input_subview);
    
    std::cout << "Input x: " << input_x 
              << " Target sin(x): " << target_y
              << " Prediction: " << prediction_value << std::endl;
}
```

### Résultats attendus

- **Coût final** : < 0.01 (MSE)
- **Précision** : Erreur absolue < 0.1 sur la plupart des points
- **Généralisation** : Bonne approximation même hors échantillons d'entraînement

## 3. Séparation linéaire

### Description du problème

Classification binaire avec frontière linéaire :
- **Règle** : Si `y > x + margin` → Classe 1, Si `y < x - margin` → Classe 0
- **Marge** : Zone d'exclusion autour de la ligne `y = x`
- **Type** : Classification binaire linéairement séparable

### Génération des données

```cpp
real margin = 0.1;
for(int i = 0; i < num_samples; ++i) {
    real x = distrib(gen);  // [-1, 1]
    real y = distrib(gen);  // [-1, 1]
    
    if (y < x - margin) {
        h_outputs(i, 0) = 0.0;      // Classe 0
    } else if (y > x + margin) {
        h_outputs(i, 0) = 1.0;      // Classe 1
    } else {
        i--; continue;  // Rejeter points dans la marge
    }
}
```

### Architecture utilisée

```cpp
std::map<int, int> sizes;
sizes[0] = 2;    // 2 inputs (x, y)
sizes[1] = 1;    // 1 output (classe)

std::vector<std::string> activations = {"sigmoid"};
```

**Justifications** :
- **Architecture simple** : Problème linéairement séparable
- **Sigmoid** : Output probabiliste pour classification
- **Pas de couche cachée** : Un perceptron suffit

### Hyperparamètres

| Optimiseur | Learning Rate | Epochs | Batch Size | Échantillons |
|------------|---------------|--------|------------|--------------|
| Adam | 0.01 | 150 | 16 | 2560 |
| SGD | 0.1 | 100 | 16 | 2560 |

### Métriques d'évaluation

```cpp
int correct_predictions = 0;
for (int i = 0; i < num_samples; ++i) {
    View1D prediction = network.forward(input_subview);
    int predicted_class = std::round(prediction_value);
    int target_class = static_cast<int>(h_outputs(i, 0));
    
    if (predicted_class == target_class) correct_predictions++;
}

real accuracy = static_cast<real>(correct_predictions) / num_samples;
```

### Résultats attendus

- **Précision** : > 95% après convergence
- **Convergence** : Rapide (< 100 epochs)
- **Frontière** : Claire séparation près de la ligne y = x

## Comparaison des optimiseurs

### Adam vs SGD

| Critère | Adam | SGD |
|---------|------|-----|
| **Convergence** | Plus rapide | Plus lente |
| **Stabilité** | Plus stable | Peut osciller |
| **Mémoire** | Plus gourmand | Moins gourmand |
| **Hyperparamètres** | Moins sensible au LR | Très sensible au LR |

### Recommandations

- **Adam** : Premier choix pour la plupart des problèmes
- **SGD** : Quand la mémoire est limitée ou pour le fine-tuning
- **Learning Rate** : 
  - Adam : 0.001 - 0.01
  - SGD : 0.01 - 1.0 (très dépendant du problème)

## Conseils de débogage

### Problèmes courants

1. **Pas de convergence**
   - Réduire le learning rate
   - Vérifier la préparation des données
   - Augmenter le nombre d'epochs

2. **Overfitting rapide**
   - Réduire la taille du réseau
   - Ajouter de la régularisation (future extension)

3. **Gradients explosifs**
   - Réduire drastiquement le learning rate
   - Vérifier l'initialisation des poids

4. **Performances médiocres**
   - Vérifier l'architecture (taille, activations)
   - Augmenter la complexité du modèle
   - Améliorer la qualité des données

### Surveillance de l'entraînement

```cpp
// Affichage régulier du coût
if ((epoch + 1) % (epochs / 20) == 0) {
    std::cout << "Epoch: " << epoch + 1 
              << ", Cost: " << avg_cost << std::endl;
}

// Critère de convergence
if (avg_cost < threshold) {
    std::cout << "Convergence reached!" << std::endl;
    break;
}
``` 