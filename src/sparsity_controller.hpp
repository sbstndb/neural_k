/*
 * sparsity_controller.hpp
 * Interface et implémentations minimales pour les stratégies de sparsité.
 */
#pragma once

#include <memory>
#include <string>
#include "types.hpp"

// Forward declaration pour éviter dépendance circulaire
class Network;

class ISparsityStrategy {
public:
    virtual ~ISparsityStrategy() = default;
    // Applique la stratégie au réseau
    virtual void apply(Network& net) = 0;
    // Nom lisible de la stratégie
    virtual std::string name() const = 0;
};

// === Stratégie simple : mise à zéro des poids |w| < threshold ===
class ThresholdSparsityStrategy : public ISparsityStrategy {
public:
    explicit ThresholdSparsityStrategy(real thr) : threshold(thr) {}
    void apply(Network& net) override;
    std::string name() const override { return "Threshold"; }
private:
    real threshold;
}; 