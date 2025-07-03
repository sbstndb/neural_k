#pragma once

#include "sparsity_controller.hpp"

class Network; // forward declaration

class L1RegularizationStrategy : public ISparsityStrategy {
public:
    explicit L1RegularizationStrategy(real lambda);
    void apply(Network& net) override;
    std::string name() const override { return "L1Regularization"; }
private:
    real lambda_;
};

class StructuralPruningStrategy : public ISparsityStrategy {
public:
    explicit StructuralPruningStrategy(real threshold);
    void apply(Network& net) override;
    std::string name() const override { return "StructuralPruning"; }
private:
    real threshold_;
};

class SensitivityPruningStrategy : public ISparsityStrategy {
public:
    explicit SensitivityPruningStrategy(real threshold);
    void apply(Network& net) override;
    std::string name() const override { return "SensitivityPruning"; }
private:
    real threshold_;
};

class ImportanceBasedPruningStrategy : public ISparsityStrategy {
public:
    explicit ImportanceBasedPruningStrategy(real sparsity_target);
    void apply(Network& net) override;
    std::string name() const override { return "ImportanceBasedPruning"; }
private:
    real sparsity_target_;
};

class ProgressivePruningStrategy : public ISparsityStrategy {
public:
    ProgressivePruningStrategy(real initial_threshold, real final_threshold, int epochs);
    void apply(Network& net) override;
    std::string name() const override { return "ProgressivePruning"; }
private:
    real initial_threshold_;
    real final_threshold_;
    int epochs_;
};

class LayerSpecificPruningStrategy : public ISparsityStrategy {
public:
    explicit LayerSpecificPruningStrategy(const std::vector<real>& thresholds);
    void apply(Network& net) override;
    std::string name() const override { return "LayerSpecificPruning"; }
private:
    std::vector<real> thresholds_;
};

class AdaptiveSparsityStrategy : public ISparsityStrategy {
public:
    explicit AdaptiveSparsityStrategy(real target_sparsity);
    void apply(Network& net) override;
    std::string name() const override { return "AdaptiveSparsity"; }
private:
    real target_sparsity_;
};

class ProgressiveSparsityStrategy : public ISparsityStrategy {
public:
    ProgressiveSparsityStrategy(real target_sparsity, real max_threshold);
    void apply(Network& net) override;
    std::string name() const override { return "ProgressiveSparsity"; }
private:
    real target_sparsity_;
    real max_threshold_;
};

class LayerAdaptiveSparsityStrategy : public ISparsityStrategy {
public:
    explicit LayerAdaptiveSparsityStrategy(real target_sparsity);
    void apply(Network& net) override;
    std::string name() const override { return "LayerAdaptiveSparsity"; }
private:
    real target_sparsity_;
}; 