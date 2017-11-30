# Evaluation 4: impact on robustness to adversarial attacks?
# Strategy: first, check if augmentations make you more susceptible to attacks
# 1. No embedding, train on no augmentations (only_originals=True), for x iterations
# 2. No embedding, train on augmentations for x/5 iterations (to make sure they both see the same number of examples)
# 3. Use https://github.com/carlini/nn_robust_attacks to compare their susceptibility
# 4. Embedding, train on no augmentations (only_originals=True), for x iterations
# 5. What is the robustness of the embedding version?
# Format: eval2_x
