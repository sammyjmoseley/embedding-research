# Evaluation 4: impact on robustness to adversarial attacks?
# Strategy: first, check if augmentations make you more susceptible to attacks
# 1. No embedding, train on no augmentations (only_originals=True), for x iterations
# 2. No embedding, train on augmentations for x/5 iterations (to make sure they both see the same number of examples)
# 3. Use https://github.com/carlini/nn_robust_attacks to compare their susceptibility
# 4. Embedding, train on no augmentations (only_originals=True), for x iterations
# 5. What is the robustness of the embedding version?
# Format: eval2_x
from models.TwoStageIntegratedHardEmbeddingClassifier import TwoStageIntegratedEmbeddingClassifier
from data_generators.HardExampleDataGenerator import RotatedMNISTDataGenerator as DataGenerator
from data_generators.augmentation_data_generator import AugmentationDataGenerator
from nn_robust_attacks.l2_attack import CarliniL2
import numpy as np
import tensorflow as tf

datagen = DataGenerator()
softmax = True

neModel = TwoStageIntegratedEmbeddingClassifier(freeze_embed=False)
neModel.construct()
neModel.train(datagen, embed_iterations=0, embed_visualize=False, iterations=0, only_originals=True)

np.random.seed(1)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	x, y_ = datagen.test(1)
	idx = np.random.permutation(1)
	y_[0] = (y_[0]+1)%10
	adv = CarliniL2(sess, neModel, max_iterations=2).attack(x, y_)
	d = np.pow(np.sum(np.square(adv-x)), 0.5)