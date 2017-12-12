# Evaluation 1: can triplet networks learn such an embedding?
# Strategy: train an embedding network for some number of iterations, and compare embedding visualizations:
# 1. In pixel space, before any embedding
# 2. After x number of embeddings
# Try this with both types of loss (softmax and triplet loss) to compare
# Format: eval1_bs_iter_softmax
from models import TwoStageIntegratedEmbeddingClassifier
import data_generators.augmentation_data_generator as DataGenerator
from data_generators.augmentation_data_generator import AugmentationDataGenerator

datagen = DataGenerator.load_augmentation_data_generator()
softmax = False

batch_size = 50
iterations = 10000

# Softmax loss:
if (softmax):
	softmaxModel = TwoStageIntegratedEmbeddingClassifier.TwoStageIntegratedEmbeddingClassifier(freeze_embed=False)
	softmaxModel.construct()
	softmaxModel.train(datagen, embed_batch_size=batch_size, embed_iterations=iterations, embed_visualize=True, iterations=0)
# Triplet loss
else:
	tripModel = TwoStageIntegratedEmbeddingClassifier.TwoStageIntegratedEmbeddingClassifier(freeze_embed=False)
	tripModel.construct(softmax=False, margin=1.0)
	tripModel.train(datagen, embed_batch_size=batch_size, embed_iterations=iterations, embed_visualize=True, iterations=0)