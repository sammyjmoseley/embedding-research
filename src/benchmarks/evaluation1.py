# Evaluation 1: can triplet networks learn such an embedding?
# Strategy: train an embedding network for some number of iterations, and compare embedding visualizations:
# 1. In pixel space, before any embedding
# 2. After x number of embeddings
# Try this with both types of loss (softmax and triplet loss) to compare
from models import TwoStageIntegratedEmbeddingClassifier
import data_generators.augmentation_data_generator as DataGenerator
from data_generators.augmentation_data_generator import AugmentationDataGenerator

datagen = DataGenerator.load_augmentation_data_generator()

# Softmax loss:
softmaxModel = TwoStageIntegratedEmbeddingClassifier.TwoStageIntegratedEmbeddingClassifier(freeze_embed=False)
softmaxModel.construct()
model.train(datagen, embed_batch_size=64, embed_iterations=100, embed_visualize=True)