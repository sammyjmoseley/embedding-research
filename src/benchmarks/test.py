from models import NoEmbeddingClassifier, TwoStageIntegratedEmbeddingClassifier, OneStageIntegratedEmbeddingClassifier, OneStageConcatenatedEmbeddingClassifier, TwoStageConcatenatedEmbeddingClassifier
# from data_generators import DefaultDataGenerator as DataGenerator
import data_generators.augmentation_data_generator as DataGenerator
from data_generators.augmentation_data_generator import AugmentationDataGenerator

datagen = DataGenerator.load_augmentation_data_generator()
#model = NoEmbeddingClassifier.NoEmbeddingClassifier()
model = TwoStageIntegratedEmbeddingClassifier.TwoStageIntegratedEmbeddingClassifier(freeze_embed=False)
#model = OneStageIntegratedEmbeddingClassifier.OneStageIntegratedEmbeddingClassifier()
#model = OneStageConcatenatedEmbeddingClassifier.OneStageConcatenatedEmbeddingClassifier()
#model = TwoStageConcatenatedEmbeddingClassifier.TwoStageConcatenatedEmbeddingClassifier(freeze_embed=False)
model.construct()
# model.train(datagen, log_freq=5, embed_iterations=100, iterations=100, keep_prob=0.5)
model.train(datagen, keep_prob=0.5, iterations=100, embed_iterations=100, embed_batch_size=32, embed_visualize=True)