from models import NoEmbeddingClassifier, TwoStageIntegratedEmbeddingClassifier, TwoStageIntegratedEmbeddingClassifierReversed, TwoStageIntegratedNoEmbeddingClassifierReversed, OneStageIntegratedEmbeddingClassifier, OneStageConcatenatedEmbeddingClassifier, TwoStageConcatenatedEmbeddingClassifier
# from data_generators import DefaultDataGenerator as DataGenerator
import data_generators.augmentation_data_generator as DataGenerator
from data_generators.augmentation_data_generator import AugmentationDataGenerator

datagen = DataGenerator.load_augmentation_data_generator(is_epochal=True)
#model = NoEmbeddingClassifier.NoEmbeddingClassifier()
model = TwoStageIntegratedEmbeddingClassifierReversed.TwoStageIntegratedEmbeddingClassifierReversed(freeze_embed=False, track_embedding_loss=True)
#model = OneStageIntegratedEmbeddingClassifier.OneStageIntegratedEmbeddingClassifier()
#model = OneStageConcatenatedEmbeddingClassifier.OneStageConcatenatedEmbeddingClassifier()
#model = TwoStageConcatenatedEmbeddingClassifier.TwoStageConcatenatedEmbeddingClassifier(freeze_embed=False)
model.construct(softmax=False)
# model.train(datagen, log_freq=5, embed_iterations=100, iterations=100, keep_prob=0.5)
model.train(datagen, keep_prob=0.5, iterations=200, batch_size=50, embed_iterations=100, embed_batch_size=50, embed_visualize=True, only_originals=True)