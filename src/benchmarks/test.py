from models import NoEmbeddingClassifier, TwoStageIntegratedEmbeddingClassifier, OneStageIntegratedEmbeddingClassifier, OneStageConcatenatedEmbeddingClassifier, TwoStageConcatenatedEmbeddingClassifier
from data_generators import DefaultDataGenerator

datagen = DefaultDataGenerator.RotatedMNISTDataGenerator()
#model = NoEmbeddingClassifier.NoEmbeddingClassifier()
model = TwoStageIntegratedEmbeddingClassifier.TwoStageIntegratedEmbeddingClassifier(freeze_embed=False)
#model = OneStageIntegratedEmbeddingClassifier.OneStageIntegratedEmbeddingClassifier()
#model = OneStageConcatenatedEmbeddingClassifier.OneStageConcatenatedEmbeddingClassifier()
#model = TwoStageConcatenatedEmbeddingClassifier.TwoStageConcatenatedEmbeddingClassifier(freeze_embed=False)
model.construct()
# model.train(datagen, log_freq=5, embed_iterations=100, iterations=100, keep_prob=0.5)
model.train(datagen, keep_prob=0.5, iterations=2000, embed_iterations=100, embed_batch_size=32)