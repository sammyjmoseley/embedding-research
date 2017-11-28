from models import NoEmbeddingClassifier
from data_generators import DefaultDataGenerator

datagen = DefaultDataGenerator.RotatedMNISTDataGenerator()
model = NoEmbeddingClassifier.NoEmbeddingClassifier()
#model = TwoStageIntegratedEmbeddingClassifier.TwoStageIntegratedEmbeddingClassifier(freeze_embed=False)
#model = OneStageIntegratedEmbeddingClassifier.OneStageIntegratedEmbeddingClassifier()
model.construct()
model.train(datagen, log_freq=5, iterations=500, keep_prob=0.5)