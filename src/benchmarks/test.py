from models import TwoStageIntegratedEmbeddingClassifier
from data_generators import DefaultDataGenerator

datagen = DefaultDataGenerator.RotatedMNISTDataGenerator()
#model = NoEmbeddingClassifier.NoEmbeddingClassifier()
model = TwoStageIntegratedEmbeddingClassifier.TwoStageIntegratedEmbeddingClassifier()
model.construct()
model.train(datagen, keep_prob=0.5)