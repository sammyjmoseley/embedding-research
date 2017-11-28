from models import NoEmbeddingClassifier
from data_generators import DefaultDataGenerator

datagen = DefaultDataGenerator.RotatedMNISTDataGenerator()
model = NoEmbeddingClassifier.NoEmbeddingClassifier()
model.construct()
model.train(datagen)