# Evaluation 3: Model B and its comparisons
from models import TwoStageIntegratedEmbeddingClassifierReversed
import data_generators.augmentation_data_generator as DataGenerator
from data_generators.augmentation_data_generator import AugmentationDataGenerator

datagen = DataGenerator.load_augmentation_data_generator(is_epochal=True)
softmax = False

embed_bs = 50
embed_iter = 5000
class_bs = 50
class_iter = 5000
kp = 0.5

tripModel = TwoStageIntegratedEmbeddingClassifierReversed.TwoStageIntegratedEmbeddingClassifierReversed(freeze_embed=False, track_embedding_loss=True)
tripModel.construct(softmax=False, margin=1.0)
tripModel.train(datagen, embed_batch_size=embed_bs, embed_iterations=embed_iter, embed_visualize=True, iterations=class_iter, batch_size=class_bs, only_originals=True, keep_prob=kp)