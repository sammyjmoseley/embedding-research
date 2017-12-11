# Evaluation 2: Can the classifier learn augmentations via the embedding?
# Strategy: train an embedding network for some number of iterations. Then, epochally train
# the classifier using the embedding as an initialization, only on the originals. Test error?
# Compare to test error with 0 embed_iterations.
#
# Try this with both types of loss (softmax and triplet loss) to compare
# Format: eval2_softmax_embedbs_embediter_classbs_classiter_kp-percent_epochal
from models import TwoStageIntegratedEmbeddingClassifier
import data_generators.augmentation_data_generator as DataGenerator
from data_generators.augmentation_data_generator import AugmentationDataGenerator

datagen = DataGenerator.load_augmentation_data_generator(is_epochal=True)
softmax = False

embed_bs = 50
embed_iter = 1000
class_bs = 50
class_iter = 1000
kp = 0.5

# Softmax loss:
if (softmax):
	softmaxModel = TwoStageIntegratedEmbeddingClassifier.TwoStageIntegratedEmbeddingClassifier(freeze_embed=False, track_embedding_loss=True)
	softmaxModel.construct()
	softmaxModel.train(datagen, embed_batch_size=embed_bs, embed_iterations=embed_iter, embed_visualize=True, iterations=class_iter, batch_size=class_bs, only_originals=True, keep_prob=kp)
	softmaxModel.train(datagen, embed_iterations=0, embed_visualize=True, iterations=class_iter, batch_size=class_bs, only_originals=True, keep_prob=kp)
	softmaxModel.train(datagen, embed_iterations=0, embed_visualize=True, iterations=(class_iter+embed_iter), batch_size=class_bs, only_originals=True, keep_prob=kp)
# Triplet loss
else:
	tripModel = TwoStageIntegratedEmbeddingClassifier.TwoStageIntegratedEmbeddingClassifier(freeze_embed=False, track_embedding_loss=True)
	tripModel.construct(softmax=False, margin=1.0)
	tripModel.train(datagen, embed_batch_size=embed_bs, embed_iterations=embed_iter, embed_visualize=True, iterations=class_iter, batch_size=class_bs, only_originals=True, keep_prob=kp)
	tripModel.train(datagen, embed_iterations=0, embed_visualize=True, iterations=class_iter, batch_size=class_bs, only_originals=True, keep_prob=kp)
	tripModel.train(datagen, embed_iterations=0, embed_visualize=True, iterations=(class_iter+embed_iter), batch_size=class_bs, only_originals=True, keep_prob=kp)

