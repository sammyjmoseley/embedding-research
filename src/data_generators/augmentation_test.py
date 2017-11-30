import cv2
from data_generators import augmentation_data_generator
import numpy as np
from data_generators.augmentation_data_generator import AugmentationDataGenerator

generator = augmentation_data_generator.load_augmentation_data_generator()
batch = generator.triplet_train(16)
# print(batch.get_positive_class().shape)
for (r, p, n), (rc, pc, nc), w in batch:
    cv2.imshow("reference", r)
    cv2.imshow("positive", p)
    cv2.imshow("negative", n)
    print(("reference-{}".format(np.argmax(rc, axis=0)), "positive-{}".format(np.argmax(pc, axis=0)), "negative-{}".format(np.argmax(nc, axis=0))))
    cv2.waitKey(0)
