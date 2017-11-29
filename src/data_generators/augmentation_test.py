import cv2
from data_generators.DefaultDataGenerator import RotatedMNISTDataGenerator

generator = RotatedMNISTDataGenerator()
batch = generator.triplet_train(16)
print(batch.get_reference().shape)
for (r, p, n), (rc, pc, nc), w in batch:
    cv2.imshow("reference", r)
    cv2.imshow("positive", p)
    cv2.imshow("negative", n)
    print(("reference-{}".format(rc), "positive-{}".format(pc), "negative-{}".format(nc)))
    print(w)
    cv2.waitKey(0)
