from sklearn import manifold
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox

# Scale and visualize the embedding vectors
def plot_embedding(X, images, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax_dist_sq = np.sum((x_max-x_min)**2)

    plt.figure()
    ax = plt.subplot(111)
    """for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})"""

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(images.shape[0]):
            dist = np.sum((X[i] - shown_images)**2, 1)
            if np.min(dist) < 5e-6*ax_dist_sq:   # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r, zoom=0.5),
                xy=X[i], frameon=False)
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def visualize(embed, x_test):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(embed)
    plot_embedding(X_tsne, x_test,
                "t-SNE embedding of the digits (time %.2fs)" %
                (time() - t0))
    plt.show()