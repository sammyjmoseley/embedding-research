import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import decomposition


def pca_visualize(x, embed, y_):
    pca = decomposition.PCA(n_components=2)
    embed = pca.fit_transform(embed)

    # two ways of visualization: scale to fit [0,1] scale
    # feat = embed - np.min(embed, 0)
    # feat /= np.max(feat, 0)

    # two ways of visualization: leave with original scale
    feat = embed
    ax_min = np.min(embed,0)
    ax_max = np.max(embed,0)
    ax_dist_sq = np.sum((ax_max-ax_min)**2)

    plt.figure()
    ax = plt.subplot(111)
    shown_images = np.array([[1., 1.]])
    for i in range(feat.shape[0]):
        dist = np.sum((feat[i] - shown_images)**2, 1)
        if np.min(dist) < 3e-4*ax_dist_sq:   # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [feat[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(x[i], zoom=0.5, cmap=plt.cm.gray_r),
            xy=feat[i], frameon=False
        )
        ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    # plt.xticks([]), plt.yticks([])
    plt.title('PCA Embedding')
    plt.show()


def tsne_visualize(x, embed, y_):
	print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(embed)
    plot_tsne_embedding(x, X_tsne, y_, "t-SNE Embedding")
    plt.show()

def plot_tsne_embedding(x, embed, y_, title):
	y_ = np.argmax(y_, axis=1)
	x_min, x_max = np.min(embed, 0), np.max(embed, 0)
    embed = (embed - x_min) / (x_max - x_min)
    ax_dist_sq = np.sum((x_max-x_min)**2)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(embed[i, 0], embed[i, 1], str(y_[i]),
                 color=plt.cm.Set1(y_[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(images.shape[0]):
            dist = np.sum((embed[i] - shown_images)**2, 1)
            if np.min(dist) < 5e-6*ax_dist_sq:   # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [embed[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(x[i], cmap=plt.cm.gray_r, zoom=0.5),
                xy=embed[i], frameon=False)
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)