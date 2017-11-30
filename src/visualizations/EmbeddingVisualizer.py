import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import decomposition, manifold


def pca_visualize(x, embed, y_):
    
    pca = decomposition.PCA(n_components=2)
    embed = pca.fit_transform(embed)

    # two ways of visualization: scale to fit [0,1] scale
    # feat = embed - np.min(embed, 0)
    # feat /= np.max(feat, 0)

    # two ways of visualization: leave with original scale
    
    ax_min = np.min(embed,0)
    ax_max = np.max(embed,0)
    ax_dist_sq = np.sum((ax_max-ax_min)**2)

    feat = embed

    plt.figure(1, figsize=(10,40))
    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    ax = plt.subplot(411)
    colors = []
    for i in range(embed.shape[0]):
        colors.append(plt.cm.tab10(y_[i] / 10.))
    plt.scatter(embed[:, 0], embed[:, 1], c=colors, alpha=0.5)
    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    plt.title('PCA Embedding Classes')

    ax = plt.subplot(412)
    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
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
    plt.title('PCA Embedding Images')


def tsne_visualize(x, embed, y_):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(embed)
    plot_tsne_embedding(x, X_tsne, y_, "t-SNE Embedding")

def plot_tsne_embedding(x, embed, y_, title):
    x_min, x_max = np.min(embed, 0), np.max(embed, 0)
    embed = (embed - x_min) / (x_max - x_min)
    ax_dist_sq = np.sum((x_max-x_min)**2)

    ax = plt.subplot(413)
    colors = []
    for i in range(embed.shape[0]):
        colors.append(plt.cm.tab10(y_[i] / 10.))
    plt.scatter(embed[:, 0], embed[:, 1], c=colors, alpha=0.5)
    plt.title('t-SNE Embedding Classes')

    ax = plt.subplot(414)
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(x.shape[0]):
            dist = np.sum((embed[i] - shown_images)**2, 1)
            if np.min(dist) < 5e-6*ax_dist_sq:   # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [embed[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(x[i], cmap=plt.cm.gray_r, zoom=0.5),
                xy=embed[i], frameon=False)
            ax.add_artist(imagebox)
    plt.title('t-SNE Embedding Images')

def visualize(x, embed, y_, file_path):
    if (x.shape[3] == 1):
        x = x[:,:,:,0]
    embed = embed.reshape(embed.shape[0], -1)
    y_ = np.argmax(y_, axis=1)
    pca_visualize(x, embed, y_)
    tsne_visualize(x, embed, y_)
    plt.savefig(file_path+".png")
    plt.cla()
    plt.clf()
    plt.close()