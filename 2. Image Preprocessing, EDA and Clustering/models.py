import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

from PIL import Image
from skimage.util import montage

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def extract_features(paths, model="res"):
    """ResNet50 Credit to https://towardsdatascience.com/image-clustering-using-transfer-learning-df5862779571
        VGG16 Credit to https://keras.io/api/applications/"""

    feat_out = []
    if model == "res":
        shoe_net = ResNet50(include_top=False, pooling="avg", weights="imagenet")
        shoe_net.layers[0].trainable = False
    elif model == "vgg":
        shoe_net = VGG16(weights="imagenet", include_top=False)
        shoe_net.layers[0].trainable = False
    for im in tqdm(paths):
        im = image.load_img(im, target_size=(224, 224))
        img = image.img_to_array(im)
        if model == "res":
            img = preprocess_input(np.expand_dims(img, axis=0))
        elif model == "vgg":
            img = vgg16.preprocess_input(np.expand_dims(img, axis=0))
        resnet_feature = shoe_net.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        feat_out.append(resnet_feature_np.flatten())
    return np.array(feat_out)


class kmeans(KMeans):
    def __init__(
        self,
        data,
        max_k=8,
        min_k=2,
        init="k-means++",
        n_clusters=8,
        n_init=10,
        max_iter=300,
        tol=0.0001,
        precompute_distances="auto",
        verbose=0,
        random_state=None,
        copy_x=True,
        n_jobs=None,
        algorithm="auto",
    ):
        KMeans.__init__(
            self,
            n_clusters=8,
            n_init=10,
            max_iter=300,
            tol=0.0001,
            precompute_distances="auto",
            verbose=0,
            random_state=None,
            copy_x=True,
            n_jobs=None,
            algorithm="auto",
        )
        self.data = data
        self.k_range = range(min_k, max_k + 1)
        self.km_results = {k: {} for k in self.k_range}
        for k in tqdm(self.k_range):
            kms = KMeans(n_clusters=k)
            self.km_results[k]["cluster_labels"] = kms.fit_predict(self.data)
            self.km_results[k]["centers"] = kms.cluster_centers_
            self.km_results[k]["inertia"] = kms.inertia_

    def draw_elbow(self):
        SSE = []
        for k in self.k_range:
            SSE.append(self.km_results[k]["inertia"])
        plt.figure(figsize=(8, 8))
        plt.plot(self.k_range, SSE, "-x")
        plt.xlabel("Number of clusters k")
        plt.ylabel("Sum of squared distance")

    def calc_silhouette(self):
        for k in self.k_range:
            preds = self.km_results[k]["cluster_labels"]
            centers = self.km_results[k]["centers"]

            score = silhouette_score(self.data, preds)
            print(f"For n_clusters = {k}, silhouette score is {score}")

    def plot_silhouette(self):
        """Code is taken from https://scikit-
        learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html"""

        for n_clusters in self.k_range:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(self.data) + (n_clusters + 1) * 10])

            cluster_labels = self.km_results[n_clusters]["cluster_labels"]

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(self.data, cluster_labels)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(self.data, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[
                    cluster_labels == i
                ]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(
                self.data[:, 0],
                self.data[:, 1],
                marker=".",
                s=30,
                lw=0,
                alpha=0.7,
                c=colors,
                edgecolor="k",
            )

            # Labeling the clusters
            centers = self.km_results[n_clusters]["centers"]
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                (
                    "Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters
                ),
                fontsize=14,
                fontweight="bold",
            )

    def k_means_montage(self, df, class_col):
        """Montage for all shoes by class"""

        n_classes = df[class_col].nunique()
        for cl in sorted(df[class_col].unique()):
            montage_df = df[df[class_col] == cl].path
            imgs = [np.array(Image.open(img)) for img in montage_df]
            imgs = np.stack(imgs)
            plt.figure(figsize=(12, 15))
            plt.imshow(montage(imgs, multichannel=True).astype(np.uint8))
            plt.title(f"Montage for Class{cl}")
