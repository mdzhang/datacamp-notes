# Unsupervised Learning

## Clustering

- **clustering**: discovering the underlying groups (or "clusters") in a dataset
- **centroid**: mean of each cluster

### Algorithms

- **k-Means**
    - tries to minimize **inertia**
    - can graph # clusters (x-axis) against inertia (y-axis)
    - rule of thumb to choose the # clusters corresponding to the "elbow" (where inertia decreases more slowly) in the inertia plot (looks like an asymptote in quadrant I)
    - feature variance = feature influence
        - should **normalize** or **standardize** feature data e.g. w/ `sklearn.preprocessing.StandardScaler` to improve clustering performance

##### Example of creating a model that can assign clusters to data points:

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# can visualize data with a scatterplot to determine # clusters to use
model = KMeans(n_clusters=3)
model.fit(samples)

labels = model.predict(samples)

# kmeans will remember the mean of each cluster i.e. centroids
# can add new samples to existing clusters

model.fit(new_samples)

print(mode.inertia_)

# graph sepal v petal length and color based on cluster labels
xs = samples[:,0]
ys = samples[:,2]

plt.scatter(xs, ys, c=labels)
plt.show()

# graph cluster centroids
centroids = model.cluster_centers_

centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# mark centroids with a diamond of size 50
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()
```

##### Example of standardizing data before fitting a model

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
kmeans = KMeans(n_clusters=4)

# create pipeline to apply scaler to feature data before fitting kmeans model
pipeline = make_pipeline(scaler, kmeans)

labels = pipeline.fit_predict(samples)
```

#### Measuring performance / Evaluating a clustering

- a good clustering has
  - **tight clusters** i.e. close together, not spread out
  - not too many clusters
- **inertia** measures how spread out clusters are by finding distance of sample from centroid

```python
import pandas as pd

# taken from preceding code snippet
labels = ...
species = samples[:,1]
df = pd.DataFrame({'labels': labels, 'species': species})

ct = pd.crosstab(df['labels'], df['species'])
#  species        versicolor      setosa   virginica
#  labels
#  0                       2          60          10
#  1                       0           1          60
#  2                      68           9           0
```

### Visualizing clusters

- **hierarchical clustering**
    - arranges samples into a hierarchy of clusters e.g. `(animals, (mammals, reptiles))`
    - often visualized with a **dendrogram**
        - x-axis are lowest level cluster names
        - y-axis aka height is distance between merging clusters
        - distance calculated with a **linkage method**
            - **complete linkage** is max distance between cluster samples i.e. distance between furthest points of clusters
            - **single linkage**: distance between closest points in clusters
    - **agglomerative hierarchical clustering** start with many clusters
      - at each step, 2 closest clusters are merged
      - continue until a single cluster remains
    - **divisive hierarchical clustering** does the reverse

    - code sample for creating a dendrogram
        ```python
        from sklearn.cluster.hierarchy import linkage, dendrogram
        import matplotlib.pyplot as plt

        mergings = linkage(samples, method='complete')

        dendrogram(mergings,
                 labels=varieties,
                 leaf_rotation=90,
                 leaf_font_size=6,
        )
        plt.show()
        ```

    - instead of arranging a hierarchy of clusters until there is a single cluster, you can use **intermediate clustering** by stopping clustering at a height `h`
        - aka don't cluster if distance between clusters >= `h`

      ```python
      from scipy.cluster.hierarchy import fcluster

      # use intermediate clustering and stop clustering at height = 6
      labels = fcluster(mergings, 6, criterion='distance')
      ```
- **t-SNE (t-distributed stochastic neighbor embedding)**: used to map multidimensional samples to 2/3D space for easier visualization
    - approximately preserves distances between samples
    - never generates exact same graph 2x
    - axes are ???

    - code sample
        ```python
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # normally choose learning rate betwee 50, 200
        model = TSNE(learning_rate=100)
        # no separate fit/transform methods
        # cannot dynamically add additional data to fit on
        transformed = model.fit_transform(samples)

        xs = transformed[:,0]
        ys = transformed[:,1]

        plt.scatter(xs, ys, c=species)
        # for x, y, company in zip(xs, ys, companies):
        #     plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
        plt.show()
        ```

## Dimension Reduction

- finds patterns in data
- uses patterns to re-express data in compressed form
  - more efficient storage and computation
  - remove "noisy" features
- many techniques
  - PCA
  - NMF

- **PCA (principal component analysis)** is a 2 step dimension reduction process
  - **principle components**: directions in which samples vary the most
  - 1st step is **decorrelation**
      - rotates features so they align with axes
          - this means that features are no longer linearly related
          - specifically aligns principle components
      - shifts them to have mean 0 (subtract mean mu from each sample)
  - 2nd step reduces dimension


  ```python
  from sklearn.decomposition import PCA

  # can optionally specify # components to keep; will keep 2 features w/ highest variance
  # good rule of thumb is to use intrinsice dimension as n_components
  model = PCA(n_components=2)
  model.fit(samples)

  transformed = model.transform(samples)
  ```

  - **intrinsic dimension**: # features needed to approximate dataset
    - can use scatter plots in 2/3 dimensions to visualize intrinsice dimension
      - e.g. if points lie on a sheet in 3D space, than has intrinsice dimension 2
    - PCA can identify intrinsic dimension for k features; intrinsic dimension is # PCA features w/ significant variance (e.g > 0.05)

  ```python
  # continued from above

  features = range(pca.n_components_)
  plt.bar(features, pca.explained_variance_)
  plt.xticks(features)
  plt.ylabel(variance)
  plt.xlabel('PCA feature')
  plt.show()
  ```

  - instead of `sklearn.decomposition.PCA` can also use `sklearn.decomposition.TruncatedSVD` which has the same interface, but accepts numpy csr_matrices, which are good for sparse matrices since they only remember non-0 entries

- **NMF (non-negative matrix factorization)**
    - easier to interpret than PCA
    - features must be non-negative >= 0
    - has components like PCA
        - dimension of components == dimension of samples
        - also non-negative
        - in images, components are parts of an image (e.g. segments of numbers in LED digital display)
        - in documents, components are topics e.g. frequently occurring words
    - can approximately reconstruct sample by multiplying each feature by component vector and summing result

  ```python
  from sklearn.decomposition import NMF

  model = NMF(n_components=2)
  model.fit(samples)

  nmf_features = model.transform(samples)

  print(model.components_)
  ```

- **cosine similarity**
  - angle between lines
  - higher value means more similarity
  - max value 1 when angle == 0 aka same line
  - can use for recommender systems to find similar content

  ```python
  import pandas as pd
  from sklearn.preprocessing import normalize

  norm_features = normalize(nmf_features)
  # index is article titles
  df = pd.DataFrame(norm_features, index=titles)

  # has 6 elements, 1 for each feature from nmf_features
  article = df.loc['Cristiano Ronaldo']
  # b/c cos(theta) = dot product(matrix A, matrix B) if matrices are normalized
  similarities = df.dot(article)

  # display articles w/ largest cosine similarity to ronaldo article
  print(similarities.nlargest())
  ```
