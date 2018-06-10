## Network Analysis

- **networks** model **relationships** between **entities**
    - can be used to:
      - find important nodes e.g. **influencers** in a social network
      - pathfinding between nodes
          - optimizing shortest path
          - model spread of disease/information
      - clustering e.g. finding communities
    - entities are **nodes**, relationships are **edges**
      - both can have associated metadata

- **NetworkX** is a python library for manipulating/analyzing graph data

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_nodes_from([1, 2, 3])
G.nodes()

G.add_edge(1, 2)
G.edges()
G.has_edge(1, 2)

node = 1
G.add_edges_from(zip([node] * len(G.neighbors(node)), G.neighbors(node)))

# see what neighbors node 1 has
G.neighbors(1)

# add node metadata
G.node[1]['label'] = 'blue'
# returns 2-element tuples where 1st element is node, 2nd element is dict of metadata
G.nodes(data=True)

nx.draw(G)

plt.show()
```

## Types of Graphs

- **undirected graphs**: no directionality associated w/ edges
    - `G = nx.Graph` is undirected
    - `D = nx.DiGraph` is directed

- if can have multiple edges between same nodes, is a **multi graph**
    - `M = nx.MultiGraph()`
    - `MD = nx.MultiDiGraph()`
    - having multiple edges per node pair can be computationally expensive, may want to collapse into single edge

- barbell graphs: have 2 clusters of nodes connected by bridge
  - nodes in a cluster are connected to all other nodes in that cluster i.e. are **cliques**
  - bridge is a series of 1+ nodes connected to no other nodes but each other (in a single path) and the barbells (for nodes on either end of the path)
  - `nx.barbell_graph(m1=5, m2=1)` where `m1` is # nodes in a cluster, `m2` is # nodes in the bridge

- **bipartite graphs**: graph that is partitioned into 2 sets where nodes in a set may not be connected to e/o, they may only be connected to nodes in the other set
    - bipartite information encoded into node metadata using bipartite keyword

    ```python
    import networkx as nx

    G = nx.Graph()
    # creates nodes with metadata dict that has key-value pair {'bipartite': 'customers'}
    G.add_nodes_from(nodes, bipartite='customers')
    ```

## Graph Algorithms

- **breadth first search (BFS)**

## Visualizing Networks

- **matrix plot**: `nv.MatrixPlot`
    - rows/columns are indexed by node
        - in **bipartite** graph, rows and columns for nodes in different partitions
        ```python
        bi_mat = nx.bipartite.biadjacency_matrix(G, row_nodes=part1_nodes, column_order=part2_nodes)
        # @ operator does dot product
        # .T has matrix transpose (swap columns and nodes)
        mat = bi_mat @ bi_mat.T

        # find max # connections
        diag = mat.diagonal()
        indices = np.where(diag == diag.max())[0]

        # find nodes w/ most shared connections
        mat.setdiag(0)
        coo = mat.tocoo()
        indices = np.where(coo.data == coo.data.max())[0]
        for idx in indices:
            print('- {0}, {1}'.format(part1_nodes[coo.row[idx]], part1_nodes[coo.col[idx]]))
        ```
    - value corresponds to edge
- **arc plot**: `nv.ArcPlot`
    - nodes are along one, ordered axis
        - good for grouping nodes by common attribute e.g. age
        - good for visualizing relationship between sorted property and connectivity
    - edges are curves between edges graphed in the same quadrant
- **circos plot**: `nv.CircosPlot`
    - like an arc plot, but instead of a single linear axis, axis is a circle

```python
import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt

# generate a graph that has 20 nodes and a 20% probability of having an edge between nodes
G = nx.erdos_renyi_graph(n=20, p=0.2)

ap = nv.ArcPlot(G)
ap.draw()
plt.show()

# to color nodes by their 'category' metadata field
# and sort them along the axis by the same field
ap2 = ArcPlot(G, node_order='category', node_color='category')
```

To visualize a subgraph:

```python
import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt

# generate a graph that has 20 nodes and a 20% probability of having an edge between nodes
G = nx.erdos_renyi_graph(n=20, p=0.2)

nodes = G.neighbors(8) + 8

G_eight = G.subgraph(nodes)

nx.draw(G_eight, with_labels=True)
plt.show()
```


## ?

- methods for determining which nodes are important
  - **degree centrality**: # neighbors a node has / total # nodes in a graph
      - if self loops not allowed, would be total # nodes - 1
      - `nx.degree_centrality(G)` returns dict of nodes => centrality score (doesn't consider self loops)
      - **degree** of a node = # neighbors node has: `nx.degree(G, n)`
      - in a **bipartite graph**: # neighbors a node has / total # nodes in the other partition in the graph
          - `nx.bipartite.degree_centrality(G, cust_nodes)` where `cust_nodes` is filtered to partition of nodes
  - **betweenness centrality**: # shortest paths through a node / all possible shortest paths
      - all shortest paths = shortest path between every pair of nodes in a graph
      - captures bottleneck nodes instead of highly connected nodes
      - `nx.betweenness_centrality(G)`

## Structures in Graphs

- **clique**: set of nodes where each node is connected to every other node
  - simplest **clique** is 2 nodes with an edge
  - simplest **complex clique** is 3 nodes connected as a triangle
  - **maximal clique**: a clique that, when extended by any one node in a graph, is no longer a clique
      - `nx.find_cliques` finds maximal cliques

      ```python
      from itertools import combinations
      for n1, n2 in combinations(G.neighbors(n), 2):
          if G.has_edge(n1, n2):
              ...
      ```

- **connected components**: subgraphs in a graph where each subgraph has connected nodes but the subgraphs aren't connected to e/o
  - `nx.connected_component_subgraphs(G)`

- **projection**: unipartite representation of bipartite connectivity
  - "projection" of a graph onto one of its partitions is the connectivity of the nodes in that partition conditioned on connections to nodes on the other partition
  - `nx.bipartite.projected_graph(G, bipartite_partition_nodes)`

## Persisting Graphs

- use CSVs
  - human readable, can use `pandas`
  - but, highly connected nodes have repetitive entries, use up disk space

- flat edge lists
    ```
    <node1_name> <node2_name> <metadata_as_dict>
    Michelle.Zhang lanthanide {'weight': 1}
    ```

    ```python
    import networkx as nx
    G = nx.read_edgelist('filename.txt')
    ```

- CSV files:
  - node list + metadata
      - each row is a node
      - each column has metadata for that node
  - edge list + metadata
      - each row is a edge
      - each column has metadata for that edge

  ```python
  nodelist = []
  for n, d in G.nodes(data=True):
      node_data = dict()
      # uniquely identifying column
      node_data['node'] = n
      node_data.update(d)
      nodelist.append(node_data)
  df = pd.DataFrame(nodelist)
  df.to_csv('nodelist.csv')
  ```

## Recommendation Systems

- find triangles (i.e. simplest **complex clique**) in a unipartite graph where 1 edge is missing and recommend nodes for the missing edge to e/o
- bipartite
  - suppose users 1, 2, 3 all work on repo 2, but user 3 also works on repo 1; recommend repo 1 to users 1 and 2
  - use intersection of set of repos user 1 contributes to, set of repos user 3 contributes to to get overlapping repo2
  - use diffence(user 3 repos, user 1 repos) to get repos to recommend to user 1

## Graph Evolution

- graphs can change over time e.g. edges between nodes, node set
    `nx.difference(G1, G2)` assumes node set has stayed the same ; what edges are in `G1` that _aren't_ in `G2`

- graph summary statistics
  - simple metrics use edgelist data
    - # nodes
    - # edges
  - graph theoretic metrics use full graph object
    - degree distribution
    - centrality distribution
    - shortest path distribution
