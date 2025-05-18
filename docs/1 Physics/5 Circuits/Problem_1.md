# Problem 1

# Sector 1: Problem Understanding and Setup

---

## Graph Representation

To analyze the circuit as a graph, we define:

- **Nodes ($V$)** as the junction points of the circuit.
- **Edges ($E$)** as the resistors connecting nodes.
- Each edge $e \in E$ has an associated **weight** $w(e)$ representing the resistance value $R$ of the resistor.

Thus, the circuit can be modeled as an undirected weighted graph:
$$
G = (V, E, w)
$$
where
- $V = \{v_1, v_2, \ldots, v_n\}$,
- $E = \{e_{ij} | v_i \text{ connected to } v_j\}$,
- $w : E \to \mathbb{R}^+$ such that $w(e_{ij}) = R_{ij} > 0$.

---

## Input Format Specification

The input graph can be represented in several ways, depending on the use case:

1. **Adjacency List**: For each node $v_i$, list all adjacent nodes $v_j$ along with weights $R_{ij}$.
   $$
   \text{AdjacencyList}(v_i) = \{ (v_j, R_{ij}) \mid (v_i, v_j) \in E \}
   $$
2. **Edge List with Weights**: A list of tuples $(v_i, v_j, R_{ij})$.
   $$
   E = \{ (v_i, v_j, R_{ij}) \mid i,j = 1, \ldots, n \}
   $$
3. **Adjacency Matrix ($\mathbf{A}$)**: A matrix of size $n \times n$ where each element $A_{ij}$ corresponds to the resistance value $R_{ij}$ if nodes are connected, or zero otherwise:
   $$
   A_{ij} = \begin{cases}
   R_{ij} & \text{if } (v_i, v_j) \in E, \\
   0 & \text{otherwise}.
   \end{cases}
   $$

---

## Graph Properties

We assume the following properties for the circuit graph:

- **Undirected edges**: The graph $G$ is undirected, so
  $$
  (v_i, v_j) \in E \implies (v_j, v_i) \in E,
  $$
  and resistance weights are symmetric:
  $$
  R_{ij} = R_{ji}.
  $$

- **Positive resistance weights**:
  $$
  R_{ij} > 0, \quad \forall (v_i, v_j) \in E.
  $$

- **Connected graph**: There exists at least one path between every pair of nodes $v_i$ and $v_j$, ensuring the graph is connected:
  $$
  \forall v_i, v_j \in V, \quad \exists \text{ path } P = (v_i, \ldots, v_j).
  $$

---

## Task Scope Selection

Given the problem scope, choose one of the following options:

- **Option 1:** Develop detailed pseudocode and describe the algorithmic approach for analyzing the circuit graph.
  
- **Option 2:** Provide a full implementation including input parsing, graph construction, and testing with sample circuit graphs.


---

# Sector 2: Algorithm Design

---

## Design Series Detection

To identify **series resistor chains**, we look for nodes with **degree 2** (connected to exactly two edges):

- Let $v_k \in V$ be a node such that
  $$
  \deg(v_k) = 2,
  $$
  where $\deg(v_k)$ is the degree of node $v_k$.

- The two edges connected to $v_k$ are $e_{ik} = (v_i, v_k)$ and $e_{kj} = (v_k, v_j)$ with resistances $R_{ik}$ and $R_{kj}$ respectively.

- Since these resistors are in series, they can be **reduced** to a single equivalent resistor:
  $$
  R_{\text{series}} = R_{ik} + R_{kj}.
  $$

- The series reduction replaces edges $(v_i, v_k)$ and $(v_k, v_j)$ by a single edge $(v_i, v_j)$ with weight $R_{\text{series}}$, and removes node $v_k$ from $V$.

---

## Design Parallel Detection

To detect **parallel resistor connections**, identify multiple edges connecting the same pair of nodes:

- For nodes $v_i, v_j \in V$, suppose there are $m$ parallel edges:
  $$
  \{e_{ij}^1, e_{ij}^2, \ldots, e_{ij}^m\},
  $$
  with resistances
  $$
  R_{ij}^1, R_{ij}^2, \ldots, R_{ij}^m.
  $$

- The equivalent resistance of parallel resistors is given by the reciprocal sum:
  $$
  \frac{1}{R_{\text{parallel}}} = \sum_{k=1}^m \frac{1}{R_{ij}^k}.
  $$

- The parallel edges are replaced by a single edge $(v_i, v_j)$ with weight $R_{\text{parallel}}$.

- Additionally, parallel connections may appear as small cycles, requiring cycle detection algorithms.

---

## Outline Reduction Strategy

The **graph reduction** algorithm iteratively applies series and parallel reductions:

1. **Initialize** with graph $G = (V, E, w)$.
2. **While** the graph has more than one edge:
   - Detect series nodes ($\deg = 2$) and reduce.
   - Detect parallel edges and reduce.
3. **Repeat** until $G$ reduces to a single edge between two nodes representing the equivalent resistance $R_{\text{eq}}$.

Formally,
$$
G^{(0)} = G, \quad G^{(t+1)} = \text{Reduce}(G^{(t)}),
$$
with termination condition
$$
|E^{(t)}| = 1.
$$

---

## Plan for Nested Configurations

Nested series-parallel configurations require careful tracking:

- Use **recursive reduction** or **stack-based tracking** to handle nested subgraphs.
- Maintain mapping of original nodes and edges to reduced edges to preserve circuit integrity.
- Ensure reduction respects:
  $$
  R_{\text{eq}} = \text{series/parallel composition of nested resistors}.
  $$

---

## Define Termination Condition

The reduction process ends when:

- The graph is simplified to a **single edge** $e_{\text{final}}$ connecting two terminal nodes $v_s$ and $v_t$:
  $$
  G_{\text{final}} = ( \{v_s, v_t\}, \{e_{\text{final}}\} ),
  $$
  where $w(e_{\text{final}}) = R_{\text{eq}}$.

---

## Select Traversal Method

Choose a traversal technique to identify reducible patterns systematically:

- **Depth-First Search (DFS)**:
  - Useful for exploring cycles and nested subgraphs.
  - Efficient for detecting series chains by node degree inspection.

- **Breadth-First Search (BFS)**:
  - Useful for level-order traversal, detecting parallel connections early.

In this context, **DFS** is generally preferred to explore connected components and nested structures:

$$
\text{DFS}(v) : \text{explore all adjacent nodes recursively from } v.
$$

---

## Code and Plots

![alt text](<indir (7)circuit.gif>)

```python
# Install necessary packages for Colab
!pip install networkx matplotlib pillow -q

import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

# Create folder for frames
if os.path.exists("frames"):
    shutil.rmtree("frames")
os.makedirs("frames")

# Step-by-step simplification colors
colors = ["red", "green", "blue", "purple", "orange"]

# Define the initial circuit graph
def create_initial_circuit():
    G = nx.Graph()
    G.add_edges_from([
        ("B+", "R1"),
        ("R1", "R2"),
        ("R2", "R3"),
        ("R3", "R4"),
        ("R2", "R5"),
        ("R3", "R6"),
        ("R5", "R6"),
        ("R4", "B-")
    ])
    return G

# Layout for reproducibility
pos_layout = {
    "B+": (0, 1),
    "R1": (1, 1),
    "R2": (2, 1),
    "R5": (2, 0),
    "R6": (3, 0),
    "R3": (3, 1),
    "R4": (4, 1),
    "B-": (5, 1)
}

# Draw and save a frame of the circuit
def draw_circuit(G, highlight_edges=[], highlight_color="red", step=0, label=""):
    plt.figure(figsize=(8, 4))
    edge_colors = [highlight_color if e in highlight_edges or (e[1], e[0]) in highlight_edges else 'black' for e in G.edges()]
    nx.draw(G, pos=pos_layout, with_labels=True, node_color='lightgrey', edge_color=edge_colors,
            node_size=1000, font_weight='bold')
    plt.title(f"Step {step}: {label}")
    plt.axis('off')
    plt.savefig(f"frames/frame_{step:02d}.png")
    plt.close()

# Start simplification process
G = create_initial_circuit()

# Step 0: Original
draw_circuit(G, step=0, label="Original Circuit")

# Step 1: Combine R5 and R6 (parallel) -> R56
G = nx.contracted_nodes(G, "R5", "R6", self_loops=False)
G = nx.relabel_nodes(G, {"R5": "R56"})
pos_layout["R56"] = (2.5, 0)
pos_layout.pop("R6")
draw_circuit(G, highlight_edges=[("R2", "R56"), ("R56", "R3")], highlight_color=colors[0], step=1, label="Parallel: R5 and R6 -> R56")

# Step 2: Combine R2 and R56 (series) -> R256
G = nx.contracted_nodes(G, "R2", "R56", self_loops=False)
G = nx.relabel_nodes(G, {"R2": "R256"})
pos_layout["R256"] = (2.25, 0.75)
pos_layout.pop("R56")
draw_circuit(G, highlight_edges=[("R1", "R256"), ("R256", "R3")], highlight_color=colors[1], step=2, label="Series: R2 and R56 -> R256")

# Step 3: Combine R1 and R256 (series) -> R1256
G = nx.contracted_nodes(G, "R1", "R256", self_loops=False)
G = nx.relabel_nodes(G, {"R1": "R1256"})
pos_layout["R1256"] = (1.5, 1)
pos_layout.pop("R256")
draw_circuit(G, highlight_edges=[("B+", "R1256"), ("R1256", "R3")], highlight_color=colors[2], step=3, label="Series: R1 and R256 -> R1256")

# Step 4: Combine R3 and R4 (series) -> R34
G = nx.contracted_nodes(G, "R3", "R4", self_loops=False)
G = nx.relabel_nodes(G, {"R3": "R34"})
pos_layout["R34"] = (3.5, 1)
pos_layout.pop("R4")
draw_circuit(G, highlight_edges=[("R1256", "R34"), ("R34", "B-")], highlight_color=colors[3], step=4, label="Series: R3 and R4 -> R34")

# Step 5: Final Combine R1256 and R34 -> Rfinal
G = nx.contracted_nodes(G, "R1256", "R34", self_loops=False)
G = nx.relabel_nodes(G, {"R1256": "Rfinal"})
pos_layout["Rfinal"] = (2.5, 1)
pos_layout.pop("R34")
draw_circuit(G, highlight_edges=[("B+", "Rfinal"), ("Rfinal", "B-")], highlight_color=colors[4], step=5, label="Final: R1256 and R34 -> Rfinal")

# Create GIF from frames
frames = [Image.open(f"frames/{frame}") for frame in sorted(os.listdir("frames")) if frame.endswith(".png")]
frames[0].save("circuit_simplification.gif", format='GIF', append_images=frames[1:], save_all=True, duration=1000, loop=0)

# Clean up frame directory
shutil.rmtree("frames")

# Display the GIF in Colab
from IPython.display import Image as IPyImage
IPyImage(filename="circuit_simplification.gif")


```

[Colab](https://colab.research.google.com/drive/1ofbvHbGM1Aj-FYEa6BG6h38_kcEYVSJ2)