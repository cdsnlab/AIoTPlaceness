import pickle
import networkx as nx

# Path to the Pickle file (update with the actual file path)
pickle_file = "../Dataset/AirBnB_Graph/20170101.gpickle"

# Load the graph from the Pickle file
with open(pickle_file, 'rb') as f:
    G = pickle.load(f)

# Print basic information about the graph
print("Graph Info:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Print all nodes
print("\nNodes:")
print(list(G.nodes()))

# Print all edges with weights
print("\nEdges:")
for edge in G.edges(data=True):
    print(edge)
