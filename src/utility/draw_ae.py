"""
Autoencoder Neural Network Graph Visualization
---------------------------------------------

This script visualizes an autoencoder architecture with encoder, latent space,
and decoder using NetworkX and Matplotlib. Weight matrices are labeled correctly.

Author      : Raj Singh
Affiliation : Ostfalia University of Applied Sciences
Date        : 11 Feb 2026
"""

import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams.update({
    "text.usetex": True
})


def draw_autoencoder(x=3, h_enc=[4, 3], latent=2, h_dec=[3, 4], figsize=(12, 6)):
    """
    x       : Number of input neurons
    h_enc   : List with sizes of encoder hidden layers
    latent  : Number of latent space neurons
    h_dec   : List with sizes of decoder hidden layers
    """

    G = nx.DiGraph()

    # ---------- Create nodes ----------
    x_nodes = [rf"$x_{i+1}$" for i in range(x)]
    latent_nodes = [rf"$z_{i+1}$" for i in range(latent)]
    h_enc_nodes = []
    h_dec_nodes = []
    output_nodes = [rf"$\hat{{x}}_{i+1}$" for i in range(x)]

    # Encoder hidden nodes
    for i, size in enumerate(h_enc):
        for j in range(size):
            h_enc_nodes.append(rf"$a_{j+1}^{{enc{i+1}}}$")

    # Decoder hidden nodes
    for i, size in enumerate(h_dec):
        for j in range(size):
            h_dec_nodes.append(rf"$a_{j+1}^{{dec{i+1}}}$")

    # Add all nodes
    G.add_nodes_from(x_nodes + h_enc_nodes + latent_nodes + h_dec_nodes + output_nodes)

    # ---------- Add edges ----------
    # Input → Encoder hidden
    for i in x_nodes:
        for j in h_enc_nodes[:h_enc[0]]:
            G.add_edge(i, j)

    # Encoder hidden → Encoder hidden (layer-wise)
    for layer in range(len(h_enc)-1):
        curr_start = sum(h_enc[:layer])
        next_start = sum(h_enc[:layer+1])
        for i in range(h_enc[layer]):
            for j in range(h_enc[layer+1]):
                G.add_edge(h_enc_nodes[curr_start + i], h_enc_nodes[next_start + j])

    # Last encoder hidden → latent
    last_enc_start = sum(h_enc[:-1])
    for i in range(h_enc[-1]):
        for j in latent_nodes:
            G.add_edge(h_enc_nodes[last_enc_start + i], j)

    # Latent → Decoder hidden
    for i in latent_nodes:
        for j in h_dec_nodes[:h_dec[0]]:
            G.add_edge(i, j)

    # Decoder hidden → Decoder hidden (layer-wise)
    for layer in range(len(h_dec)-1):
        curr_start = sum(h_dec[:layer])
        next_start = sum(h_dec[:layer+1])
        for i in range(h_dec[layer]):
            for j in range(h_dec[layer+1]):
                G.add_edge(h_dec_nodes[curr_start + i], h_dec_nodes[next_start + j])

    # Last decoder hidden → output
    last_dec_start = sum(h_dec[:-1])
    for i in range(h_dec[-1]):
        for j in output_nodes:
            G.add_edge(h_dec_nodes[last_dec_start + i], j)

    # ---------- Node positions ----------
    pos = {}

    # Input layer
    for i, node in enumerate(x_nodes):
        pos[node] = (0, (x - 1) / 2 - i)

    # Encoder hidden layers
    k = 0
    for layer, size in enumerate(h_enc):
        for j in range(size):
            pos[h_enc_nodes[k]] = (layer + 1, (size - 1) / 2 - j)
            k += 1

    # Latent layer
    for i, node in enumerate(latent_nodes):
        pos[node] = (len(h_enc) + 1, (latent - 1) / 2 - i)

    # Decoder hidden layers
    k = 0
    for layer, size in enumerate(h_dec):
        for j in range(size):
            pos[h_dec_nodes[k]] = (len(h_enc) + 2 + layer, (size - 1) / 2 - j)
            k += 1

    # Output layer
    for i, node in enumerate(output_nodes):
        pos[node] = (len(h_enc) + len(h_dec) + 2, (x - 1) / 2 - i)

    # ---------- Colors ----------
    node_colors = []
    for node in G.nodes():
        if node in x_nodes or node in h_enc_nodes:
            node_colors.append("lightgreen")       # Encoder
        elif node in latent_nodes:
            node_colors.append("plum")             # Latent space
        elif node in h_dec_nodes or node in output_nodes:
            node_colors.append("skyblue")          # Decoder / output

    # ---------- Draw ----------
    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        node_color=node_colors,
        font_size=12,
        arrows=True,
        edge_color="black"
    )

    # ---------- Layer-wise weight annotation ----------
    layers = [x] + h_enc + [latent] + h_dec + [x]
    total_weights = len(layers) - 1  # weights = connections between layers
    for i in range(total_weights):
        plt.text(
            i + 0.3,
            max(layers) / 2,
            rf"$\mathbf{{W}}^{{{i+1}}}$",
            fontsize=12
        )

    plt.axis("off")
    plt.savefig("./images/autoencoder.svg", format="svg", bbox_inches="tight")
    plt.show()


# Example usage
#draw_autoencoder(x=3, h_enc=[3, 2], latent=2, h_dec=[2, 3], figsize=(12, 6))
