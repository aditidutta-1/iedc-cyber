import numpy as np
import matplotlib.pyplot as plt

def activation(x):
    return np.maximum(0, x)  # ReLU

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def GCN_ResNet_CrossTransformer():
    print("Step 1: GCN-ResNet + Transformer")

    modalities = ['trend', 'proximity', 'period']
    num_nodes = 4
    input_dim = 4
    hidden_dim = 4
    L = 2  # GCN layers
    h = 2  # Attention heads

    # Dummy adjacency matrix and degree matrix
    A = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ])
    D = np.diag(np.sum(A, axis=1))

    # Dummy input for each modality
    features = {
        mod: np.random.rand(num_nodes, input_dim) for mod in modalities
    }

    # Dummy weights
    Wmod = {mod: [np.random.rand(input_dim, hidden_dim) for _ in range(L)] for mod in modalities}
    WQ = [np.random.rand(hidden_dim, hidden_dim) for _ in range(h)]
    WK = [np.random.rand(hidden_dim, hidden_dim) for _ in range(h)]
    WV = [np.random.rand(hidden_dim, hidden_dim) for _ in range(h)]
    WO = np.random.rand(h * hidden_dim, hidden_dim)

    def FFN(x):
        return activation(x @ np.random.rand(x.shape[1], hidden_dim))

    Hmod_final = {}

    # GCN + ResNet
    for mod in modalities:
        H = features[mod]
        Hprev = H.copy()
        for l in range(L):
            A_hat = np.linalg.inv(np.sqrt(D)) @ A @ np.linalg.inv(np.sqrt(D))
            H_G = activation(A_hat @ H @ Wmod[mod][l])
            H = activation(H_G + Hprev)
            Hprev = H
        Hmod_final[mod] = H

    # Multi-Head Attention
    MultiHead = {}
    for mod in modalities:
        heads = []
        H = Hmod_final[mod]
        for i in range(h):
            Q = H @ WQ[i]
            K = H @ WK[i]
            V = H @ WV[i]
            attn_scores = softmax(Q @ K.T / np.sqrt(H.shape[1]))
            head = attn_scores @ V
            heads.append(head)
        MultiHead[mod] = np.concatenate(heads, axis=-1) @ WO

    # Cross-modality Fusion
    Hconcat = np.concatenate([MultiHead[m] for m in modalities], axis=-1)
    Hspatial = FFN(Hconcat)
    print("Hspatial (Final Spatial Embedding):\n", Hspatial)
    return Hspatial


def Attention_Based_ConvLSTM():
    print("\nStep 2: Attention-Based ConvLSTM")

    T = 5  # time steps
    H, W, C = 4, 4, 1  # height, width, channels

    # Random temporal data: X[t] is a 4x4x1 feature map
    X = [np.random.rand(H, W, C) for _ in range(T)]

    # Dummy attention weights
    We = np.random.rand(T)

    # Dummy LSTM state
    h_t = np.zeros((H, W, C))
    c_t = np.zeros((H, W, C))
    outputs = []

    for t in range(T):
        x_t = X[t]
        f_t = sigmoid(x_t)
        i_t = sigmoid(x_t)
        c_tilde = np.tanh(x_t)
        c_t = f_t * c_t + i_t * c_tilde
        o_t = sigmoid(x_t)
        h_t = o_t * np.tanh(c_t)
        outputs.append(h_t)

    outputs = np.stack(outputs, axis=0)
    attn_scores = softmax(We.reshape(T, 1, 1, 1))  # Apply attention
    context_vector = np.sum(attn_scores * outputs, axis=0)
    print("Htemporal (Temporal Context Embedding):\n", context_vector)
    return context_vector


def plot_heatmap(data, title, filename, xlabel="Features", ylabel="Nodes"):
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data[:, :, 0]
    plt.figure(figsize=(6, 4))
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    print("Running real.py...\n")

    Hspatial = GCN_ResNet_CrossTransformer()
    plot_heatmap(Hspatial, "Hspatial - Spatial Embedding Heatmap", "hspatial_heatmap.png")

    Htemporal = Attention_Based_ConvLSTM()
    plot_heatmap(Htemporal, "Htemporal - Temporal Embedding Heatmap", "htemporal_heatmap.png", xlabel="Width", ylabel="Height")

    print("\n=== Finished Running ===")


