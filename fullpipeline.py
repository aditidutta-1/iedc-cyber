import numpy as np
import matplotlib.pyplot as plt

### UTILITIES ###
def sigmoid(x): return 1 / (1 + np.exp(-x))
def softmax(x): return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
def relu(x): return np.maximum(0, x)

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

def plot_bar(values, title, filename, xlabel="Time Step", ylabel="Attention Weight"):
    plt.figure(figsize=(6, 2.5))
    plt.bar(range(len(values)), values, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

### ALGORITHM 1: GCN-ResNet + Multi-Head Attention ###
def GCN_ResNet_CrossTransformer():
    modalities = ['trend', 'proximity', 'period']
    num_nodes, input_dim, hidden_dim = 4, 4, 4
    L, h = 2, 2  # GCN layers, attention heads

    A = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ])
    D = np.diag(np.sum(A, axis=1))

    features = {mod: np.random.rand(num_nodes, input_dim) for mod in modalities}
    Wmod = {mod: [np.random.rand(input_dim, hidden_dim) for _ in range(L)] for mod in modalities}
    WQ = [np.random.rand(hidden_dim, hidden_dim) for _ in range(h)]
    WK = [np.random.rand(hidden_dim, hidden_dim) for _ in range(h)]
    WV = [np.random.rand(hidden_dim, hidden_dim) for _ in range(h)]
    WO = np.random.rand(h * hidden_dim, hidden_dim)

    def FFN(x): return relu(x @ np.random.rand(x.shape[1], hidden_dim))

    Hmod_final = {}
    for mod in modalities:
        H = features[mod]
        Hprev = H.copy()
        for l in range(L):
            A_hat = np.linalg.inv(np.sqrt(D)) @ A @ np.linalg.inv(np.sqrt(D))
            H_G = relu(A_hat @ H @ Wmod[mod][l])
            H = relu(H_G + Hprev)
            Hprev = H
        Hmod_final[mod] = H

    MultiHead = {}
    for mod in modalities:
        heads = []
        H = Hmod_final[mod]
        for i in range(h):
            Q, K, V = H @ WQ[i], H @ WK[i], H @ WV[i]
            attn = softmax((Q @ K.T) / np.sqrt(H.shape[1]))
            heads.append(attn @ V)
        MultiHead[mod] = np.concatenate(heads, axis=-1) @ WO

    Hconcat = np.concatenate([MultiHead[m] for m in modalities], axis=-1)
    Hspatial = FFN(Hconcat)
    return Hspatial

### ALGORITHM 2: Attention-Based ConvLSTM ###
def Attention_Based_ConvLSTM(T=5, H=4, W=4, C=1):
    X = [np.random.rand(H, W, C) for _ in range(T)]
    We = np.random.rand(T)
    attn_scores = softmax(We)

    h_t, c_t = np.zeros((H, W, C)), np.zeros((H, W, C))
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
    attn_scores = attn_scores.reshape(T, 1, 1, 1)
    Htemporal = np.sum(attn_scores * outputs, axis=0)

    return Htemporal, attn_scores.flatten()

### MAIN PIPELINE ###
def run_full_pipeline():
    print("\n=== Running Full Pipeline ===")

    # Step 1: Spatial Embedding
    Hspatial = GCN_ResNet_CrossTransformer()
    print("Hspatial Shape:", Hspatial.shape)
    plot_heatmap(Hspatial, "Hspatial - Spatial Embedding", "hspatial_pipeline.png")

    # Step 2: Temporal Embedding
    Htemporal, attn_weights = Attention_Based_ConvLSTM()
    print("Htemporal Shape:", Htemporal.shape)
    plot_heatmap(Htemporal, "Htemporal - Temporal Context Embedding", "htemporal_pipeline.png")
    plot_bar(attn_weights, "Attention Weights Over Time", "temporal_attention_weights.png")

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    run_full_pipeline()
