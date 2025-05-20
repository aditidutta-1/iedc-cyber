import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def Attention_Based_ConvLSTM(T=5, H=4, W=4, C=1):
    print("Running Attention-Based ConvLSTM...\n")

    # Dummy temporal spatial data: sequence of 4x4x1 feature maps
    X = [np.random.rand(H, W, C) for _ in range(T)]

    # Random attention weights for each time step
    attention_weights = np.random.rand(T)
    attention_scores = softmax(attention_weights)

    # Dummy ConvLSTM weights (simplified to identity behavior)
    h_t = np.zeros((H, W, C))
    c_t = np.zeros((H, W, C))
    outputs = []

    for t in range(T):
        x_t = X[t]
        f_t = sigmoid(x_t)       # Forget gate
        i_t = sigmoid(x_t)       # Input gate
        c_tilde = np.tanh(x_t)   # Candidate memory
        c_t = f_t * c_t + i_t * c_tilde
        o_t = sigmoid(x_t)       # Output gate
        h_t = o_t * np.tanh(c_t)
        outputs.append(h_t)

    outputs = np.stack(outputs, axis=0)  # Shape: (T, H, W, C)

    # Apply attention over time
    attention_scores = attention_scores.reshape(T, 1, 1, 1)
    context_vector = np.sum(attention_scores * outputs, axis=0)

    print("Context vector (Htemporal) shape:", context_vector.shape)
    return context_vector, outputs, attention_scores


def plot_temporal_embedding(context_vector, attention_scores):
    # Remove channel dimension for 2D plotting
    if context_vector.shape[-1] == 1:
        context_vector = context_vector[:, :, 0]

    plt.figure(figsize=(5, 5))
    plt.imshow(context_vector, cmap='magma', aspect='auto')
    plt.colorbar()
    plt.title("Htemporal - Temporal Context Embedding")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.savefig("acl_temporal_heatmap.png")
    plt.show()

    # Optional: plot attention scores over time
    plt.figure(figsize=(6, 2))
    plt.bar(range(len(attention_scores)), attention_scores.flatten(), color='skyblue')
    plt.title("Attention Scores over Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.savefig("acl_attention_scores.png")
    plt.show()


if __name__ == "__main__":
    Htemporal, outputs, attention_scores = Attention_Based_ConvLSTM()
    plot_temporal_embedding(Htemporal, attention_scores)
    print("\n=== Finished Attention-Based ConvLSTM ===")
