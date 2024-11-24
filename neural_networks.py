import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)


# Activation functions and their derivatives
def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)


def tanh(Z):
    return np.tanh(Z)


def tanh_derivative(Z):
    return 1 - np.tanh(Z) ** 2


# MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_name = activation.lower()
        if self.activation_name == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
            # He initialization for ReLU
            self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        elif self.activation_name == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
            # Xavier initialization for sigmoid
            self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1. / input_dim)
        elif self.activation_name == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
            # Xavier initialization for tanh
            self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1. / input_dim)
        else:
            raise ValueError("Unsupported activation function")

        self.b1 = np.zeros((1, hidden_dim))
        # Xavier initialization for hidden to output
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1. / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        # To store gradients
        self.grads = {
            'W1': np.zeros_like(self.W1),
            'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2),
            'b2': np.zeros_like(self.b2)
        }

    def forward(self, X):
        # Input to hidden
        self.Z1 = np.dot(X, self.W1) + self.b1  # Shape: (n_samples, hidden_dim)
        self.A1 = self.activation(self.Z1)  # Shape: (n_samples, hidden_dim)

        # Hidden to output
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Shape: (n_samples, output_dim)
        self.A2 = sigmoid(self.Z2)  # Shape: (n_samples, output_dim)

        return self.A2

    def compute_loss(self, Y, A2):
        # Binary cross-entropy loss
        m = Y.shape[0]
        # To avoid log(0)
        epsilon = 1e-15
        A2 = np.clip(A2, epsilon, 1 - epsilon)
        loss = - (1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
        return loss

    def backward(self, X, Y):
        m = Y.shape[0]

        # Output layer gradients
        dZ2 = self.A2 - Y  # Shape: (n_samples, output_dim)
        self.grads['W2'] = np.dot(self.A1.T, dZ2) / m  # Shape: (hidden_dim, output_dim)
        self.grads['b2'] = np.sum(dZ2, axis=0, keepdims=True) / m  # Shape: (1, output_dim)

        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)  # Shape: (n_samples, hidden_dim)
        dZ1 = dA1 * self.activation_derivative(self.Z1)  # Shape: (n_samples, hidden_dim)
        self.grads['W1'] = np.dot(X.T, dZ1) / m  # Shape: (input_dim, hidden_dim)
        self.grads['b1'] = np.sum(dZ1, axis=0, keepdims=True) / m  # Shape: (1, hidden_dim)

        # Update weights and biases
        self.W2 -= self.lr * self.grads['W2']
        self.b2 -= self.lr * self.grads['b2']
        self.W1 -= self.lr * self.grads['W1']
        self.b1 -= self.lr * self.grads['b1']

    def get_gradients(self):
        # Return a copy of gradients
        return {
            'W1': self.grads['W1'].copy(),
            'b1': self.grads['b1'].copy(),
            'W2': self.grads['W2'].copy(),
            'b2': self.grads['b2'].copy()
        }


def generate_data(n_samples=200):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # 0 or 1
    y = y.reshape(-1, 1)
    return X, y


# Function to plot gradient graph
def plot_gradient_graph(ax, gradients):
    ax.clear()
    ax.set_title("Gradients at Step")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.4)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.41, 0.2))
    ax.grid(True)

    # Define node positions
    node_positions = {
        'x1': (0.2, 1.2),
        'x2': (0.2, 0.4),
        'h1': (0.5, 1.2),
        'h2': (0.5, 0.8),
        'h3': (0.5, 0.4),
        'y': (0.8, 0.8)
    }
    nodes = ['x1', 'x2', 'h1', 'h2', 'h3', 'y']

    # Plot nodes
    for node in nodes:
        x, y_pos = node_positions[node]
        ax.scatter(x, y_pos, s=300, color='blue', edgecolors='k', zorder=3)
        ax.text(x, y_pos + 0.05, node, horizontalalignment='center', verticalalignment='bottom', fontsize=10, zorder=4)

    # Define edges with corresponding gradients
    edges = [
        ('x1', 'h1'),
        ('x1', 'h2'),
        ('x1', 'h3'),
        ('x2', 'h1'),
        ('x2', 'h2'),
        ('x2', 'h3'),
        ('h1', 'y'),
        ('h2', 'y'),
        ('h3', 'y')
    ]

    # Collect all gradient magnitudes
    grad_magnitudes = []
    edge_grad_map = {}
    for edge in edges:
        src, dst = edge
        if src.startswith('x') and dst.startswith('h'):
            # Weights from input to hidden
            idx = int(src[1:]) - 1  # 'x1'->0, 'x2'->1
            hid = {'h1': 0, 'h2': 1, 'h3': 2}[dst]
            grad = np.abs(gradients['W1'][idx, hid])
        elif src.startswith('h') and dst == 'y':
            # Weights from hidden to output
            hid = {'h1': 0, 'h2': 1, 'h3': 2}[src]
            grad = np.abs(gradients['W2'][hid, 0])
        else:
            grad = 0  # For any other connections that don't exist
        grad_magnitudes.append(grad)
        edge_grad_map[edge] = grad

    # Normalize gradient magnitudes for thickness
    max_grad = max(grad_magnitudes) if grad_magnitudes else 1
    min_grad = min(grad_magnitudes) if grad_magnitudes else 0
    for edge in edges:
        grad = edge_grad_map[edge]
        # Normalize between 1 and 5
        thickness = 1 + 4 * (grad - min_grad) / (max_grad - min_grad + 1e-8)
        src, dst = edge
        x1, y1 = node_positions[src]
        x2, y2 = node_positions[dst]
        ax.plot([x1, x2], [y1, y2], linewidth=thickness, color='purple')

    ax.set_aspect('equal')



# Function to plot hidden space
# 全局变量
grid_size = 20  # 网格密度


# 修改后的 plot_hidden_space 函数
def plot_hidden_space(ax, hidden_features, y, step, xlim, ylim, zlim, grid_hidden=None):
    ax.clear()
    ax.set_title(f"Hidden Space at Step {step}")
    ax.set_xlabel('h1')
    ax.set_ylabel('h2')
    ax.set_zlabel('h3')

    # Set fixed axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Fix the view angle
    ax.view_init(elev=30, azim=45)

    # Plot the transformed grid as a surface
    if grid_hidden is not None:
        H1 = grid_hidden[:, 0].reshape((grid_size, grid_size))
        H2 = grid_hidden[:, 1].reshape((grid_size, grid_size))
        H3 = grid_hidden[:, 2].reshape((grid_size, grid_size))
        ax.plot_surface(H1, H2, H3, color='lightblue', alpha=0.3, linewidth=0)

    # Scatter plot
    ax.scatter(hidden_features[y.ravel() == 0, 0],
               hidden_features[y.ravel() == 0, 1],
               hidden_features[y.ravel() == 0, 2],
               color='blue', label='Class 0', alpha=0.6)
    ax.scatter(hidden_features[y.ravel() == 1, 0],
               hidden_features[y.ravel() == 1, 1],
               hidden_features[y.ravel() == 1, 2],
               color='red', label='Class 1', alpha=0.6)

    # Fit a plane (hyperplane) to separate the classes in hidden space
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(hidden_features, y.ravel())

    # Create grid to plot the hyperplane
    h1_range = np.linspace(xlim[0], xlim[1], grid_size)
    h2_range = np.linspace(ylim[0], ylim[1], grid_size)
    H1_plane, H2_plane = np.meshgrid(h1_range, h2_range)

    # Compute H3 based on the plane equation
    w = clf.coef_[0]
    b = clf.intercept_[0]
    # Avoid division by zero
    if np.abs(w[2]) < 1e-4:
        return  # Skip plotting the plane if w[2] is too small
    H3_plane = (-w[0]*H1_plane - w[1]*H2_plane - b) / w[2]

    # Optionally, remove clipping to allow the hyperplane to extend beyond zlim
    # H3_plane = np.clip(H3_plane, zlim[0], zlim[1])

    # Plot the plane
    ax.plot_surface(H1_plane, H2_plane, H3_plane, color='orange', alpha=0.5)

    ax.legend()


def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, step_num, xlim, ylim, zlim, step_per_frame=10):
    current_step = frame * step_per_frame
    for _ in range(step_per_frame):
        mlp.forward(X)
        mlp.backward(X, y)
        current_step += 1
        if current_step >= step_num:
            break

    # Get hidden features
    hidden_features = mlp.A1  # Shape: (n_samples, hidden_dim)

    # Generate grid in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx_grid, yy_grid = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                                   np.linspace(y_min, y_max, grid_size))
    grid_input = np.c_[xx_grid.ravel(), yy_grid.ravel()]

    # Map grid points to hidden space
    mlp.forward(grid_input)
    grid_hidden = mlp.A1  # Shape: (grid_size*grid_size, hidden_dim)

    # Get gradients
    gradients = mlp.get_gradients()

    # Plot hidden space with transformed grid and decision hyperplane
    plot_hidden_space(ax_hidden, hidden_features, y, current_step, xlim, ylim, zlim, grid_hidden=grid_hidden)

    # Plot input space
    plot_input_space(ax_input, mlp, X, y, current_step)

    # Plot gradient graph
    plot_gradient_graph(ax_gradient, gradients)

    print(f"Completed step {current_step}")

    # Stop the animation if current_step exceeds step_num
    if current_step >= step_num:
        plt.close()



# 修改后的 visualize 函数
def visualize(activation, lr, step_num):
    X, y = generate_data(n_samples=200)
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Forward pass before training to get initial hidden features
    mlp.forward(X)
    hidden_features = mlp.A1

    # Manually set axis limits
    xlim = (-1.2, 1.2)
    ylim = (-1.2, 1.2)
    zlim = (-1.2, 1.2)

    # Set up visualization
    matplotlib.use('agg')  # Use non-interactive backend
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Total frames
    step_per_frame = 10
    total_frames = (step_num + step_per_frame - 1) // step_per_frame  # Ensure covering all steps

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                                     ax_gradient=ax_gradient, X=X, y=y, step_num=step_num, xlim=xlim, ylim=ylim,
                                     zlim=zlim, step_per_frame=step_per_frame),
                        frames=total_frames, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=5)
    plt.close()


# Function to plot input space decision boundary
def plot_input_space(ax, mlp, X, y, step):
    ax.clear()
    ax.set_title(f"Input Space at Step {step}")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.forward(grid)
    probs = probs.reshape(xx.shape)

    # Plot the decision boundary
    contour = ax.contourf(xx, yy, probs, levels=50, cmap='bwr', alpha=0.6)
    # plt.colorbar(contour, ax=ax)

    # Contour line for probability=0.5
    ax.contour(xx, yy, probs, levels=[0.5], colors='k', linewidths=1)

    # Scatter plot of the data points
    ax.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1],
               color='blue', label='Class 0', edgecolors='k', alpha=0.6)
    ax.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1],
               color='red', label='Class 1', edgecolors='k', alpha=0.6)

    ax.legend()


# Visualization update function


if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
