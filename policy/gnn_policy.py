import torch.nn as nn
import torch.nn.functional as F
import torch


def preprocess(A):
    # Get size of the adjacency matrix
    size = len(A)
    # Get the degrees for each node
    degrees = []
    for node_adjaceny in A:
        num = 0
        for node in node_adjaceny:
            if node == 1.0:
                num = num + 1
        # Add an extra for the "self loop"
        num = num + 1
        degrees.append(num)
    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(degrees)
    # Cholesky decomposition of D
    D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Turn adjacency matrix into a numpy matrix
    A = np.matrix(A)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    return A_hat, D


class GnnPolicy(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super().__init__()
        import pdb; pdb.set_trace()

        hidden_layers = [64, 32, 16, 8]
        n_layers = len(hidden_layers)

        # Number of hidden layers to use in the "m" networks from Equation 3.
        m_hidden = 8

		self.conv_weights = nn.ParameterList([
                nn.Parameter(torch.randn(hidden_layers[i], hidden_layers[i+1]))
                for i in range(n_layers - 1)
            ])

        self.conv_weights_critic = nn.ParameterList([
                nn.Parameter(torch.randn(hidden_layers[i], hidden_layers[i+1]))
                for i in range(n_layers - 1)
            ])

        # The networks described in Equation 3.
        self.m_f = self.Linear(hidden_layers[-1], m_hidden)
        self.m_s = self.Linear(hidden_layers[-1], m_hidden)
        self.m_e = self.Linear(hidden_layers[-1], m_hidden)
        self.m_t = self.Linear(hidden_layers[-1], m_hidden)

        # Our critic function
        self.critic = self.Linear(hidden_layers[-1], 1)
        self.softmax = nn.Softmax()

    def __gcn_pass(self, inputs, weights):
        cur_h = inputs

        for conv_weight in weights:
            # Equation 2 of the paper.
            cur_h = F.relu(D.mm(A).mm(D).mm(cur_h).mm(conv_weight))

        return cur_h

    def forward(self, adj_matrix):
        A, D = preprocess(adj_matrix)
        A = torch.tensor(A)
        D = torch.tensor(D)
        adj_matrix = torch.tensor(adj_matrix)

        # Represents the current hidden node representation.
        # According to the original GCN paper this just starts as the adjacency
        # matrix.
        X = __gcn_pass(adj_matrix, self.conv_weights)

        X_first = self.m_f(X)
        X_second = self.m_s(torch.cat(X_first, X))
        X_edge = self.m_e(torch.cat(X_first, X_second))
        X_stop = self.m_t(torch.sum(X))

        X_first = self.softmax(X_first)
        X_second = self.softmax(X_second)
        X_edge = self.softmax(X_edge)
        X_stop = self.softmax(X_stop)

        action = torch.cat([X_first, X_second, X_edge, X_stop])

        critic_X = __gcn_pass(adj_matrix, self.conv_weights_critic)
        critic = self.critic(critic_X)

        return action, critic
