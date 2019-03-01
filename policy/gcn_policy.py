import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def preprocess(A):
    # Get size of the adjacency matrix
    #size = len(A)
    # Get the degrees for each node
    #degrees = []
    #for node_adjaceny in use_A:
    #    num = 0
    #    for node in node_adjaceny:
    #        if node == 1.0:
    #            num = num + 1
    #    # Add an extra for the "self loop"
    #    num = num + 1
    #    degrees.append(num)
    degrees = torch.sum(A, dim=1)
    degrees += 1

    # Create diagonal matrix D from the degrees of the nodes
    D = torch.stack([torch.diag(d) for d in degrees])
    # Cholesky decomposition of D
    D = torch.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    D = torch.inverse(D)
    # Create an identity matrix of size x size
    I = torch.eye(A.size(1))

    if A.is_cuda:
        I = I.cuda()
        D = D.cuda()

    # Create A hat
    A_hat = A + I
    # Return A_hat
    return A_hat, D


class GcnPolicy(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super().__init__()

        hidden_layers = [input_shape['adj'].shape[1], 20, 16, 8]
        n_layers = len(hidden_layers)

        # Number of hidden layers to use in the "m" networks from Equation 3.
        m_hidden = input_shape['node'].shape[1]

        self.conv_weights = nn.ParameterList([
                nn.Parameter(torch.randn(hidden_layers[i], hidden_layers[i+1]))
                for i in range(n_layers - 1)
            ])

        self.conv_weights_critic = nn.ParameterList([
                nn.Parameter(torch.randn(hidden_layers[i], hidden_layers[i+1]))
                for i in range(n_layers - 1)
            ])

        # The networks described in Equation 3.
        self.m_f = nn.Linear(hidden_layers[-1], m_hidden)
        self.m_s = nn.Linear(hidden_layers[-1] + m_hidden, m_hidden)
        self.m_e = nn.Linear(m_hidden + m_hidden, m_hidden)
        self.m_t = nn.Linear(hidden_layers[-1], 1)

        # Our critic function
        self.critic = nn.Linear(hidden_layers[-1], 1)
        self.softmax = nn.Softmax(dim=-1)

    def __gcn_pass(self, A, D, inputs, weights):
        cur_h = inputs

        for conv_weight in weights:
            # Equation 2 of the paper.
            cur_h = F.relu(D.bmm(A).bmm(D).bmm(cur_h).matmul(conv_weight))
            # Equation 2 also includes an aggregation function
            #cur_h = torch.sum(cur_h, dim=1)

        return cur_h

    def forward(self, adj_matrix):
        adj_matrix = adj_matrix.squeeze(1)
        A, D = preprocess(adj_matrix)

        # Represents the current hidden node representation.
        # According to the original GCN paper this just starts as the adjacency
        # matrix.
        X = self.__gcn_pass(A, D, adj_matrix, self.conv_weights)
        # Not sure if this should go here.
        X = torch.sum(X, dim=1)

        X_first = self.m_f(X)
        X_second = self.m_s(torch.cat([X_first, X], dim=-1))
        X_edge = self.m_e(torch.cat([X_first, X_second], dim=-1))
        X_stop = self.m_t(X)

        X_first = self.softmax(X_first)
        X_second = self.softmax(X_second)
        X_edge = self.softmax(X_edge)
        X_stop = self.softmax(X_stop)
        import pdb; pdb.set_trace()

        action = torch.cat([X_first, X_second, X_edge, X_stop])

        critic_X = __gcn_pass(adj_matrix, self.conv_weights_critic)
        critic = self.critic(critic_X)

        return action, critic
