import torch
import torch.nn as nn


class RoleAssignmentLSTM(nn.Module):
    def __init__(self, num_roles, filler_embedding_dim, num_fillers, hidden_dim,
                 role_embedding_dim, num_layers=1):
        super(RoleAssignmentLSTM, self).__init__()
        # OPTION this LSTM could share the embedding for filler_i with the TensorProductEncoder
        # TODO: when we move to language models, we will need to use pre-trained word embeddings.
        # See embedder_squeeze in TensorProductEncoder
        self.filler_embedding = nn.Embedding(num_fillers, filler_embedding_dim)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_roles = num_roles

        # OPTION we may want the LSTM to be bidirectional for things like RTL roles.
        # Also, should the output size be the number of roles for the weight vector?
        # Or is the output of variable size and we apply a linear transformation
        # to get the weight vector?
        self.lstm = nn.LSTM(filler_embedding_dim, hidden_dim)
        self.role_weight_predictions = nn.Linear(hidden_dim, num_roles)
        # The output of role_weight_predictions is shape (sequence_length, batch_size, num_roles)
        # We want to softmax across the roles so set dim=2
        self.softmax = nn.Softmax(dim=2)

        self.role_embedding = nn.Embedding(num_roles, role_embedding_dim)
        self.role_indices = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def forward(self, filler_tensor):
        """
        :param filler_tensor: This input tensor should be of shape (batch_size, sequence_length)
        :return: A tensor of size (sequence_length, batch_size, role_embedding_dim) with the role
            embeddings for the input filler_tensor.
        """
        batch_size = len(filler_tensor)
        hidden = self.init_hidden(batch_size)

        fillers_embedded = self.filler_embedding(filler_tensor)
        # The shape of fillers_embedded should be
        # (batch_size, sequence_length, filler_embedding_dim)
        # Pytorch LSTM expects data in the shape (sequence_length, batch_size, feature_dim)
        fillers_embedded = torch.transpose(fillers_embedded, 0, 1)

        lstm_out, hidden = self.lstm(fillers_embedded, hidden)
        role_predictions = self.role_weight_predictions(lstm_out)
        role_predictions = self.softmax(role_predictions)
        # role_predictions is size (sequence_length, batch_size, num_roles)

        role_embeddings = self.role_embedding(self.role_indices)
        # role_embeddings is size (num_roles, role_embedding_dim)

        roles = torch.matmul(role_predictions, role_embeddings)
        # roles is size (sequence_length, batch_size, role_embedding_dim)

        return roles

    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, batch_size, hidden_dim)
        # We need a tuple for the hidden state and the cell state of the LSTM.
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))


if __name__ == "__main__":
    num_roles = 10
    filler_embedding_dim = 20
    num_fillers = 10
    hidden_dim = 30
    role_embedding_dim = 20
    lstm = RoleAssignmentLSTM(
        num_roles,
        filler_embedding_dim,
        num_fillers,
        hidden_dim,
        role_embedding_dim
    )

    import torch

    data = [[1, 2, 3, 4], [1, 8, 1, 0]]
    data_tensor = torch.tensor(data)

    out = lstm(data_tensor)
