import torch
import torch.nn as nn

embedding_dim = 50


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.dropout = nn.Dropout(p=0.2)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)

        user_embeds = self.dropout(user_embeds)
        item_embeds = self.dropout(item_embeds)

        user_b = self.user_bias(user_indices).squeeze()
        item_b = self.item_bias(item_indices).squeeze()

        dot_product = (user_embeds * item_embeds).sum(1)

        output = dot_product + user_b + item_b
        return torch.sigmoid(output) * 5.5
