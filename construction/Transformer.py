import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock

class ConditionTransformer(nn.Module):

    def __init__(self, cross_attention_dim, num_heads):
        super().__init__()

        head_dim = cross_attention_dim // num_heads

        # Formation information transformer
        self.transformer1 = BasicTransformerBlock(
            dim=cross_attention_dim,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            cross_attention_dim=cross_attention_dim,
        )
        # Boreholes transformer
        self.transformer2 = BasicTransformerBlock(
            dim=cross_attention_dim,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            cross_attention_dim=cross_attention_dim,
        )
        # Quaternary transformer
        self.transformer3 = BasicTransformerBlock(
            dim=cross_attention_dim,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            cross_attention_dim=cross_attention_dim,
        )

    def forward(self, X, embedded_info, encoded_boreholes, encoded_quaternary):

        X = self.transformer1(X, encoder_hidden_states=embedded_info)
        X = self.transformer2(X, encoder_hidden_states=encoded_boreholes)
        X = self.transformer3(X, encoder_hidden_states=encoded_quaternary)

        return X