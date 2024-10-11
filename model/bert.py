import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size = 30522, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        
        self.output_projection = nn.Linear(hidden, 768)

    def forward(self, x, segment_info, attention_mask):
        """
        :param x: Input token IDs (batch_size, seq_len)
        :param segment_info: Segment info (token_type_ids) indicating sentence pairs.
        :param attention_mask: Attention mask (1 for real tokens, 0 for padding).
        """

        # Embed the indexed sequence to sequence of vectors, using segment info for sentence separation
        x = self.embedding(x, segment_info)

        # Attention mask processing
        # The mask shape should be [batch_size, 1, 1, seq_len], suitable for transformer attention
        mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Passing the sequence through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        # Projecting the output to the required hidden size (768 dimensions)
        x = self.output_projection(x)

        return x


