import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleHeadAttention(nn.Module):
    """
    Single-head self-attention mechanism implementation.

    This module implements the core attention mechanism as described in "Attention Is All You Need".
    It computes attention weights between all positions in a sequence and applies them to values.

    Attributes:
        d_model (int): The dimensionality of the input embeddings and output.
        d_k (int): The dimensionality of the key vectors (equals d_model for single head).
        w_q (nn.Linear): Linear transformation for query vectors.
        w_k (nn.Linear): Linear transformation for key vectors.
        w_v (nn.Linear): Linear transformation for value vectors.

    Example:
        >>> attention = SingleHeadAttention(d_model=512)
        >>> x = torch.randn(32, 100, 512)  # batch_size=32, seq_len=100, d_model=512
        >>> output, weights = attention(x, x, x)
        >>> print(output.shape)  # torch.Size([32, 100, 512])
        >>> print(weights.shape)  # torch.Size([32, 100, 100])
    """

    def __init__(self, d_model: int):
        """
        Initialize the single-head attention module.

        Args:
            d_model: The dimensionality of the input embeddings and output vectors.
                    Must be a positive integer.

        Raises:
            ValueError: If d_model is not a positive integer.
        """
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")

        self.d_model = d_model
        self.d_k = d_model  # For single head, key dimension equals model dimension

        # Linear transformations for query, key, and value projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention output and attention weights.

        This method implements the scaled dot-product attention mechanism:
        1. Project inputs to query, key, and value spaces
        2. Compute attention scores using scaled dot-product
        3. Apply optional masking
        4. Compute attention weights using softmax
        5. Apply attention weights to values

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
                   Represents what the model is looking for.
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
                 Represents what information is available.
            value: Value tensor of shape (batch_size, seq_len_k, d_model).
                   Represents what information should be returned.
            mask: Optional attention mask of shape (batch_size, seq_len_q, seq_len_k).
                  Values of 0 indicate positions to mask (set to -inf).
                  Values of 1 indicate positions to attend to.
                  If None, no masking is applied.

        Returns:
            tuple: A tuple containing:
                - output: Attention output tensor of shape (batch_size, seq_len_q, d_model)
                - attention_weights: Attention weight tensor of shape (batch_size, seq_len_q, seq_len_k)

        Raises:
            ValueError: If input tensors have incompatible shapes or d_model dimensions.
            RuntimeError: If mask shape is incompatible with input shapes.

        Note:
            The attention weights sum to 1 along the last dimension for each position.
            This ensures that the attention mechanism is properly normalized.
            For cross-attention, seq_len_q (query length) and seq_len_k (key length) can differ.
        """
        # Validate input shapes
        if query.size(-1) != self.d_model or key.size(-1) != self.d_model or value.size(-1) != self.d_model:
            raise ValueError(
                f"Input tensors must have last dimension equal to d_model ({self.d_model})")

        # For cross-attention, query and key/value can have different sequence lengths
        # but must have the same batch size
        if query.size(0) != key.size(0) or query.size(0) != value.size(0):
            raise ValueError(
                "Query, key, and value must have the same batch size")

        if key.size(1) != value.size(1):
            raise ValueError(
                "Key and value must have the same sequence length")

        # Project inputs to query, key, and value spaces
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Compute attention scores: Q * K^T / sqrt(d_k)
        # Result shape: (batch_size, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply masking if provided
        if mask is not None:
            # Ensure mask is broadcastable to (batch_size, n_heads, seq_len_q, seq_len_k)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights using softmax
        # Shape: (batch_size, seq_len_q, seq_len_k)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        # Shape: (batch_size, seq_len_q, d_model)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism implementation.

    This module implements multi-head attention by running multiple attention mechanisms
    in parallel and concatenating their outputs. Each head can focus on different
    aspects of the input sequence, allowing the model to attend to information from
    different representation subspaces at different positions.

    Attributes:
        d_model (int): The dimensionality of the input embeddings and output.
        n_heads (int): The number of attention heads.
        d_k (int): The dimensionality of each head's key vectors (d_model // n_heads).
        w_q (nn.Linear): Linear transformation for query vectors.
        w_k (nn.Linear): Linear transformation for key vectors.
        w_v (nn.Linear): Linear transformation for value vectors.
        w_o (nn.Linear): Output projection layer.
        dropout (nn.Dropout): Dropout layer for attention weights.

    Example:
        >>> attention = MultiHeadAttention(d_model=512, n_heads=8)
        >>> x = torch.randn(32, 100, 512)  # batch_size=32, seq_len=100, d_model=512
        >>> output, weights = attention(x, x, x)
        >>> print(output.shape)  # torch.Size([32, 100, 512])
        >>> print(weights.shape)  # torch.Size([32, 8, 100, 100])
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize the multi-head attention module.

        Args:
            d_model: The dimensionality of the input embeddings and output vectors.
                    Must be a positive integer.
            n_heads: The number of attention heads. Must be a positive integer
                    that divides d_model evenly.
            dropout: Dropout probability for attention weights. Must be in [0, 1).

        Raises:
            ValueError: If d_model is not divisible by n_heads, or if parameters
                       are outside valid ranges.
        """
        super().__init__()

        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(n_heads, int) or n_heads <= 0:
            raise ValueError("n_heads must be a positive integer")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Linear transformations for query, key, and value projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection layer
        self.w_o = nn.Linear(d_model, d_model)

        # Dropout layer for attention weights
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention output and attention weights.

        This method implements the multi-head attention mechanism:
        1. Project inputs to query, key, and value spaces
        2. Reshape tensors to separate heads
        3. Compute attention for each head in parallel
        4. Concatenate head outputs
        5. Apply final linear projection

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model).
                   Represents what the model is looking for.
            key: Key tensor of shape (batch_size, seq_len_k, d_model).
                 Represents what information is available.
            value: Value tensor of shape (batch_size, seq_len_k, d_model).
                   Represents what information should be returned.
            mask: Optional attention mask of shape (batch_size, seq_len_q, seq_len_k).
                  Values of 0 indicate positions to mask (set to -inf).
                  Values of 1 indicate positions to attend to.
                  If None, no masking is applied.

        Returns:
            tuple: A tuple containing:
                - output: Attention output tensor of shape (batch_size, seq_len_q, d_model)
                - attention_weights: Attention weight tensor of shape 
                  (batch_size, n_heads, seq_len_q, seq_len_k)

        Raises:
            ValueError: If input tensors have incompatible shapes or d_model dimensions.
            RuntimeError: If mask shape is incompatible with input shapes.

        Note:
            Each head computes attention independently, allowing the model to focus
            on different types of relationships in the input sequence.
            For cross-attention, seq_len_q (query length) and seq_len_k (key length) can differ.
        """
        batch_size, seq_len_q = query.size(0), query.size(1)
        seq_len_k = key.size(1)

        # Validate input shapes
        if query.size(-1) != self.d_model or key.size(-1) != self.d_model or value.size(-1) != self.d_model:
            raise ValueError(
                f"Input tensors must have last dimension equal to d_model ({self.d_model})")

        # For cross-attention, query and key/value can have different sequence lengths
        # but must have the same batch size
        if query.size(0) != key.size(0) or query.size(0) != value.size(0):
            raise ValueError(
                "Query, key, and value must have the same batch size")

        if key.size(1) != value.size(1):
            raise ValueError(
                "Key and value must have the same sequence length")

        # Project inputs to query, key, and value spaces
        Q = self.w_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.w_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.w_v(value)  # (batch_size, seq_len_k, d_model)

        # Reshape to separate heads: (batch_size, seq_len, n_heads, d_k)
        # Then transpose to: (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len_q, self.n_heads,
                   self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads,
                   self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.n_heads,
                   self.d_k).transpose(1, 2)

        # Compute attention for each head
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask)

        # Concatenate head outputs: (batch_size, seq_len_q, d_model)
        # attention_output shape: (batch_size, n_heads, seq_len_q, d_k)
        # Ensure we maintain the correct 3D structure
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, seq_len_q, self.d_model)

        # Apply final linear projection
        output = self.w_o(attention_output)

        return output, attention_weights

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                     mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention for multi-head attention.

        This method implements the core attention computation for each head:
        1. Compute attention scores using scaled dot-product
        2. Apply optional masking
        3. Compute attention weights using softmax
        4. Apply dropout to attention weights
        5. Apply attention weights to values

        Args:
            Q: Query tensor of shape (batch_size, n_heads, seq_len_q, d_k).
            K: Key tensor of shape (batch_size, n_heads, seq_len_k, d_k).
            V: Value tensor of shape (batch_size, n_heads, seq_len_k, d_k).
            mask: Optional attention mask of shape (batch_size, seq_len_q, seq_len_k).
                  Will be expanded to (batch_size, 1, seq_len_q, seq_len_k) for broadcasting.
                  Values of 0 indicate positions to mask (set to -inf).
                  Values of 1 indicate positions to attend to.
                  If None, no masking is applied.

        Returns:
            tuple: A tuple containing:
                - output: Attention output tensor of shape (batch_size, n_heads, seq_len_q, d_k)
                - attention_weights: Attention weight tensor of shape 
                  (batch_size, n_heads, seq_len_q, seq_len_k)

        Note:
            The scaling factor sqrt(d_k) prevents the dot products from growing too large
            in magnitude, which would push the softmax function into regions with small gradients.
            For cross-attention, seq_len_q (query length) and seq_len_k (key length) can differ.
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.

    This module adds positional information to input embeddings since the attention
    mechanism has no inherent sense of sequence order. The positional encoding uses
    sine and cosine functions of different frequencies to encode position information
    in a way that allows the model to generalize to sequence lengths not seen during training.

    The encoding is computed as:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Attributes:
        d_model (int): The dimensionality of the input embeddings.
        max_seq_length (int): The maximum sequence length for which positional
                             encodings are pre-computed.
        pe (torch.Tensor): Pre-computed positional encoding tensor of shape
                          (max_seq_length, d_model).

    Example:
        >>> pos_encoding = PositionalEncoding(d_model=512, max_seq_length=1000)
        >>> embeddings = torch.randn(100, 32, 512)  # seq_len=100, batch_size=32, d_model=512
        >>> encoded = pos_encoding(embeddings)
        >>> print(encoded.shape)  # torch.Size([100, 32, 512])

    Reference:
        Vaswani, A., et al. "Attention is all you need." Advances in neural information
        processing systems 30 (2017).
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000):
        """
        Initialize the positional encoding module.

        Args:
            d_model: The dimensionality of the input embeddings. Must be a positive integer.
            max_seq_length: The maximum sequence length for which positional encodings
                           are pre-computed. Must be a positive integer.

        Raises:
            ValueError: If d_model or max_seq_length are not positive integers.

        Note:
            The positional encoding is pre-computed and stored as a buffer to avoid
            recomputation during forward passes. This improves efficiency but limits
            the maximum sequence length that can be processed.
        """
        super().__init__()

        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(max_seq_length, int) or max_seq_length <= 0:
            raise ValueError("max_seq_length must be a positive integer")

        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Pre-compute positional encoding
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(
            0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Compute frequency terms for sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape for broadcasting: (max_seq_length, d_model) -> (1, max_seq_length, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer (not a parameter, so it won't be updated during training)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        This method adds the pre-computed positional encoding to the input embeddings.
        The positional encoding is added element-wise, providing the model with
        information about the position of each token in the sequence.

        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model).
               The sequence length must not exceed max_seq_length.

        Returns:
            torch.Tensor: Tensor of shape (seq_len, batch_size, d_model) with
                         positional encoding added to the input embeddings.

        Raises:
            ValueError: If the input tensor has incorrect shape or the sequence
                       length exceeds max_seq_length.

        Note:
            The positional encoding is added directly to the embeddings without
            any learnable parameters. This allows the model to learn to use
            positional information effectively during training.
        """
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Input tensor must have last dimension equal to d_model ({self.d_model})")

        if x.size(0) > self.max_seq_length:
            raise ValueError(
                f"Sequence length ({x.size(0)}) exceeds max_seq_length ({self.max_seq_length})")

        # Add positional encoding to input embeddings
        return x + self.pe[:x.size(0), :]


def test_single_head_attention():
    """
    Test the SingleHeadAttention module with basic functionality.

    This function creates a simple test case to verify that the SingleHeadAttention
    module works correctly. It checks input/output shapes, attention weight normalization,
    and basic forward pass functionality.

    Returns:
        tuple: A tuple containing the output tensor and attention weights tensor
               from the test forward pass.

    Raises:
        AssertionError: If any of the test conditions are not met.

    Note:
        This function is intended for development and debugging purposes.
        It should not be used in production code.
    """
    batch_size, seq_len, d_model = 2, 4, 8

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize attention module
    attention = SingleHeadAttention(d_model)

    # Perform forward pass
    output, weights = attention(x, x, x)  # Self-attention

    # Verify output shapes
    assert output.shape == (
        batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert weights.shape == (
        batch_size, seq_len, seq_len), f"Expected weights shape {(batch_size, seq_len, seq_len)}, got {weights.shape}"

    # Verify attention weights sum to 1 along the last dimension
    assert weights.sum(dim=-1).allclose(torch.ones(batch_size,
                                                   seq_len)), "Attention weights must sum to 1"

    return output, weights


def test_multi_head_attention():
    """
    Test the MultiHeadAttention module with basic functionality.

    This function creates a simple test case to verify that the MultiHeadAttention
    module works correctly. It checks input/output shapes, number of heads,
    and basic forward pass functionality.

    Returns:
        tuple: A tuple containing the output tensor and attention weights tensor
               from the test forward pass.

    Raises:
        AssertionError: If any of the test conditions are not met.

    Note:
        This function is intended for development and debugging purposes.
        It should not be used in production code.
    """
    batch_size, seq_len, d_model, n_heads = 2, 4, 8, 2

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize attention module
    attention = MultiHeadAttention(d_model, n_heads)

    # Perform forward pass
    output, weights = attention(x, x, x)

    # Verify output shapes
    assert output.shape == (
        batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert weights.shape == (batch_size, n_heads, seq_len,
                             seq_len), f"Expected weights shape {(batch_size, n_heads, seq_len, seq_len)}, got {weights.shape}"

    return output, weights


def test_positional_encoding():
    """
    Test the PositionalEncoding module with basic functionality.

    This function creates a simple test case to verify that the PositionalEncoding
    module works correctly. It checks input/output shapes and basic forward pass functionality.

    Returns:
        torch.Tensor: The output tensor from the test forward pass.

    Raises:
        AssertionError: If any of the test conditions are not met.

    Note:
        This function is intended for development and debugging purposes.
        It should not be used in production code.
    """
    seq_len, batch_size, d_model = 10, 2, 8

    # Create random input embeddings
    embeddings = torch.randn(seq_len, batch_size, d_model)

    # Initialize positional encoding module
    pos_encoding = PositionalEncoding(d_model)

    # Perform forward pass
    encoded = pos_encoding(embeddings)

    # Verify output shape
    assert encoded.shape == (
        seq_len, batch_size, d_model), f"Expected output shape {(seq_len, batch_size, d_model)}, got {encoded.shape}"

    return encoded


if __name__ == "__main__":
    test_single_head_attention()
    test_multi_head_attention()
    test_positional_encoding()
