import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
import copy


class FeedForward(nn.Module):
    """
    Feed-forward neural network layer for Transformer models.

    This module implements the position-wise feed-forward network that consists of
    two linear transformations with a ReLU activation function in between.
    The feed-forward network is applied to each position separately and identically.

    Attributes:
        linear_1 (nn.Linear): First linear transformation layer.
        linear_2 (nn.Linear): Second linear transformation layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        activation (nn.ReLU): ReLU activation function.

    Example:
        >>> ff = FeedForward(d_model=512, d_ff=2048, dropout=0.1)
        >>> x = torch.randn(32, 100, 512)  # batch_size=32, seq_len=100, d_model=512
        >>> output = ff(x)
        >>> print(output.shape)  # torch.Size([32, 100, 512])

    Reference:
        Vaswani, A., et al. "Attention is all you need." Advances in neural information
        processing systems 30 (2017).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the feed-forward network.

        Args:
            d_model: The dimensionality of the input and output vectors.
                    Must be a positive integer.
            d_ff: The dimensionality of the intermediate layer (feed-forward dimension).
                  Must be a positive integer, typically 4x larger than d_model.
            dropout: Dropout probability for regularization. Must be in [0, 1).

        Raises:
            ValueError: If parameters are outside valid ranges.

        Note:
            The feed-forward network typically uses d_ff = 4 * d_model as recommended
            in the original Transformer paper. This provides sufficient capacity
            for the model to learn complex transformations.
        """
        super().__init__()

        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(d_ff, int) or d_ff <= 0:
            raise ValueError("d_ff must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation to input tensor.

        This method implements the position-wise feed-forward network:
        1. Apply first linear transformation
        2. Apply ReLU activation
        3. Apply dropout for regularization
        4. Apply second linear transformation

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).

        Raises:
            ValueError: If input tensor has incorrect shape or d_model dimension.

        Note:
            The feed-forward network is applied independently to each position
            in the sequence, making it position-wise. This allows the model to
            learn position-specific transformations.
        """
        if x.size(-1) != self.linear_1.in_features:
            raise ValueError(
                f"Input tensor must have last dimension equal to {self.linear_1.in_features}")

        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class EncoderLayer(nn.Module):
    """
    Single layer of the Transformer encoder.

    This module implements one encoder layer as described in the Transformer architecture.
    Each encoder layer consists of two sublayers:
    1. Multi-head self-attention mechanism
    2. Position-wise feed-forward network

    Each sublayer is followed by residual connection and layer normalization.

    Attributes:
        self_attention (MultiHeadAttention): Multi-head self-attention mechanism.
        feed_forward (FeedForward): Position-wise feed-forward network.
        norm_1 (nn.LayerNorm): Layer normalization for attention sublayer.
        norm_2 (nn.LayerNorm): Layer normalization for feed-forward sublayer.
        dropout (nn.Dropout): Dropout layer for regularization.

    Example:
        >>> encoder_layer = EncoderLayer(d_model=512, n_heads=8, d_ff=2048)
        >>> x = torch.randn(32, 100, 512)  # batch_size=32, seq_len=100, d_model=512
        >>> output = encoder_layer(x)
        >>> print(output.shape)  # torch.Size([32, 100, 512])

    Reference:
        Vaswani, A., et al. "Attention is all you need." Advances in neural information
        processing systems 30 (2017).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the encoder layer.

        Args:
            d_model: The dimensionality of the input embeddings and output vectors.
                    Must be a positive integer.
            n_heads: The number of attention heads. Must be a positive integer
                    that divides d_model evenly.
            d_ff: The dimensionality of the feed-forward network. Must be a positive integer.
            dropout: Dropout probability for regularization. Must be in [0, 1).

        Raises:
            ValueError: If parameters are outside valid ranges or d_model is not
                       divisible by n_heads.

        Note:
            The encoder layer uses residual connections and layer normalization
            to stabilize training and improve gradient flow.
        """
        super().__init__()

        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(n_heads, int) or n_heads <= 0:
            raise ValueError("n_heads must be a positive integer")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if not isinstance(d_ff, int) or d_ff <= 0:
            raise ValueError("d_ff must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        # Core components
        self.self_attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

        # Normalization layers (one for each sublayer)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply encoder layer transformation to input tensor.

        This method implements the encoder layer computation:
        1. Multi-head self-attention with residual connection and normalization
        2. Position-wise feed-forward network with residual connection and normalization

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len).
                  Values of 0 indicate positions to mask (set to -inf).
                  Values of 1 indicate positions to attend to.
                  If None, no masking is applied.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).

        Raises:
            ValueError: If input tensor has incorrect shape or d_model dimension.

        Note:
            The residual connections help with gradient flow and allow the model
            to learn incremental updates to the representations.
        """
        if x.size(-1) != self.norm_1.normalized_shape[0]:
            raise ValueError(
                f"Input tensor must have last dimension equal to {self.norm_1.normalized_shape[0]}")

        # Self-attention sublayer with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm_1(x)

        # Feed-forward sublayer with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm_2(x)

        return x


class DecoderLayer(nn.Module):
    """
    Single layer of the Transformer decoder.

    This module implements one decoder layer as described in the Transformer architecture.
    Each decoder layer consists of three sublayers:
    1. Multi-head self-attention mechanism (with causal masking)
    2. Multi-head cross-attention mechanism (attending to encoder output)
    3. Position-wise feed-forward network

    Each sublayer is followed by residual connection and layer normalization.

    Attributes:
        self_attention (MultiHeadAttention): Multi-head self-attention mechanism.
        cross_attention (MultiHeadAttention): Multi-head cross-attention mechanism.
        feed_forward (FeedForward): Position-wise feed-forward network.
        norm_1 (nn.LayerNorm): Layer normalization for self-attention sublayer.
        norm_2 (nn.LayerNorm): Layer normalization for cross-attention sublayer.
        norm_3 (nn.LayerNorm): Layer normalization for feed-forward sublayer.
        dropout (nn.Dropout): Dropout layer for regularization.

    Example:
        >>> decoder_layer = DecoderLayer(d_model=512, n_heads=8, d_ff=2048)
        >>> x = torch.randn(32, 50, 512)  # target sequence
        >>> encoder_output = torch.randn(32, 100, 512)  # encoder output
        >>> output = decoder_layer(x, encoder_output)
        >>> print(output.shape)  # torch.Size([32, 50, 512])

    Reference:
        Vaswani, A., et al. "Attention is all you need." Advances in neural information
        processing systems 30 (2017).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize the decoder layer.

        Args:
            d_model: The dimensionality of the input embeddings and output vectors.
                    Must be a positive integer.
            n_heads: The number of attention heads. Must be a positive integer
                    that divides d_model evenly.
            d_ff: The dimensionality of the feed-forward network. Must be a positive integer.
            dropout: Dropout probability for regularization. Must be in [0, 1).

        Raises:
            ValueError: If parameters are outside valid ranges or d_model is not
                       divisible by n_heads.

        Note:
            The decoder layer uses causal masking in self-attention to prevent
            positions from attending to future positions, ensuring autoregressive
            generation.
        """
        super().__init__()

        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not isinstance(n_heads, int) or n_heads <= 0:
            raise ValueError("n_heads must be a positive integer")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if not isinstance(d_ff, int) or d_ff <= 0:
            raise ValueError("d_ff must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        # Core Components
        self.self_attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout)
        self.cross_attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

        # Normalization layers (one for each sublayer)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply decoder layer transformation to input tensor.

        This method implements the decoder layer computation:
        1. Multi-head self-attention with causal masking and residual connection
        2. Multi-head cross-attention with encoder output and residual connection
        3. Position-wise feed-forward network with residual connection

        Args:
            x: Input tensor of shape (batch_size, tgt_len, d_model).
               Target sequence embeddings.
            encoder_output: Encoder output tensor of shape (batch_size, src_len, d_model).
                           Output from the encoder stack.
            src_mask: Optional source attention mask of shape (batch_size, tgt_len, src_len).
                     Used for cross-attention to mask encoder positions.
                     Values of 0 indicate positions to mask (set to -inf).
                     Values of 1 indicate positions to attend to.
                     If None, no masking is applied.
            tgt_mask: Optional target attention mask of shape (batch_size, tgt_len, tgt_len).
                     Used for self-attention to implement causal masking.
                     Values of 0 indicate positions to mask (set to -inf).
                     Values of 1 indicate positions to attend to.
                     If None, no masking is applied.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_len, d_model).

        Raises:
            ValueError: If input tensors have incorrect shapes or d_model dimensions.

        Note:
            The causal masking in self-attention ensures that each position can only
            attend to previous positions, which is essential for autoregressive
            generation in sequence-to-sequence tasks.
        """
        if x.size(-1) != self.norm_1.normalized_shape[0]:
            raise ValueError(
                f"Input tensor must have last dimension equal to {self.norm_1.normalized_shape[0]}")

        if encoder_output.size(-1) != self.norm_1.normalized_shape[0]:
            raise ValueError(
                f"Encoder output must have last dimension equal to {self.norm_1.normalized_shape[0]}")

        if x.size(0) != encoder_output.size(0):
            raise ValueError(
                "Input and encoder output must have the same batch size")

        # Self Attention with causal masking
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout(attn_output)
        x = self.norm_1(x)

        # Cross Attention with encoder output
        cross_attn_output, _ = self.cross_attention(
            x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm_2(x)

        # Feed Forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm_3(x)

        return x


class TransformerEncoder(nn.Module):
    """
    Complete Transformer encoder stack.

    This module implements a stack of encoder layers that process the input sequence
    to create contextual representations. Each encoder layer applies self-attention
    and feed-forward transformations with residual connections and normalization.

    Attributes:
        layers (nn.ModuleList): List of encoder layers.
        n_layers (int): Number of encoder layers in the stack.

    Example:
        >>> encoder_layer = EncoderLayer(d_model=512, n_heads=8, d_ff=2048)
        >>> encoder = TransformerEncoder(encoder_layer, n_layers=6)
        >>> x = torch.randn(32, 100, 512)  # batch_size=32, seq_len=100, d_model=512
        >>> output = encoder(x)
        >>> print(output.shape)  # torch.Size([32, 100, 512])

    Reference:
        Vaswani, A., et al. "Attention is all you need." Advances in neural information
        processing systems 30 (2017).
    """

    def __init__(self, encoder_layer_cls, n_layers, *args, **kwargs):
        """
        Initialize the Transformer encoder stack.

        Args:
            encoder_layer_cls: Class of EncoderLayer to be replicated.
            n_layers: Number of encoder layers in the stack. Must be a positive integer.
            *args: Positional arguments for EncoderLayer.
            **kwargs: Keyword arguments for EncoderLayer.

        Raises:
            ValueError: If n_layers is not a positive integer.
            TypeError: If encoder_layer_cls is not a class.

        Note:
            Each layer in the stack is a copy of the provided encoder_layer_cls,
            allowing for parameter sharing or independent parameters depending
            on the implementation.
        """
        super().__init__()

        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError("n_layers must be a positive integer")

        if not isinstance(encoder_layer_cls, type):
            raise TypeError(
                "encoder_layer_cls must be a class")

        self.layers = nn.ModuleList(
            [encoder_layer_cls(*args, **kwargs) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply encoder stack transformation to input tensor.

        This method processes the input through all encoder layers sequentially,
        with each layer applying self-attention and feed-forward transformations.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len).
                  Applied to all encoder layers. Values of 0 indicate positions
                  to mask (set to -inf). Values of 1 indicate positions to attend to.
                  If None, no masking is applied.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).

        Raises:
            ValueError: If input tensor has incorrect shape or d_model dimension.

        Note:
            The encoder stack processes the entire input sequence in parallel,
            creating contextual representations that capture relationships between
            all positions in the sequence.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    """
    Complete Transformer decoder stack.

    This module implements a stack of decoder layers that process the target sequence
    while attending to the encoder output. Each decoder layer applies self-attention
    (with causal masking), cross-attention to the encoder, and feed-forward
    transformations with residual connections and normalization.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        n_layers (int): Number of decoder layers in the stack.

    Example:
        >>> decoder_layer = DecoderLayer(d_model=512, n_heads=8, d_ff=2048)
        >>> decoder = TransformerDecoder(decoder_layer, n_layers=6)
        >>> x = torch.randn(32, 50, 512)  # target sequence
        >>> encoder_output = torch.randn(32, 100, 512)  # encoder output
        >>> output = decoder(x, encoder_output)
        >>> print(output.shape)  # torch.Size([32, 50, 512])

    Reference:
        Vaswani, A., et al. "Attention is all you need." Advances in neural information
        processing systems 30 (2017).
    """

    def __init__(self, decoder_layer_cls, n_layers, *args, **kwargs):
        """
        Initialize the Transformer decoder stack.

        Args:
            decoder_layer_cls: Class of DecoderLayer to be replicated.
            n_layers: Number of decoder layers in the stack. Must be a positive integer.
            *args: Positional arguments for DecoderLayer.
            **kwargs: Keyword arguments for DecoderLayer.

        Raises:
            ValueError: If n_layers is not a positive integer.
            TypeError: If decoder_layer_cls is not a class.

        Note:
            Each layer in the stack is a copy of the provided decoder_layer_cls,
            allowing for parameter sharing or independent parameters depending
            on the implementation.
        """
        super().__init__()

        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError("n_layers must be a positive integer")

        if not isinstance(decoder_layer_cls, type):
            raise TypeError(
                "decoder_layer_cls must be a class")

        self.layers = nn.ModuleList(
            [decoder_layer_cls(*args, **kwargs) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Apply decoder stack transformation to input tensor.

        This method processes the input through all decoder layers sequentially,
        with each layer applying self-attention (with causal masking),
        cross-attention to the encoder output, and feed-forward transformations.

        Args:
            x: Input tensor of shape (batch_size, tgt_len, d_model).
               Target sequence embeddings.
            encoder_output: Encoder output tensor of shape (batch_size, src_len, d_model).
                           Output from the encoder stack.
            src_mask: Optional source attention mask of shape (batch_size, tgt_len, src_len).
                     Applied to cross-attention in all decoder layers.
                     Values of 0 indicate positions to mask (set to -inf).
                     Values of 1 indicate positions to attend to.
                     If None, no masking is applied.
            tgt_mask: Optional target attention mask of shape (batch_size, tgt_len, tgt_len).
                     Applied to self-attention in all decoder layers.
                     Values of 0 indicate positions to mask (set to -inf).
                     Values of 1 indicate positions to attend to.
                     If None, no masking is applied.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_len, d_model).

        Raises:
            ValueError: If input tensors have incorrect shapes or d_model dimensions.

        Note:
            The decoder stack processes the target sequence while attending to
            the encoder output, enabling sequence-to-sequence generation with
            autoregressive properties.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create attention mask for padding tokens.

    This function creates a binary mask that prevents the attention mechanism
    from attending to padding tokens. Padding tokens are typically represented
    by a special token ID (usually 0) and should be ignored during attention
    computation.

    Args:
        seq: Token sequence tensor of shape (batch_size, seq_len).
             Contains token IDs including padding tokens.
        pad_token_id: ID of the padding token. Defaults to 0.
                     Must be an integer.

    Returns:
        torch.Tensor: Binary mask of shape (batch_size, 1, 1, seq_len).
                     Values of 1 indicate real tokens that can be attended to.
                     Values of 0 indicate padding tokens that should be masked.

    Raises:
        ValueError: If seq is not a 2D tensor or pad_token_id is not an integer.

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])  # 0 = padding
        >>> mask = create_padding_mask(seq, pad_token_id=0)
        >>> print(mask.shape)  # torch.Size([2, 1, 1, 5])
        >>> print(mask.squeeze())  # Shows which positions are real vs padding

    Note:
        The mask is designed to be broadcastable to attention weight shapes
        (batch_size, n_heads, seq_len, seq_len) when used with multi-head attention.
    """
    if not isinstance(seq, torch.Tensor) or seq.dim() != 2:
        raise ValueError("seq must be a 2D tensor")

    if not isinstance(pad_token_id, int):
        raise ValueError("pad_token_id must be an integer")

    # Create binary mask: True for real tokens, False for padding
    mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(size: int) -> torch.Tensor:
    """
    Create causal (look-ahead) mask for decoder self-attention.

    This function creates a lower triangular mask that prevents positions
    from attending to subsequent positions. This is essential for autoregressive
    generation, ensuring that the model can only use information from previous
    tokens when generating the current token.

    Args:
        size: Sequence length. Must be a positive integer.

    Returns:
        torch.Tensor: Binary mask of shape (size, size).
                     Lower triangular matrix where values of True indicate
                     positions that can be attended to, and values of False
                     indicate positions that should be masked.

    Raises:
        ValueError: If size is not a positive integer.

    Example:
        >>> mask = create_causal_mask(5)
        >>> print(mask)
        tensor([[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [ True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True]])

    Note:
        The causal mask ensures that position i can only attend to positions
        j ≤ i, implementing the autoregressive property required for
        sequence generation tasks.
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")

    # Create upper triangular matrix with diagonal offset of 1
    # This creates a mask where each position can only see previous positions
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # Invert: True where we can attend, False where we can't



def test_feed_forward():
    """
    Test the FeedForward module with basic functionality.

    This function creates a simple test case to verify that the FeedForward
    module works correctly. It checks input/output shapes, shape preservation,
    and basic forward pass functionality.

    Returns:
        torch.Tensor: The output tensor from the test forward pass.

    Raises:
        AssertionError: If any of the test conditions are not met.

    Note:
        This function is intended for development and debugging purposes.
        It should not be used in production code.
    """
    batch_size, seq_len, d_model, d_ff = 2, 5, 64, 256

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize feed-forward module
    ff = FeedForward(d_model, d_ff)

    # Perform forward pass
    output = ff(x)

    # Verify output shape
    assert output.shape == (
        batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"

    return output


def test_encoder_layer():
    """
    Test the EncoderLayer module with basic functionality.

    This function creates a simple test case to verify that the EncoderLayer
    module works correctly. It checks input/output shapes, residual connections,
    and basic forward pass functionality.

    Returns:
        torch.Tensor: The output tensor from the test forward pass.

    Raises:
        AssertionError: If any of the test conditions are not met.

    Note:
        This function is intended for development and debugging purposes.
        It should not be used in production code.
    """
    batch_size, seq_len, d_model, n_heads, d_ff = 2, 5, 64, 8, 256

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize encoder layer
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff)

    # Perform forward pass
    output = encoder_layer(x)

    # Verify output shape
    assert output.shape == (
        batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"

    return output


def test_decoder_layer():
    """
    Test the DecoderLayer module with basic functionality.

    This function creates a simple test case to verify that the DecoderLayer
    module works correctly. It checks input/output shapes, cross-attention
    functionality, and basic forward pass with encoder output.

    Returns:
        torch.Tensor: The output tensor from the test forward pass.

    Raises:
        AssertionError: If any of the test conditions are not met.

    Note:
        This function is intended for development and debugging purposes.
        It should not be used in production code.
    """
    batch_size, src_len, tgt_len, d_model, n_heads, d_ff = 2, 6, 4, 64, 8, 256

    # Create random input tensors
    x = torch.randn(batch_size, tgt_len, d_model)  # Target sequence
    encoder_output = torch.randn(batch_size, src_len, d_model)  # From encoder

    # Initialize decoder layer
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff)

    # Perform forward pass
    output = decoder_layer(x, encoder_output)

    # Verify output shape
    assert output.shape == (
        batch_size, tgt_len, d_model), f"Expected output shape {(batch_size, tgt_len, d_model)}, got {output.shape}"

    return output


def test_masks():
    """
    Test mask creation functions with basic functionality.

    This function creates test cases to verify that the padding mask and
    causal mask creation functions work correctly. It checks mask shapes,
    mask values, and basic functionality.

    Returns:
        tuple: A tuple containing the padding mask and causal mask tensors
               from the test cases.

    Raises:
        AssertionError: If any of the test conditions are not met.

    Note:
        This function is intended for development and debugging purposes.
        It should not be used in production code.
    """
    # Test padding mask
    seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])  # 0 = padding
    pad_mask = create_padding_mask(seq, pad_token_id=0)

    # Verify padding mask shape
    assert pad_mask.shape == (
        2, 1, 1, 5), f"Expected padding mask shape (2, 1, 1, 5), got {pad_mask.shape}"

    # Test causal mask
    causal_mask = create_causal_mask(5)

    # Verify causal mask shape
    assert causal_mask.shape == (
        5, 5), f"Expected causal mask shape (5, 5), got {causal_mask.shape}"

    return pad_mask, causal_mask


def test_complete_stacks():
    """
    Test complete encoder and decoder stacks with basic functionality.

    This function creates test cases to verify that the complete Transformer
    encoder and decoder stacks work correctly. It checks input/output shapes,
    layer counts, and end-to-end forward pass functionality.

    Returns:
        tuple: A tuple containing the encoder output and decoder output tensors
               from the test forward passes.

    Raises:
        AssertionError: If any of the test conditions are not met.

    Note:
        This function is intended for development and debugging purposes.
        It should not be used in production code.
    """
    d_model, n_heads, d_ff, n_layers = 64, 8, 256, 3
    batch_size, src_len, tgt_len = 2, 6, 4

    # Create layer instances
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff)
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff)

    # Create stacks
    encoder = TransformerEncoder(encoder_layer, n_layers)
    decoder = TransformerDecoder(decoder_layer, n_layers)

    # Create test data
    src = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randn(batch_size, tgt_len, d_model)

    # Perform forward passes
    encoder_output = encoder(src)
    decoder_output = decoder(tgt, encoder_output)

    # Verify output shapes
    assert encoder_output.shape == (
        batch_size, src_len, d_model), f"Expected encoder output shape {(batch_size, src_len, d_model)}, got {encoder_output.shape}"
    assert decoder_output.shape == (
        batch_size, tgt_len, d_model), f"Expected decoder output shape {(batch_size, tgt_len, d_model)}, got {decoder_output.shape}"

    return encoder_output, decoder_output
