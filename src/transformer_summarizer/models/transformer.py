import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
from pathlib import Path

# import components
from .attention import PositionalEncoding
from .layers import EncoderLayer, DecoderLayer, TransformerEncoder, TransformerDecoder
from .layers import create_padding_mask, create_causal_mask


class Transformer(nn.Module):

    def __init__(self, config_path=None, **kwargs):
        """
        Initialize the Transformer model with either a configuration path or keyword arguments.

        Parameters
        ----------
        config_path : str, optional
            Path to a YAML configuration file containing model hyperparameters.
        **kwargs
            Key-value pairs of model hyperparameters. If config_path is not provided, these are
            used to construct the configuration dictionary. If config_path is provided, these
            arguments are ignored.

        Notes
        -----
        If config_path is provided, the configuration file must contain all required hyperparameters.
        If config_path is not provided, the following hyperparameters are required:

        - vocab_size
        - d_model
        - n_heads
        - n_layers
        - d_ff
        - max_seq_length
        - dropout
        - pad_token_id
        - bos_token_id
        - eos_token_id

        The following hyperparameters have default values:

        - vocab_size: 30522
        - d_model: 384
        - n_heads: 6
        - n_layers: 6
        - d_ff: 1536
        - max_seq_length: 384
        - dropout: 0.1
        - pad_token_id: 0
        - bos_token_id: 101
        - eos_token_id: 102
        """
        super().__init__()

        # Load configuration
        if config_path:
            self.config = self.load_config(config_path)
        else:
            # Use provided kwargs or defaults
            self.config = {
                'vocab_size': kwargs.get('vocab_size', 30522),
                'd_model': kwargs.get('d_model', 384),
                'n_heads': kwargs.get('n_heads', 6),
                'n_layers': kwargs.get('n_layers', 6),
                'd_ff': kwargs.get('d_ff', 1536),
                'max_seq_length': kwargs.get('max_seq_length', 384),
                'dropout': kwargs.get('dropout', 0.1),
                'pad_token_id': kwargs.get('pad_token_id', 0),
                'bos_token_id': kwargs.get('bos_token_id', 101),
                'eos_token_id': kwargs.get('eos_token_id', 102),
            }

        # Validate required parameters
        required_params = ['vocab_size', 'd_model', 'n_heads', 'n_layers', 'd_ff',
                           'max_seq_length', 'dropout', 'pad_token_id', 'bos_token_id', 'eos_token_id']

        missing_params = [
            param for param in required_params if param not in self.config]
        if missing_params:
            raise ValueError(
                f"Missing required configuration parameters: {missing_params}")

        # Store key parameters
        self.vocab_size = self.config['vocab_size']
        self.d_model = self.config['d_model']
        self.pad_token_id = self.config['pad_token_id']
        self.bos_token_id = self.config['bos_token_id']
        self.eos_token_id = self.config['eos_token_id']

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(
            self.d_model, self.config["max_seq_length"])

        self.encoder = TransformerEncoder(
            EncoderLayer,
            self.config["n_layers"],
            d_model=self.d_model,
            n_heads=self.config["n_heads"],
            d_ff=self.config["d_ff"],
            dropout=self.config["dropout"]
        )

        self.decoder = TransformerDecoder(
            DecoderLayer,
            self.config["n_layers"],
            d_model=self.d_model,
            n_heads=self.config["n_heads"],
            d_ff=self.config["d_ff"],
            dropout=self.config["dropout"]
        )

        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        # Tie output projection weights to input token embeddings to improve convergence
        self.output_projection.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(self.config["dropout"])

        self.init_weights()

    def load_config(self, config_path):
        """
        Load model configuration from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to a YAML configuration file containing model hyperparameters.

        Returns
        -------
        config : dict
            A dictionary containing all model hyperparameters.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If required parameters are missing from the configuration.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract model config if nested
        if 'model' in config:
            config = config['model']

        # Set default values for missing parameters
        defaults = {
            'vocab_size': 30522,
            'd_model': 384,
            'n_heads': 6,
            'n_layers': 6,
            'd_ff': 1536,
            'max_seq_length': 384,
            'dropout': 0.1,
            'pad_token_id': 0,
            'bos_token_id': 101,
            'eos_token_id': 102,
        }

        # Update config with defaults for missing values
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

        return config

    def init_weights(self):
        """
        Initialize all model weights using Xavier uniform initialization.

        For all parameters `p` in the model, apply Xavier uniform initialization
        if `p` has more than one dimension, and initialize to zero otherwise.

        This is a common initialization scheme for neural networks, which helps
        stabilize the optimization process by keeping the weights small and
        evenly distributed around zero.

        See Also:
            `torch.nn.init.xavier_uniform_`
            `torch.nn.init.zeros_`
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Compute the output of the transformer model given input sequences `src` and `tgt`.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, src_len) containing source sequence tokens.
        tgt : torch.Tensor
            Input tensor of shape (batch_size, tgt_len) containing target sequence tokens.
        src_mask : torch.Tensor, optional
            Optional source attention mask of shape (batch_size, src_len, src_len).
            If None, no masking is applied.
        tgt_mask : torch.Tensor, optional
            Optional target attention mask of shape (batch_size, tgt_len, tgt_len).
            If None, no masking is applied.

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (batch_size, tgt_len, vocab_size) containing predicted
            probabilities over the target vocabulary.
        """
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)

        # Scale embeddings by sqrt(d_model) as per Transformer paper
        src_embed = self.token_embedding(src) * math.sqrt(self.d_model)
        src_embed = self.positional_encoding(
            src_embed.transpose(0, 1)).transpose(0, 1)
        src_embed = self.dropout(src_embed)

        tgt_embed = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.positional_encoding(
            tgt_embed.transpose(0, 1)).transpose(0, 1)
        tgt_embed = self.dropout(tgt_embed)

        encoder_output = self.encoder(src_embed, src_mask)
        decoder_output = self.decoder(
            tgt_embed, encoder_output, src_mask, tgt_mask)

        output = self.output_projection(decoder_output)
        return output

    def create_src_mask(self, src):
        """
        Create source attention mask for encoder input.

        This function generates a padding mask for the source sequence,
        preventing the attention mechanism from attending to padding tokens.
        Padding tokens are typically represented by a special token ID and
        should be ignored during attention computation.

        Args:
            src: Input tensor of shape (batch_size, src_len) containing source sequence tokens.

        Returns:
            torch.Tensor: Binary mask of shape (batch_size, 1, 1, src_len).
                        Values of 1 indicate real tokens that can be attended to.
                        Values of 0 indicate padding tokens that should be masked.
        """

        src_mask = create_padding_mask(src, self.pad_token_id)
        return src_mask

    def create_tgt_mask(self, tgt):
        """
        Create target attention mask for decoder input.

        This function generates two masks for the target sequence:
        1. Padding mask: prevents attention to padding tokens.
        2. Causal mask: prevents looking at future tokens (i.e., implements autoregressive property).

        The two masks are combined using element-wise logical AND. The resulting mask is a binary tensor
        of shape (batch_size, 1, 1, tgt_len) where values of 1 indicate real tokens that can be attended to,
        and values of 0 indicate positions that should be masked.

        Args:
            tgt: Input tensor of shape (batch_size, tgt_len) containing target sequence tokens.

        Returns:
            torch.Tensor: Binary mask of shape (batch_size, 1, 1, tgt_len).
                        Values of 1 indicate real tokens that can be attended to.
                        Values of 0 indicate positions that should be masked.
        """
        batch_size, tgt_len = tgt.size()

        # Padding mask
        tgt_padding_mask = create_padding_mask(tgt, self.pad_token_id)

        # Causal mask (prevent looking at future tokens)
        tgt_causal_mask = create_causal_mask(tgt_len).to(tgt.device)

        # Combine masks (both must be True to attend)
        tgt_mask = tgt_padding_mask & tgt_causal_mask.unsqueeze(0).unsqueeze(0)

        return tgt_mask

    def generate(self, src, max_length=100, temperature=0.8, top_k=50):
        """
        Generate sequences using the model as a language model.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, src_len) containing source sequence tokens.
        max_length : int, optional
            Maximum length of generated sequence. Defaults to 100.
        temperature : float, optional
            Temperature for sampling next token probabilities. Defaults to 0.8.
        top_k : int, optional
            Number of top tokens to consider when sampling. Defaults to 50.

        Returns
        -------
        generated : torch.Tensor
            Generated sequence tensor of shape (batch_size, generated_len).
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        generated = torch.full(
            (batch_size, 1), self.bos_token_id, device=device)

        src_mask = self.create_src_mask(src)
        src_embed = self.token_embedding(src) * math.sqrt(self.d_model)
        src_embed = self.positional_encoding(
            src_embed.transpose(0, 1)).transpose(0, 1)
        encoder_output = self.encoder(src_embed, src_mask)

        for _ in range(max_length):
            tgt_mask = self.create_tgt_mask(generated)
            tgt_embed = self.token_embedding(
                generated) * math.sqrt(self.d_model)
            tgt_embed = self.positional_encoding(
                tgt_embed.transpose(0, 1)).transpose(0, 1)
            decoder_output = self.decoder(
                tgt_embed, encoder_output, src_mask, tgt_mask)

            next_token_logits = self.output_projection(
                decoder_output[:, -1, :])
            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    next_token_logits, top_k, dim=-1)
                next_token_logits = torch.full_like(
                    next_token_logits, -float('inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.eos_token_id:
                break

        return generated

    def greedy_generate(self, src, max_length=100):
        """
        Generate sequences using the model as a language model in a greedy manner.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, src_len) containing source sequence tokens.
        max_length : int, optional
            Maximum length of generated sequence. Defaults to 100.

        Returns
        -------
        generated : torch.Tensor
            Generated sequence tensor of shape (batch_size, generated_len).
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        generated = torch.full(
            (batch_size, 1), self.bos_token_id, device=device)

        src_mask = self.create_src_mask(src)
        src_embedded = self.token_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(
            src_embedded.transpose(0, 1)).transpose(0, 1)
        encoder_output = self.encoder(src_embedded, src_mask)

        for step in range(max_length):
            tgt_mask = self.create_tgt_mask(generated)

            tgt_embedded = self.token_embedding(
                generated) * math.sqrt(self.d_model)
            tgt_embedded = self.positional_encoding(
                tgt_embedded.transpose(0, 1)).transpose(0, 1)
            decoder_output = self.decoder(
                tgt_embedded, encoder_output, src_mask, tgt_mask)

            next_token_logits = self.output_projection(
                decoder_output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == self.eos_token_id).all():
                break

        return generated


def test_complete_model():
    """Test the complete transformer model"""

    # Model configuration - ensure d_model is divisible by n_heads
    config = {
        'vocab_size': 1000,
        'd_model': 128,  # Must be divisible by n_heads
        'n_heads': 4,    # 128 / 4 = 32 (valid)
        'n_layers': 2,
        'd_ff': 512,
        'max_seq_length': 100,
        'dropout': 0.1,
        'pad_token_id': 0,
        'bos_token_id': 1,
        'eos_token_id': 2,
    }

    # Validate configuration
    if config['d_model'] % config['n_heads'] != 0:
        raise ValueError(
            f"d_model ({config['d_model']}) must be divisible by n_heads ({config['n_heads']})")

    # Create model
    model = Transformer(**config)

    # Test data
    batch_size, src_len, tgt_len = 2, 10, 8
    src = torch.randint(3, config['vocab_size'], (batch_size, src_len))
    tgt = torch.randint(3, config['vocab_size'], (batch_size, tgt_len))

    # Forward pass
    output = model(src, tgt)

    # Test generation
    with torch.no_grad():
        generated = model.greedy_generate(src, max_length=15)

    return model, output, generated


def load_model_from_config(config_path):
    """Load model from configuration file"""

    try:
        model = Transformer(config_path=config_path)
        return model

    except Exception as e:
        return None
