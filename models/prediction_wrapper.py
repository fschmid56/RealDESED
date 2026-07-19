import os

import torch
import torch.nn as nn
from torch.hub import download_url_to_file

from config import RESOURCES_FOLDER, CHECKPOINT_URLS


class PredictionsWrapper(nn.Module):
    """
        Wraps a frame-level audio transformer and optionally loads released weights.

        Args:
            base_model (BaseModelWrapper): transformer that returns sequence embeddings.
            checkpoint (str, optional): checkpoint name for loading pre-trained weights. Default is None.
            embed_dim (int, optional): Embedding dimension of the base model output. Default is 768.
            seq_len (int, optional): Desired sequence length. Default is 250 (40 ms resolution).
        """

    def __init__(self,
                 base_model,
                 checkpoint=None,
                 embed_dim=768,
                 seq_len=250,
                 seq_model_type=None,
                 head_type=None,
                 ):
        super(PredictionsWrapper, self).__init__()
        if seq_model_type is not None or head_type is not None:
            raise ValueError("The public ATST-F baseline uses sequence embeddings without an internal head.")

        self.model = base_model
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.seq_model = nn.Identity()
        self.num_features = self.embed_dim

        if checkpoint is not None:
            print("Loading pretrained checkpoint: ", checkpoint)
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        os.makedirs(RESOURCES_FOLDER, exist_ok=True)
        ckpt_file = os.path.join(RESOURCES_FOLDER, checkpoint + ".pt")
        if not os.path.exists(ckpt_file):
            download_url_to_file(CHECKPOINT_URLS[checkpoint], ckpt_file)
        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        keys_to_remove = {
            "weak_head.bias",
            "weak_head.weight",
            "strong_head.bias",
            "strong_head.weight",
        }

        # Older released checkpoints may not include torchaudio mel-transform buffers.
        allowed_missing = {key for key in self.state_dict() if "mel_transform" in key}

        state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_remove}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in unexpected if key not in keys_to_remove]
        disallowed_missing = [key for key in missing if key not in allowed_missing]
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "Failed to load pretrained checkpoint "
                f"{checkpoint!r}. Missing keys: {disallowed_missing}. "
                f"Unexpected keys: {unexpected}."
            )

    def separate_params(self):
        if hasattr(self.model, "separate_params"):
            return self.model.separate_params()
        else:
            raise NotImplementedError("The base model has no 'separate_params' method!'")

    def has_separate_params(self):
        return hasattr(self.model, "separate_params")

    def mel_forward(self, x):
        return self.model.mel_forward(x)

    def forward(self, x):
        # Base model is expected to output [batch, frames, embedding_dim].
        # (batch size x sequence length x embedding dimension)
        x = self.model(x)

        assert len(x.shape) == 3

        if x.size(-2) > self.seq_len:
            x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.seq_len).transpose(1, 2)
        elif x.size(-2) < self.seq_len:
            x = torch.nn.functional.interpolate(x.transpose(1, 2), size=self.seq_len,
                                                mode='linear').transpose(1, 2)

        return self.seq_model(x)

    def forward_features(self, x):
        """
        Returns sequence features AFTER seq_model.
        Shape: [B, T, D]
        """
        x = self.model(x)

        assert len(x.shape) == 3

        if x.size(-2) > self.seq_len:
            x = torch.nn.functional.adaptive_avg_pool1d(
                x.transpose(1, 2), self.seq_len
            ).transpose(1, 2)
        elif x.size(-2) < self.seq_len:
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2),
                size=self.seq_len,
                mode='linear'
            ).transpose(1, 2)

        x = self.seq_model(x)

        return x
