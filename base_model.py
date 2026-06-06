import logging
import torch
from torch import nn


class BaseModel(nn.Module):
    """
    A base model class that can be used to build other specific models 
    on top of.
    """

    def _initialise_weights(self) -> None:
        """
        Apply appropriate weight initialisation across all layer types:
        - Linear & Conv layers: Xavier normal for weights, 
            zeros for biases.
        - Recurrent layers (LSTM, GRU, RNN): Xavier normal for weights, 
            zeros for biases.
        - Normalisation layers (BatchNorm, LayerNorm, etc.): ones for 
            weights, zeros for biases.
        - Embedding layers: Standard normal distribution.
        - MultiheadAttention: Xavier normal for weights, 
            zeros for biases.
        """
        for module in self.modules():
            # Linear & Convolutional 
            if isinstance(module, (
                nn.Linear,
                nn.Conv1d, 
                nn.Conv2d, 
                nn.Conv3d,
                nn.ConvTranspose1d, 
                nn.ConvTranspose2d, 
                nn.ConvTranspose3d,
            )):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # Recurrent
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

            # Normalisation
            elif isinstance(module, (
                nn.BatchNorm1d, 
                nn.BatchNorm2d, 
                nn.BatchNorm3d,
                nn.LayerNorm, 
                nn.GroupNorm,
                nn.InstanceNorm1d, 
                nn.InstanceNorm2d, 
                nn.InstanceNorm3d,
            )):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # Embedding
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

            # MultiheadAttention
            elif isinstance(module, nn.MultiheadAttention):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def save(self, destination: str, prefix: str=None)-> None:
        """
        Save internal state to file.

        :param destination: Directory/file to output model to.
        :type destination: str
        :param prefix: prefix for the filename.
        :type prefix: str
        """
        filename = \
            f"{destination}/{prefix if prefix is not None else ''}" \
            f"best_{self.__class__.__name__}.pth"
        if ".pth" in destination:
            filename = destination
        self.logger.info(f"Saving model to {filename}...")
        torch.save(self.state_dict(), filename)
