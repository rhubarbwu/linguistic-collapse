from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel


def truncate_and_pad(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    assert len(batch) > 0
    if len(batch) == 1:
        return None, batch[0].unsqueeze(0)

    masks = [[1] * len(seq) for seq in batch]
    longest = max([sum(seq) for seq in masks])

    masks = torch.tensor([m + [0] * (longest - len(m)) for m in masks])
    batch = pad_sequence(batch, batch_first=True).clone().detach()

    return masks, batch


def process_batch(
    model: PreTrainedModel,
    batch: List[Tensor],
    stats_device: Union[str, torch.device] = "cpu",
) -> Tuple[Tensor, Tensor]:
    masks, batch = truncate_and_pad(batch)
    output = model(
        batch.to(model.device),
        attention_mask=masks if masks is None else masks.to(model.device),
        output_hidden_states=False,
    )

    embeds = output.logits.to(stats_device)
    if masks is not None:
        embeds = torch.unsqueeze(masks, -1) * embeds

    # offset by one for the next word prediction
    Y = batch[:, 1:].to(stats_device)
    X = embeds[:, :-1].to(stats_device)
    if masks is not None:
        idx = masks[:, 1:].bool()
        Y, X = Y[idx], X[idx]

    return X.squeeze(), Y.squeeze()
