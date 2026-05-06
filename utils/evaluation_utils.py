import rootutils
import warnings
import torch
import numpy as np
from typing import Literal

warnings.filterwarnings("ignore")
ROOT = rootutils.setup_root(search_from=__file__, cwd=True, pythonpath=False)


torch.set_float32_matmul_precision("high")


@torch.no_grad()
def evaluate_val_loss(
    model,
    val_loader,
    device,
    max_batches: int = 50,
    loss_type: str = "mse",
    conditioning_mode: Literal["auto", "single", "two_branch"] = "auto",
    single_emb_key: str = "fused",
    vtxt_key: str = "vtxt",
    ctxt_key: str = "ctxt",
    ctxt_mask_key: str = "ctxt_mask",
    target_key: str = "fb_traj_embedding",
    mask_key: str = "mask",
):
    model.eval()
    losses = []

    for bi, batch in enumerate(val_loader):
        if bi >= max_batches:
            break

        z_seq_gt = batch[target_key].to(device, non_blocking=True)
        mask = batch[mask_key].to(device, non_blocking=True)

        mode = conditioning_mode
        if mode == "auto":
            if (vtxt_key in batch) and (ctxt_key in batch):
                mode = "two_branch"
            elif single_emb_key in batch:
                mode = "single"
            else:
                raise KeyError(
                    f"Could not auto-detect conditioning mode. Expected either '{single_emb_key}' "
                    f"or both '{vtxt_key}' and '{ctxt_key}'. Available keys: {list(batch.keys())}"
                )

        with torch.amp.autocast(device, dtype=torch.bfloat16):
            if mode == "single":
                text_emb = batch[single_emb_key].to(device, non_blocking=True)
                loss = model.update(
                    text_emb=text_emb,
                    z_seq_gt=z_seq_gt,
                    mask=mask,
                    loss=loss_type,
                )
            elif mode == "two_branch":
                vtxt = batch[vtxt_key].to(device, non_blocking=True)
                ctxt = batch[ctxt_key].to(device, non_blocking=True)

                ctxt_mask = batch.get(ctxt_mask_key, None)
                if ctxt_mask is None:
                    ctxt_mask = ctxt.abs().sum(dim=-1) > 0
                ctxt_mask = ctxt_mask.to(device, non_blocking=True).bool()

                try:
                    loss = model.update(
                        vtxt_input=vtxt,
                        ctxt_input=ctxt,
                        ctxt_mask=ctxt_mask,
                        z_seq_gt=z_seq_gt,
                        x_mask=mask,
                        loss=loss_type,
                    )
                except TypeError:
                    loss = model.update(
                        vtxt_input=vtxt,
                        ctxt_input=ctxt,
                        ctxt_mask=ctxt_mask,
                        z_seq_gt=z_seq_gt,
                        mask=mask,
                        loss=loss_type,
                    )
            else:
                raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")

        losses.append(float(loss.item()))

    model.train()
    return float(np.mean(losses)) if losses else float("nan")