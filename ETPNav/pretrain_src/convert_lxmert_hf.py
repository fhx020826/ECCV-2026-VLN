"""
Convert HuggingFace LXMERT (unc-nlp/lxmert-base-uncased) weights
to the format expected by ETPNav pretraining (UNC model_LXRT.pth format).

ETPNav expects keys like:
  bert.embeddings.*
  bert.encoder.layer.*        (language layers)
  bert.encoder.x_layers.*     (cross-modal layers)
  cls.predictions.*

HuggingFace LXMERT uses:
  lxmert.embeddings.*         (or bert.embeddings.*)
  lxmert.encoder.layer.*      (or bert.encoder.layer.*)
  lxmert.encoder.x_layers.*   (or bert.encoder.x_layers.*)
  cls.predictions.*
"""
import sys
import torch


def convert(src_path: str, dst_path: str) -> None:
    print(f"Loading HF LXMERT from: {src_path}")
    state = torch.load(src_path, map_location="cpu")

    # Print first 20 keys to understand structure
    keys = list(state.keys())
    print(f"Total keys: {len(keys)}")
    print("First 20 keys:")
    for k in keys[:20]:
        print(f"  {k}  {state[k].shape}")

    # Determine prefix
    prefix = None
    if any(k.startswith("lxmert.") for k in keys):
        prefix = "lxmert"
        print("\nDetected HuggingFace LXMERT format (prefix='lxmert')")
    elif any(k.startswith("bert.") for k in keys):
        prefix = "bert"
        print("\nDetected BERT-style format (prefix='bert') — may already be compatible")
    else:
        print("\nWARNING: Unknown key format, attempting no-op conversion")

    new_state = {}
    for k, v in state.items():
        if prefix == "lxmert":
            # lxmert.* → bert.*
            new_k = k.replace("lxmert.", "bert.", 1)
        else:
            new_k = k
        new_state[new_k] = v

    # Verify key coverage
    lang_keys = [k for k in new_state if "bert.encoder.layer" in k]
    x_keys = [k for k in new_state if "bert.encoder.x_layers" in k]
    print(f"\nConverted: {len(lang_keys)} lang_encoder keys, {len(x_keys)} x_layer keys")

    torch.save(new_state, dst_path)
    print(f"Saved converted weights to: {dst_path}")
    print("Done.")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "pretrain_src/datasets/pretrained/LXMERT/lxmert_hf.bin"
    )
    dst = sys.argv[2] if len(sys.argv) > 2 else (
        "pretrain_src/datasets/pretrained/LXMERT/model_LXRT.pth"
    )
    convert(src, dst)
