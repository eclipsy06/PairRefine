# Paper checkpoint

`pairrefine_epoch008.pt` is the portable form of the paper's epoch-8 PairRefine pose head. It contains 68,550 parameters and can be loaded with `torch.load(path, map_location="cpu", weights_only=True)`. The dictionary contains `head_type`, `model_config`, `model_state_dict`, `epoch`, and curated training and provenance metadata.

All 17 tensors were verified to be bitwise equal to the original checkpoint. The file SHA256 differs because local filesystem paths and machine-specific arguments were removed before serialization. `provenance.json` records both hashes and the precise training and checkpoint-selection protocol.

The archived official scores were generated from the original checkpoint. The portable checkpoint has identical weights. A separate release regression ran fresh inference and official evaluation on all 65 targets and reproduced every archived official value at the scorer's output precision; see `../docs/release_verification.json`.
