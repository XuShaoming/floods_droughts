# HMTL Global Pipeline Design

## Goals & Scope
- Extend the existing hierarchical multi-task learning (HMTL) approach to a global, multi-watershed setting that reuses the current experiment/configuration structure.
- Support intermediate targets that feed their predictions back into the final LSTM inputs in real time so gradients propagate through the hierarchy.
- Incorporate static watershed attributes via neural network embeddings to keep the model agnostic to attribute dimensionality.
- Provide end-to-end tooling (data loader, training, inference, and analysis) plus new `config_global.yaml` experiments driving the workflow.
- Non-goals: altering legacy single-watershed pipelines (`train_hmtl.py`, `inference_mtl.py`, etc.) or redesigning visualization assets beyond what the new analysis script requires.

## Requirements & Assumptions
- Multiple CSVs per watershed/scenario live under `data_processed/` and already follow the `{watershed}_{scenario}_combined.csv` pattern used by `dataloader_global.py`.
- Static attributes come from `hspf_CAT_attributes.csv`; we assume a consistent schema across watersheds/models, and encode them with an MLP to compress potentially large vectors.
- Intermediate predictions must be concatenated with the original dynamic features before entering the final LSTM, matching `HierarchicalLSTMModel` behavior, so errors can backprop to intermediate branches.
- Config files can optionally specify per-intermediate loss weights; unspecified targets default to uniform weighting.
- Training/inference artifacts mirror the existing global pipeline layout (`experiments/global_models/<experiment>/...`), including saved scalers and metadata.

## Components

### 1. `dataloader_hmtl_global.py`
**Responsibilities**
- Merge functionality from `dataloader_hmtl.py` (intermediate targets, hierarchical scalers) and `dataloader_global.py` (multi-watershed handling, static attributes, dataset splits).
- Produce pre-windowed tensors for dynamic features, static embeddings, intermediate targets, and final targets, plus metadata for reconstruction.

**Key Concepts**
- `GlobalHierarchicalDataset`: returns `(dynamic_window, static_vector, final_target_window, intermediate_target_window, metadata_index)`.
- Sliding windows follow the configured `window_size`/`stride`, and optionally support many-to-many targets just like the other dataloaders.
- Static attributes (if enabled) are aligned per watershed and repeated for each window; optional scaling occurs via `static_scaler`.
- Metadata dict stores arrays for `watershed`, `scenario`, `timestamp_range`, and index boundaries per split for later reconstruction.

**APIs**
- Initialization arguments superset existing global loader: includes `intermediate_targets`, `scale_intermediate_targets`, `intermediate_loss_weights`, etc.
- `prepare_data()` builds window arrays and populates scalers.
- `create_data_loaders(shuffle_train: bool)` returns loaders and metadata dicts for `train`, `val`, `test`.
- Utility methods persist/load scalers & metadata to disk for inference reuse.

### 2. `models/CTLSTM_Global.py`
**Responsibilities**
- Extend `HierarchicalLSTMModel` to ingest static attributes alongside dynamic features.
- Maintain real-time feedback loop: intermediate predictions are concatenated with original dynamic inputs before the final LSTM, identical to the existing per-watershed HMTL design.

**Architecture**
- Shared embedding network for static attributes (mirrors `CTLSTM` logic) produces a vector broadcast across time steps.
- Intermediate branches: per-target LSTMs with batch norm + dropout; outputs feed both loss computations and the final branch inputs.
- Final branch: LSTM over `[dynamic_features || static_embedding || intermediate_preds]`, followed by BN + dropout + linear projection to final targets.
- Forward returns a dict: `{"intermediate": {target_name: Tensor}, "final": Tensor}`.

### 3. `train_hmtl_global.py`
**Responsibilities**
- CLI entry point for training the hierarchical global model.
- Load experiment config, instantiate dataloader/model, train with multi-loss objective, and manage checkpoints/metrics/logs.

**Flow**
1. Parse CLI options: `--config`, `--experiment`, `--seed`, optional overrides for loss weights and dataloader filters.
2. Resolve experiment via YAML (similar to `train_global.py:118-150`), ensuring required keys exist (`dataset_splits`, `intermediate_targets`, etc.).
3. Instantiate `GlobalHierarchicalDataLoader`, then call `prepare_data()` and `create_data_loaders(shuffle_train=True)`.
4. Build `CTLSTM_Global` with feature/static sizes from the dataloader and config hyperparameters; move to requested device.
5. Define loss weights:
   - `final_loss_weight` scalar.
   - `intermediate_loss_weights`: either a dict keyed by target name or a shared scalar vector (derived from config). They default to uniform `1.0`.
6. Training loop:
   - Iterate over train loader batches, compute predictions, accumulate per-target MSE, combine using weights, backprop, clip gradients.
   - Track metrics per target and log to TensorBoard.
   - Validate each epoch; early stopping via `EarlyStopping` (reused from `train.py`).
7. Save artifacts: `best_model.pth`, `last_model.pth`, `config.yaml`, `model_config.json`, scalers (pickle), metadata (JSON), and training curves.

### 4. `inference_hmtl_global.py`
**Responsibilities**
- Load trained `CTLSTM_Global`, rebuild dataloaders for specified splits, run inference, denormalize predictions, and save both window-level and reconstructed timeseries outputs (for intermediate + final targets).

**Flow**
1. CLI: `--config`, `--experiment`, `--checkpoint`, `--splits`, `--watersheds`, `--scenarios`, `--output-dir`.
2. Load experiment config and training artifacts (model config, saved scalers/metadata).
3. Instantiate dataloader with identical parameters; ensure scalers use saved statistics to keep inference consistent.
4. Iterate through batches, gather predictions and targets, apply inverse transforms separately for intermediate and final targets.
5. Save results:
   - Window-level per (watershed, scenario, split) in CSV/Parquet, including columns `pred_<target>` and `obs_<target>` for all targets.
   - Reconstructed continuous timeseries via weighted overlap or averaging (reuse logic from `inference_global.py`).
   - Metrics JSON (per target, per watershed, overall) for both intermediate and final outputs.
   - `results_manifest.json` summarizing file paths.

### 5. `analysis_hmtl_global.py`
**Responsibilities**
- Consume inference artifacts, compute diagnostics, and produce plots across intermediate and final targets.

**Features**
- Accept CLI args for config/experiment/results directory plus optional filters.
- Extend `analysis_global.py` metrics helpers to handle both intermediate and final targets; export aggregated metrics tables (CSV/JSON).
- Provide plots:
  - Window timeseries: overlay observed vs predicted for selected targets.
  - Reconstructed timeseries: include time range filters, multi-panel layout when plotting both target types simultaneously.
  - Error histograms/residual plots for each target type.

### 6. `config_global.yaml` Updates
- Add `streamflow_hmtl_global` experiment inheriting from `base_global_config` and specifying:
  - `training_strategy: "hmtl_global"`
  - `intermediate_targets: [...]`
  - `final_targets`/`target_cols`
  - `loss_weights` block, e.g.:
    ```yaml
    loss_weights:
      final: 1.0
      intermediate:
        PET: 0.5
        ET: 0.5
        default: 1.0
    ```
  - Optional `static_embedding_layers`, dropout, scheduler tweaks.
- Add inference/analysis experiment entries referencing the new training experiment (paralleling `_inference` configs already present).

### 7. Testing Strategy
- **Unit tests / sanity checks**:
  - Create a minimal script (manual run) to instantiate the new dataloader with two watersheds and ensure tensor shapes align.
  - Smoke-test `train_hmtl_global.py --experiment streamflow_hmtl_global_test` for a few epochs to confirm loss decreases and checkpoints save.
  - Run `inference_hmtl_global.py --splits val` and verify reconstructed outputs share timestamps with the input CSV slices.
  - Execute `analysis_hmtl_global.py` on inference outputs and check generated metrics files list both intermediate and final targets.
- **Manual validation**:
  - Spot-check static attribute embeddings by inspecting saved scaler stats for expected dimensions.
  - Compare metrics between old `analysis_global.py` and the new script when focusing solely on final targets—they should match.

## Open Questions (Resolved)
- **Intermediate feedback**: Intermediate predictions are concatenated with dynamic inputs for the final LSTM; gradients flow from final losses back to intermediate branches.
- **Loss weights**: Config supports per-intermediate weights plus an optional default; missing entries fall back to uniform weighting.
- **Static attributes**: Attributes can be high-dimensional; encode them via an MLP (configurable via `static_embedding_layers`) to reduce dimensionality and capture relevant patterns.

## Implementation Checklist
1. Add `dataloader_hmtl_global.py` and ensure it integrates with both static attributes and hierarchical scalers.
2. Add `models/CTLSTM_Global.py`.
3. Create `train_hmtl_global.py`, `inference_hmtl_global.py`, and `analysis_hmtl_global.py`.
4. Update `config_global.yaml` with the new experiments (training, inference, analysis variants, plus optional `_test` entry).
5. Wire up saving/loading utilities (scalers, metadata) shared across training and inference scripts.
6. Add README snippet or usage comments (optional) describing how to run the new pipeline.
