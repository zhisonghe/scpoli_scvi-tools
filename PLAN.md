# scPoli on scvi-tools: feasibility and plan (v2)

## Status (2026-11-10)

| Phase | State |
|---|---|
| 0 Environments, golden outputs | Done: synthetic (`tests/golden/synthetic`) and pancreas tutorial, 3 seeds (`data/golden/pancreas_seed*`, from `data/pancreas_sparse.h5ad`). |
| 1 Data layer | Done (`scpolivi/data`). |
| 2 Module, 6 recon losses | Done. Forward-equivalent to scArches (≤1e-5) for nb, zinb, poisson, mse, encoder/decoder-only injection, BN, 1–3 hidden layers, 1–2 condition keys. |
| 3 Training (splitter, plan, early stopping) | Done. |
| 4 Prototypes | Done. Loss and update functions match scArches' trainer exactly. |
| 5 Query mapping, shot_surgery | Done (shot_surgery generalized to several condition keys). |
| 6 scArches checkpoint import | Done (`SCPOLI.from_scarches`). Latents match scArches to 1e-4 for reference and query checkpoints. |
| 7 Benchmark | Synthetic: at 2.4k cells, reference kNN accuracy 0.962 (scArches 0.935), query label transfer 0.798 (scArches 0.787). Pancreas tutorial, 3 seeds (`scripts/run_pancreas_scpolivi.py`, `scripts/eval_pancreas.py`): query accuracy 0.972–0.977 (scArches 0.974–0.978); kNN cell-type accuracy, cell-type ASW, study mixing and within-study neighbour overlap all within scArches' seed range. ATAC and a large atlas still pending. |
| 9 Projection | `get_latent`/`classify` on new AnnData work as in scArches (cell-type columns optional, unknown conditions rejected with a clear error); in addition, gene order is checked. |
| 10 Graph Laplacian loss (scANCVI) | Done: `set_graph`, `train(graph_weight=..., graph_sampling=...)`, `fine_tune`, `scpolivi.graph.construct_sample_graphs`. In-batch estimator matches scANCVI's loss exactly; a neighbour-sampling estimator targets the same objective. Pancreas (seed 0 only): within-study neighbour overlap 0.238 → 0.245–0.249, study mixing 0.47 → 0.42–0.45, label transfer unchanged at weight 10, lower at 100. |
| 8 Docs, release | Not started. |

Environments: `.venv` (runtime, no scArches) and `.venv-parity` (adds scArches, `pandas<3`, because
scArches scPoli fails with pandas 3: positional `Series[...]` indexing). Tests: `.venv/bin/python -m pytest`.

### Deviations from scArches (implemented)
1. `bernoulli`: scArches crashes because its decoder returns nothing. Implemented properly here (BCE on `x>0`).
2. `nb_dist`: scArches crashes (list-valued `n_conditions`). Treated as `nb`, with a warning.
3. `use_mmd`/`mmd_*`: dead code in scArches. Rejected on checkpoint import if `use_mmd=True`.
4. `freeze_expression`: no effect in scArches; accepted and ignored here as well.
5. `log_distance` in `classify`: unused in scArches; accepted and ignored here as well.
6. `get_prototypes_info` typo (`"l"`): fixed, so labeled prototypes get their names.
7. Stratified split: unlabeled cells are shuffled (seeded); scArches takes them in order.
8. `reload_best`: scArches stores `state_dict()` references, so only prototypes revert. Here weights and prototypes both revert (deep copy).
9. Stratified sampling with several condition keys crashes in scArches; here weights use the combined condition.
10. Poisson rate is not multiplied by library size (scArches behavior). `poisson_library_scaling=True` opts in to scaling.
11. `seed=None` by default: uses `scvi.settings.seed` instead of reseeding with 2020.
12. Prototypes without labeled cells are NaN (uninitialized) and skipped, instead of producing NaN losses.
13. `classify(use_mean=False)` samples z as scArches does. On under-trained models this costs a lot of
    accuracy (synthetic: 0.27 sampled vs 0.47 mean), so `use_mean=True` is recommended.

### Not yet covered
- Multi-GPU/DDP (epoch-end prototype passes run on one device), minified AnnData.

Sources reviewed (in `external/`):
- scArches `d35cde6` (2026-06): `models/scpoli/`, `trainers/scpoli/`, `dataset/scpoli/`, `models/base/_base.py`
- scvi-tools `1.5.2` (`ea5d9e8b`, 2026-11-09)
- scancvi `0.1.0`: used only as an example of a standalone package built on scvi-tools internals

## 0. Requirements (decided)

1. **Standalone package** (working name `scpolivi`). It depends on `scvi-tools` only. It is unrelated to scancvi.
2. **Full scPoli functionality**, including multiple `condition_keys` and multiple (hierarchical) `cell_type_keys`.
3. **No runtime dependency on scArches.** The package must still be able to read scArches scPoli checkpoints.
4. **API policy:** use public scvi-tools APIs where they cost nothing. Use internal classes (`scvi.data.fields._*`,
   `scvi.module._vae.VAE`, `scvi.train._trainingplans`, `scvi.model.base._archesmixin`, …) wherever they make the
   implementation simpler or more complete. Pin `scvi-tools>=1.5.2,<1.6`, because internals can move between minor versions.
   Add a CI job against scvi-tools `main` to catch breakage early.

## 1. Verdict

**Feasible.** None of scPoli's features is blocked by scvi-tools. They fall into three groups:
- **Reused as-is:** encoder/decoder building blocks, the NB/ZINB/Poisson likelihoods, Lightning training,
  save/load, the scArches weight-padding logic, categorical-field extension for query data, minification, device handling.
- **Reused with a thin subclass:** the VAE module, data fields, data splitter, training plan, and ArchesMixin freezing.
- **New code:** prototypes (labeled and unlabeled, hierarchical), multi-key condition embeddings, the extra
  likelihoods (MSE, Bernoulli), and the scPoli-specific API (`classify`, `add_new_cell_type`, `shot_surgery`,
  `get_prototypes_info`, `get_conditional_embeddings`, obs_metadata).

I estimate the new code at **roughly 2.5–3.5k lines including tests**. That is about the size of scArches' scPoli (~3k lines),
but this version gets Lightning, minification, MPS/CUDA, DDP-readiness, and the scvi ecosystem on top.

## 2. Full feature inventory → implementation mapping

### 2.1 Model / architecture (`scpoli.py`, `scPoli.__init__`)

| scPoli feature | Implementation in `scpolivi` |
|---|---|
| `condition_keys` (≥1), one `nn.Embedding(n_cond_k, emb_dim_k, max_norm)` per key, concatenated | Custom `CategoricalJointObsField` under key `"conditions"`. It keeps one category mapping per key and supports `extend_categories`. The module holds one `scvi.nn.Embedding` per key in `EmbeddingModuleMixin.embeddings_dict` (`"cond_0"`, …). ArchesMixin's dim-0 padding and its rule that `_embeddings_dict.*` parameters stay trainable then cover query extension for free. |
| `conditions_combined` (joint key, drives NB/ZINB dispersion `theta[:, combined]`) | An internal obs column `_scpoli_conditions_combined`, registered as scvi's `BATCH_KEY`. That gives `dispersion="gene-batch"`, which is exactly scPoli's theta. The batch-aware scvi utilities then work for free (`get_normalized_expression(transform_batch=…)`, per-batch library priors). |
| `embedding_dims` (int or per-key list), `embedding_max_norm` | `scvi.nn.Embedding(..., max_norm=...)`. `Embedding.extend` keeps `max_norm`. |
| `inject_condition` ⊆ {encoder, decoder} | Module flags. The concatenated embedding goes into the encoder and/or decoder input. |
| `hidden_layer_sizes` (list, default `[ceil(sqrt(n_genes))]`), `latent_dim`, `dr_rate`, `use_bn`, `use_ln` | `scvi.nn.FCLayers` takes a single `n_hidden` for all layers. scPoli allows **heterogeneous widths** (e.g. `[256, 64]`), so we need a small `FCLayersList` subclass, or build the encoder/decoder from `torch.nn` blocks that mirror scPoli's order: Linear → BN/LN(no affine) → ReLU → Dropout. Encoder variance = `exp(logvar)+1e-4`. Decoder layers are bias-free, as in scPoli. |
| `recon_loss`: `nb`, `zinb`, `nb_dist`, `poisson`, `mse`, `bernoulli` | `nb`/`zinb`/`poisson` use `scvi.distributions` with softmax scale × observed size factor (`x.sum(1)`). `nb_dist` is an alias of `nb`. `mse` is a squared-error loss on raw `X` with a ReLU head (scPoli semantics, not a Gaussian likelihood). `bernoulli` is BCE with a sigmoid head. scArches' bernoulli path is actually **broken**, because `Decoder.forward` doesn't return; we implement it properly. |
| `beta`, `use_mmd`, `mmd_on`, `mmd_boundary` | **Dead code in scArches**: `use_mmd` is hard-wired to `False` and `mmd` is not even imported. We accept these arguments for checkpoint compatibility, warn, and ignore them. |
| `cell_type_keys` (≥1, hierarchical), `cell_types` dict {name: [levels]}, `unknown_ct_names` (list), `labeled_indices` | Custom `HierarchicalLabelsField(BaseJointField)` under key `"labels"`, of shape (n_cells, n_levels). It uses **one shared category namespace across levels**, as in scPoli (a name that appears at two levels shares one prototype). Any value in `unknown_ct_names` maps to −1. It supports `extend_categories` on transfer and stores the `cell_types` name→levels map in its state registry. `labeled_indices` becomes a `NumericalObsField` (`_scpoli_labeled` ∈ {0,1}). |
| `share_metadata`, `obs_metadata` (first obs row per condition) | Stored in a model attribute (`obs_metadata_`), saved via `attr_dict`, and concatenated on query load. |
| Prototypes `{"mean","cov"}` labeled, `{"mean"}` unlabeled | Module **buffers** `proto_mean`, `proto_cov`, `proto_unlab`, plus `n_ref_labels` / `new_proto_mask`. Buffers go into `state_dict`, so save/load, best-checkpoint reload, and ArchesMixin padding all handle them. |

### 2.2 Training (`scPoliTrainer`, `scPoli.train`)

| scPoli option | Implementation |
|---|---|
| `n_epochs`, `lr`, `eps` (Adam eps 0.01!), `weight_decay` 0.04 except on embeddings | `SCPoliTrainingPlan.configure_optimizers` with two param groups. This is the internal-API part: we override instead of using `optimizer_creator`. |
| `alpha_kl`, `alpha_epoch_anneal` (default 100 in `.train`), `alpha_iter_anneal` | `max_kl_weight`, `n_epochs_kl_warmup`, `n_steps_kl_warmup` in scvi's `TrainingPlan`. These map 1:1 (scPoli starts at epoch 0 → weight 0). |
| `pretraining_epochs` (default `floor(0.9·n_epochs)`; every epoch is pretraining when there are no `cell_type_keys`) | Gate in the training plan using `current_epoch`. |
| `prototype_training`, `unlabeled_prototype_training`, `eta`, `unlabeled_weight`, `p_prototype_loss` | Plan and module loss kwargs. |
| `clustering` (`leiden` / `kmeans`), `clustering_res`, `n_clusters` | Initialization of unlabeled prototypes at the first prototype epoch. Leiden goes through scanpy (optional extra) and KMeans through sklearn. |
| Epoch-end updates: labeled prototypes recomputed from the **sampled** latent of the whole train split (only the `new_prototypes` mask during surgery); unlabeled prototypes get one Adam step on the unlabeled loss | `on_train_epoch_end` hook. It does an inference pass over the train indices of the `DataSplitter` (`self.trainer.datamodule`). The unlabeled prototypes use a separate optimizer owned by the plan and are written back into the buffer. |
| Labeled loss: for each class present, the mean distance to its prototype; **summed** over classes. A cell contributes to every level it is labeled at. | Ported verbatim, vectorized with `scatter`/`index_add`, with a parity test against the loop version. |
| `train_frac` (0.9) with **stratified split** by the finest cell-type level (labeled cells) or by `conditions_combined` | Custom `SCPoliDataSplitter(DataSplitter)` that overrides index generation. Also supports scvi's `validation_size`. |
| `use_stratified_sampling` (WeightedRandomSampler, weight ∝ 1/condition count), `n_samples` (samples per epoch) | The splitter's train dataloader uses scvi's `AnnDataLoader(sampler=…)`. |
| `batch_size`, `clip_value`, `seed`, `n_workers`, `monitor` | Lightning `gradient_clip_val` with `gradient_clip_algorithm="value"`, `scvi.settings.seed`, `dataloader num_workers`, progress bar. |
| `EarlyStopping` (metric `val_unweighted_loss`, patience 20, threshold, reduce_lr with lr_patience 13 and lr_factor 0.1), **ES disabled during pretraining**, `reload_best` | A custom `SCPoliEarlyStopping` callback that skips epochs before `pretraining_epochs`. The plan's `reduce_lr_on_plateau` covers the LR part, and we log `validation_unweighted_loss` to match the monitored metric. `reload_best` uses scvi's `SaveCheckpoint` / `load_best_on_end`, and the prototypes come back with it because they are buffers. |
| `freeze` during query (BN `affine=False`, `track_running_stats=False` for frozen BN) | See 2.3. |

### 2.3 Query mapping / transfer

| scPoli feature | Implementation |
|---|---|
| `load_query_data(adata, reference_model, labeled_indices, unknown_ct_names, freeze, freeze_expression, remove_dropout, return_new_conditions, map_location)` | Override of `ArchesMixin.load_query_data`. `super()` handles field transfer with `extend_categories` (conditions, combined, labels), model re-instantiation, and weight padding. After that: (a) apply **scPoli freezing**, where only embeddings and `px_r` (theta) are trainable; scArches' `freeze_expression` branch has **no effect** in the original code, and we reproduce that and document it; (b) apply the BN handling; (c) set dropout to 0 when `remove_dropout`; (d) fix the reference prototypes, mark new labeled cell types for init and update, re-cluster the unlabeled prototypes; (e) return the new conditions when asked. |
| `shot_surgery` (no training: map each new condition to the reference condition with lowest recon loss and copy its embedding and theta; optional fine-tuning) | Port. Lift the single-key restriction by trying each key separately (for multiple keys, we document how the search works per key). Uses a `get_recon_loss` helper. |
| `_load_expand_params_from_dict` | Covered by ArchesMixin. |

### 2.4 Inference / analysis API

| scPoli method | Implementation |
|---|---|
| `get_latent(adata, mean)` | `get_latent_representation(adata, give_mean=…)` plus a `get_latent` alias. |
| `classify(adata, prototype, p, get_prob, log_distance, scale_uncertainties)` → per-level `{preds, uncert, weighted_distances}` | Module `classify(z, classes_list)` and model-level loop over levels, restricted to the prototype indices of each level. Note: in scArches, `log_distance` computes a value it **never uses**. We keep the argument and implement the obvious intent (`log1p` on the distance) behind a flag, with the default reproducing scArches. |
| `add_new_cell_type(name, obs_key, prototypes, x, c)` | Port: grows the label registry, `proto_mean`, and `proto_cov` (buffer resize helper). |
| `get_prototypes_info(prototype_set)` | Port. Fixes the scArches typo `prototype_set == "l"`, which means truth names are never filled for labeled prototypes. |
| `get_conditional_embeddings()` → AnnData per key | Port, with `obs_metadata` attached. |
| `update_labeled_indices` | Port (the labeled-flag field is re-registered). |
| `save` / `load` | scvi `BaseModelClass` format. |
| `get_normalized_expression`, `differential_expression`, `posterior_predictive_sample`, `minify_adata` | Free from `VAEMixin`/`RNASeqMixin`, as long as the module follows the `VAE` inference/generative contract. These are **new capabilities** that scArches never had. |
| `scarches.dataset.reads_to_fragments` (ATAC helper) | Port into `scpolivi.pp`. |

### 2.5 Reading scArches checkpoints without scArches

scArches `save()` writes three files: `model_params.pt` (state_dict), `attr.pkl` (a pickled dict of public attributes: lists, dicts,
pandas DataFrame, torch tensors), and `var_names.csv`. None of these need scArches classes to unpickle.
`SCPOLI.from_scarches(dir, adata)` will:
1. read the three files with `pickle`, `torch.load(map_location)`, and `numpy`, then rebuild the init params;
2. register the adata with categories in **scArches' encoder order**, which preserves the embedding rows and theta columns;
3. map parameter names (`CondLayers.expr_L`/`cond_L` → our first-layer weight split, `embeddings.i` → `cond_i`,
   `theta` → `px_r`, prototypes → buffers);
4. run a forward-equivalence self-check on a few cells.

The parity tests that build reference checkpoints need scArches, but only in a **dev-only** test environment. It is never a runtime or install dependency.

## 3. scArches behaviors to decide on (faithful vs. fix)

Found while reading the code. Default plan: **reproduce the intended behavior, fix crashes, and document every deviation.**
1. The `bernoulli` decoder path returns `None`, so it crashes. → Fix.
2. `use_mmd`/MMD is dead code. → Accept the args and ignore them.
3. `freeze_expression` has no effect. → Reproduce; optionally add a real option later.
4. `log_distance` is unused. → Default keeps scArches behavior.
5. `get_prototypes_info` typo (`"l"`). → Fix.
6. `nb_dist` dispersion uses `one_hot_encoder(batch, n_conditions)` with a list-valued `n_conditions`, which looks broken for that mode. → Treat as `nb` with combined-condition dispersion.
7. The stratified split never shuffles the unlabeled cells. → Shuffle them (seeded).
8. Prototypes use the **sampled** z, not the mean. → Reproduce, with an option to use the mean.

## 4. Package layout

```
scpolivi/
  pyproject.toml          deps: scvi-tools>=1.5.2,<1.6; extras: [leiden]=scanpy+igraph+leidenalg, [dev]=pytest, scarches(test only)
  src/scpolivi/
    __init__.py
    _constants.py          registry keys: CONDITIONS, LABELS (hierarchical), LABELED_FLAG, INDICES
    data/_fields.py        HierarchicalLabelsField, ConditionsField (CategoricalJointObsField + combined batch col)
    data/_splitter.py      SCPoliDataSplitter (stratified split, weighted sampler, n_samples)
    nn/_layers.py          heterogeneous-width encoder/decoder, likelihood heads (nb/zinb/poisson/mse/bernoulli)
    module/_module.py      SCPOLIVAE(EmbeddingModuleMixin, BaseMinifiedModeModuleClass)
                             inference/generative/loss, prototype buffers, classify, prototype losses
    train/_plan.py         SCPoliTrainingPlan (param groups, pretraining gate, prototype init/update hooks)
    train/_callbacks.py    SCPoliEarlyStopping
    model/_model.py        SCPOLI(ArchesMixin, VAEMixin, RNASeqMixin, BaseMinifiedModeModelClass)
                             setup_anndata, train, classify, add_new_cell_type, get_prototypes_info,
                             get_conditional_embeddings, load_query_data, shot_surgery, from_scarches
    model/_scarches_io.py  checkpoint reader + parameter mapping
    pp/_atac.py            reads_to_fragments
  tests/                   unit tests (synthetic_iid), parity tests (marked, need scarches dev env)
  docs/ + notebooks/       ports of scpoli_surgery_pipeline and scpoli_ATAC tutorials
```

## 5. Phased plan

**Phase 0 – Environment and baselines (1 day)**
- Create a uv project. Install the runtime env (editable `external/scvi-tools`) and a separate dev/parity env that adds
  `external/scarches`.
- Run the two scArches tutorials (`notebooks/scpoli_surgery_pipeline.ipynb`, `scpoli_ATAC.ipynb`) and save their checkpoints,
  latents, predictions, and uncertainties as **golden outputs**. scArches has no scPoli unit tests, so these are the reference.

**Phase 1 – Data layer (2 days)**
- `ConditionsField`, `HierarchicalLabelsField`, labeled flag, combined-batch column, `setup_anndata`.
- Tests: multi-key registration, unknown names → −1, shared namespace across levels, transfer with `extend_categories`
  (new conditions, new cell types), and the category order matching scArches encoders.

**Phase 2 – Module and likelihoods (2–3 days)**
- Heterogeneous-width encoder/decoder, multi-key embeddings, `inject_condition`, all six recon losses, `px_r` gene-batch.
- **Forward-equivalence test:** copy scArches weights into the module. With dropout off and z = mean, recon and KL must match
  to about 1e-5 for every recon loss and for 1 and 2 condition keys. This test also builds most of `from_scarches`.

**Phase 3 – Training without prototypes (2 days)**
- Data splitter (stratified split, weighted sampler, `n_samples`), training plan (param groups, eps, KL anneal, clipping),
  early-stopping callback, reload-best.
- Compare loss curves with scArches pretraining-only runs (same seed and split): same scale and convergence.

**Phase 4 – Prototypes (4–5 days)**
- Buffers, labeled/unlabeled losses (vectorized, checked against a loop reference), epoch-end updates, Leiden/KMeans init,
  separate prototype optimizer, hierarchical classes, `classify`, `get_prototypes_info`, `add_new_cell_type`, `update_labeled_indices`.
- Tests: all labeled, partly labeled, all unknown, multi-level labels, save/load round-trip with prototypes.

**Phase 5 – Query mapping (3 days)**
- `load_query_data` (all options, freezing that matches scArches, prototype masks), `shot_surgery`, obs_metadata merge.
- Tests: new conditions on every key, new cell types in the query, unlabeled query, reference embeddings unchanged after
  query training (weight decay = 0 on embeddings), `return_new_conditions`.

**Phase 6 – scArches checkpoint import (1–2 days)**
- `from_scarches` for every recon loss. Round-trip tests: a scArches checkpoint should give the same latent,
  `classify` output, and recon loss in scpolivi, and query mapping should work from an imported reference.

**Phase 7 – Benchmark against scArches (3 days)**
- Pancreas (surgery tutorial), ATAC tutorial, and one larger atlas subset (~200k cells, ≥2 condition keys, 2 label levels).
  ≥3 seeds each.
- Metrics: scIB integration (reference and reference+query), label-transfer accuracy/F1 per level, uncertainty AUROC,
  novel-cell-type detection (one type held out), runtime and memory on CPU/MPS/CUDA.
- Acceptance: metrics within seed variance of scArches; numerical equivalence where Phase 2 proved it.

**Phase 8 – Docs, tutorials, release (2 days)**
- API docs, the two ported tutorials, a migration guide (scArches args → scpolivi args, list of deviations from §3), CI
  (pinned scvi-tools plus a `main` canary).

**Total: about 4–5 weeks** for a feature-complete, validated v1.

## 6. Remaining risks

- **Internal scvi-tools APIs** (fields, `ArchesMixin` internals, `TrainingPlan` hooks) can change between minor releases.
  Mitigation: a version pin, a canary CI job against `main`, and keeping overrides in a few small, isolated subclasses.
- **Epoch-end full-pass prototype updates:** the cost grows with dataset size, and Leiden on more than 1M cells is slow.
  v1 targets a single device, with options for KMeans or a subsampled init. Multi-GPU (DDP) needs an all-gather of latents and labels, so it is v2.
- **Exact training-trajectory parity is not expected**: RNG streams and dataloader order differ. Parity is defined as
  forward equivalence plus statistical equivalence of the benchmark metrics.
- **Checkpoint import across scArches versions:** older scPoli releases may have pickled slightly different attribute sets.
  Test with checkpoints from at least two scArches versions, if available, including published references.
