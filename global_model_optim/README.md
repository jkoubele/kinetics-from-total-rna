# Global model optimizer experiments

Harness for tuning the training of the global models. One experiment = one fitting strategy; the
data, the subsetting and the output schema are held fixed so that differences between runs come
from the optimizer and nothing else.

```
global_model_optim/
  data/<model_type>_models/<dataset>/   exported Nextflow work dirs (not in git, ~500-800 MB each)
  experiments/experiment_00N.py         one module per strategy
  run_experiment.py                     entry point, same for every experiment
  global_model_optim.nf                 Slurm/container wrapper
  nextflow.config
  results/                              published outputs
```

## Data

Each dataset folder is the work directory of a finished `FitGlobalRnaKineticsModel` or
`FitGlobalIntronCoverageModel` task, exported with `../tools/export_work_dir.sh`. It holds the
staged inputs under fixed names plus the `model_parameters.tsv` / `test_results_raw.tsv` the
unmodified pipeline produced, which is what `experiment_001` should reproduce.

## Running

Locally, on a subset:

```bash
export PYTHONPATH=/home/jakub/Desktop/pol-ii-speed:/home/jakub/Desktop/pol-ii-speed/global_model_optim
./run_experiment.py \
    --dataset_dir data/rna_kinetics_models/BCLAF1_RNAi_GSE303836 \
    --model_type rna_kinetics --experiment experiment_001 \
    --num_genes 200 --num_lrt_tests 1 --output_folder /tmp/try
```

Everything, through Nextflow. Docker is on by default, so no `-profile` is needed on the laptop;
`-profile beyer_cluster` switches the executor to Slurm and bumps the resource request:

```bash
nextflow run global_model_optim.nf \
    --experiments experiment_001,experiment_002 \
    --num_genes 1000 --num_lrt_tests 1

nextflow run global_model_optim.nf -profile beyer_cluster \
    --experiments experiment_001,experiment_002 \
    --num_genes 1000 --num_lrt_tests 1
```

`--experiments` is a comma-separated list and is crossed with every dataset and model type, so one
submission gives the whole comparison. Omitting `--num_genes` / `--num_introns` / `--num_lrt_tests`
uses everything.

## Outputs

Each invocation creates its own `results_NNN/` folder next to the pipeline, auto-numbered, so runs
at different settings sit side by side instead of overwriting each other. `--run_name` overrides the
numbering when a run deserves a name. The settings live in `results_NNN/run_spec.json` (experiments,
model types, subsetting, seed, profile, git revision, full command line) rather than in the folder
name.

Per run, in `results_NNN/<experiment>/<model_type>/<dataset>/`:

| file | schema |
|---|---|
| `summary.tsv` | **fixed** — one row per LRT result, with `chi2_is_negative`, timings and provenance |
| `test_results.tsv`, `model_parameters.tsv` | same as the pipeline produces |
| `training_log.tsv` | per-epoch losses: `fit_label`, `stage`, `epoch`, `loss` |
| `run_metadata.json` | experiment, dataset, sizes, seed, git revision, torch version and thread count, host, wall time |

`results_NNN/all_summaries.tsv` concatenates every `summary.tsv`. It is a few hundred rows, so that is
the file to copy off the cluster; the rest stays put until a specific run needs investigating.

**Keep `summary.tsv` fixed-schema.** Anything experiment-specific belongs in `training_log.tsv`,
whose format is allowed to change. If the summary drifts, the cross-experiment comparison stops
working, which is the whole point of the harness.

The acceptance criterion is `chi2_is_negative == False` everywhere: a negative chi2 means a reduced
(more constrained) model reached a lower loss than the full one, which cannot happen at an optimum.

## Experiments

| module | strategy |
|---|---|
| `experiment_001` | the current two-stage LBFGS, unchanged, with the loss curves kept |
| `experiment_002` | refit the full model warm-started from any reduced fit that beats it, repeating until none does |
| `experiment_003` | full model only, fitted from a grid of starting values for the shared LFCs, to test whether the optimum is global |

`experiment_002` reduces exactly to `experiment_001` when no restart is triggered, so it cannot
regress. Its knobs are module constants: `MAX_FULL_MODEL_RESTARTS` (10) and `CHI2_TOLERANCE` (0.1,
below which a negative chi2 is treated as numerical noise rather than a failed fit). It adds
`num_full_model_restarts` to `summary.tsv` and a `stage = 'initial'`, `epoch = 0` row per fit to
`training_log.tsv`, which is the loss at the starting point -- the number that shows whether a warm
start landed where it should.

⚠ The design matrices carry **no intercept column**: the models hold their own intercepts. So a
reduced design is nested in `[1, X]`, not in `X`, and warm-starting the full model from a reduced
fit has to project onto `[1, X]` and absorb the leftover constant into the intercepts. Projecting
onto `X` alone is exact only for a two-level factor, where the reduced matrix is empty. On a
three-level design the gap was +109 (beta) and +8254 (gamma) in a direct check, so this matters for
any dataset with a multi-level factor -- CCR4-NOT above all.

`experiment_003` runs no reduced models, so it produces no chi2 and no LRT rows. Instead it reports
one row per grid point with `init_beta`, `init_gamma`, the loss reached, and the fitted
`final_beta` / `final_gamma`. `INIT_GRID_VALUES` is a module constant. The grid moves every feature
of a parameter together -- beta starts at (b, b, ...) -- since a per-feature grid is hopeless once a
design has more than a couple of columns, and the optimizer separates the components afterwards.

`--datasets` takes a comma-separated list of dataset folder names, so a run can target one dataset
without rearranging `data_dir`.

## Adding an experiment

Copy `experiments/experiment_001.py`, change only the training code, and import everything else
from `rna_kinetics`. Keeping the modules thin is what makes
`diff experiments/experiment_001.py experiments/experiment_007.py` show the actual strategy
difference rather than 800 lines of unchanged code.

Each module must define `EXPERIMENT_DESCRIPTION` and both of:

```python
fit_global_rna_kinetics(global_gene_data, dataset_metadata, device='cpu', verbose=False)
fit_global_intron_coverage(coverage, dataset_metadata, intron_names, device='cpu', verbose=False)
```

each returning `(model_param_df, test_results_df, training_log_df)`.

## Notes

- **Do not use `-resume` while iterating.** Nextflow hashes the `script:` block and the declared
  inputs, and `run_experiment.py` and `experiments/` are neither — so edits to the fitting code do
  not invalidate the cache and you will silently rerun nothing.
- Gene and intron subsets are drawn at random under `--seed`, so "1000 genes" is the same 1000
  genes for every experiment on a given dataset. LRTs are capped by taking the first n, since their
  order is meaningful.
- `--num_genes` constrains both model types: for `intron_coverage` it samples that many genes and
  keeps all of their introns. `--num_introns` applies to `intron_coverage` only and wins if both
  are given. The problem size is printed before fitting starts, so a run that is accidentally at
  full scale is visible immediately instead of after 45 minutes.
- Docker is always enabled rather than sitting behind a profile, so every fit appears in
  `docker ps`. A fit that runs on the host instead is invisible there, and an interrupted Nextflow
  run can leave those orphaned and burning CPU for as long as they take to finish.
- The fit is deterministic: running the same command twice gives byte-identical outputs.
- A missing `stage_1` in `training_log.tsv` is not an error. Stage 1 only optimizes the shared
  parameters, and when a reduced design matrix has no columns there are none to optimize, so the
  stage is skipped. `rna_kinetics.estimation` behaves the same way.
- Wall times under Nextflow on one machine are not comparable to a single local run: concurrent
  PyTorch processes contend for CPU. Compare timings within a run, not across setups.
- The timeline and trace reports are not written, since their paths are fixed at config-parse time
  and would be overwritten by every run. `wall_time_seconds` in `summary.tsv` covers the timing that
  matters here.
- For the development loop, cut datasets rather than scale. The cluster wall time is dominated by
  the number of tasks, and a single dataset at 200 genes runs locally in about 7 seconds -- which is
  enough to reproduce the negative-chi2 failure on
  `minor_spliceosome_inhibition_GSE294209` / `rna_kinetics`.
