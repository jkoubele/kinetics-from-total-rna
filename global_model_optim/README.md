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

Per run, in `results/<experiment>/<model_type>/<dataset>/`:

| file | schema |
|---|---|
| `summary.tsv` | **fixed** — one row per LRT result, with `chi2_is_negative`, timings and provenance |
| `test_results.tsv`, `model_parameters.tsv` | same as the pipeline produces |
| `training_log.tsv` | per-epoch losses: `fit_label`, `stage`, `epoch`, `loss` |
| `run_metadata.json` | experiment, dataset, sizes, seed, git revision, torch version and thread count, host, wall time |

`results/all_summaries.tsv` concatenates every `summary.tsv`. It is a few hundred rows, so that is
the file to copy off the cluster; the rest stays put until a specific run needs investigating.

**Keep `summary.tsv` fixed-schema.** Anything experiment-specific belongs in `training_log.tsv`,
whose format is allowed to change. If the summary drifts, the cross-experiment comparison stops
working, which is the whole point of the harness.

The acceptance criterion is `chi2_is_negative == False` everywhere: a negative chi2 means a reduced
(more constrained) model reached a lower loss than the full one, which cannot happen at an optimum.

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
