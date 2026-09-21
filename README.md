# kinetics-from-total-rna

This repository contains a pipeline that uses total RNA-seq to estimate changes in the kinetic
parameters of transcription and RNA processing: the transcription initiation rate, the elongation
speed, the intron splicing speed, and the RNA degradation rate. Similarly to a differential
expression analysis, the pipeline compares samples from different conditions (for example treated
versus control), and estimates how these parameters change between them. Because total RNA-seq
observes RNA at steady state, these parameters can be identified only relative to each other, and
every effect size reported by the pipeline is therefore a fold change of a ratio of two of them (see
the section [What the model estimates](#what-the-model-estimates) below).

The code is organized as a Nextflow pipeline. To run it, you will need FASTQ
data of total RNA-seq; please read the documentation below for more details.

The model itself is implemented using PyTorch; the code can be found in
the [rna_kinetics](rna_kinetics) folder
(which can also be installed as a pip package).

> **Note on the preprint.** Our preprint is
> [available on bioRxiv](https://doi.org/10.1101/2025.08.24.672013), but it describes an earlier
> version of the method. The code in this repository has changed substantially since then
> (including which quantities are reported, and how they are named), and it no longer corresponds
> to the description in the preprint. We are currently preparing an updated manuscript; until it is
> finished, please use this README as the description of what the code actually does.
>
> The current draft of the manuscript is included in this repository as
> [preprint_draft_2026-08-21.pdf](preprint_draft_2026-08-21.pdf). Please note that the draft is not
> finished, and several of its sections are only placeholders. However, the sections
> *Dynamical Model of RNA Metabolism* and *Inference Model* are mostly complete; they describe the
> underlying physical model of the RNA metabolism, and the inference model implemented in this
> repository. We therefore recommend reading these sections if you want to understand what the
> pipeline is fitting.

## What the model estimates

Every effect size reported by the pipeline is a log fold change (LFC) of a ratio of two rates, and
never of a single rate. This is a consequence of the data we are using: total RNA-seq observes RNA
at steady state, and the steady-state abundances constrain the kinetic rates only relative to each
other. If we scale all rates by a common factor, the predicted abundances stay the same, and
therefore no individual rate can be identified.

The pipeline reports the following parameters, using the same names as in the `parameter_type`
column of the output:

| Parameter | Interpretation |
| --- | --- |
| `lfc_init_over_deg` | LFC(initiation rate) − LFC(degradation rate) |
| `lfc_elong_over_deg` | LFC(elongation speed) − LFC(degradation rate) |
| `lfc_splice_over_deg` | LFC(splicing speed) − LFC(degradation rate) |
| `lfc_elong_over_splice` | LFC(elongation speed) − LFC(splicing speed) |

Therefore, a significant result tells us which pair of rates has changed relative to each other, but
not which member of the pair is responsible for the change. For example, a positive
`lfc_elong_over_deg` is equally consistent with a faster elongation and with a slower degradation.
It may help to interpret the parameters jointly: if `lfc_elong_over_deg` and `lfc_splice_over_deg`
shift by a similar amount, while `lfc_elong_over_splice` stays near zero, then a change of the
degradation rate is a more parsimonious explanation than a simultaneous change of both the
elongation and the splicing speed.

Some of these ratios also correspond to quantities that are used elsewhere. For example,
`lfc_init_over_deg` is the LFC of the steady-state abundance, which is what a conventional
differential expression analysis reports. Further ratios can be obtained by subtracting the
parameters from each other: `lfc_init_over_deg` − `lfc_elong_over_deg` = LFC(initiation rate) −
LFC(elongation speed) describes a change of the polymerase density along the gene body. However,
only the four parameters listed in the table are currently covered by the likelihood-ratio tests.

Note that `lfc_init_over_deg` is always estimated per gene, in all of our models. Some of the models
estimate a single effect for the whole dataset instead of a per-gene one (these variants are
described in the section [Model variants](#model-variants) below), but such a dataset-wide version
of `lfc_init_over_deg` would not be identifiable, since RNA-seq is compositional data that captures
only relative abundances; an expression shift shared by all genes is therefore indistinguishable
from a change of the sequencing depth, and is absorbed by the library size factors. The same holds for any dataset-wide ratio that involves the initiation rate (for example
`lfc_init_over_elong` would not work either), because the initiation rate enters the data only
through the overall transcript abundance. The remaining ratios are not affected by this, since they
are obtained from comparisons within a single gene (intron versus exon read counts, and the coverage
profile along an intron), in which the library size cancels out; these within-gene signals are then
pooled across genes to obtain a dataset-wide estimate. For this reason, the global RNA kinetics
model tests only `lfc_elong_over_deg` and `lfc_splice_over_deg`.

The effect sizes are always reported as log2 fold changes. Their significance is assessed by
likelihood-ratio tests, with the null hypothesis that the corresponding LFC is equal to zero (see
*LRT contrasts* below).

## Model variants

The pipeline provides six models, which differ in two aspects. You can enable any combination of them
in the ```dataset_params.yaml``` file, and each of them writes its results into a separate output
subfolder.

The first aspect is which data are used by the model:

* **RNA kinetics models** are fitted to the exon read counts, intron read counts and intron coverage
  jointly. Since they also use the read counts, they can resolve all three parameters that are
  referenced to the degradation rate (`lfc_init_over_deg`, `lfc_elong_over_deg` and
  `lfc_splice_over_deg`).
* **Intron coverage models** are fitted to the intron coverage only, and report a single parameter,
  `lfc_elong_over_splice`. The coverage along an intron is informative about the ratio of the
  elongation and splicing speeds, and does not depend on the degradation rate at all. These models
  are therefore simpler, but they do not estimate the three parameters that are referenced to the
  degradation rate.

The second aspect is the granularity, i.e. how many introns share one estimated effect. A coarser
granularity pools more data, and therefore has a higher statistical power; on the other hand, it
assumes that the effect is indeed shared.

The six resulting models are listed below. Each cell of the table gives the parameter of the
```dataset_params.yaml``` file that enables the corresponding model:

| Granularity | RNA kinetics model | Intron coverage model |
| --- | --- | --- |
| **Global**, one effect for the whole dataset | *fit_global_rna_kinetics_model* | *fit_global_intron_coverage_model* |
| **Per gene**, one effect per gene, shared by its introns | *fit_rna_kinetics_model* | *fit_gene_specific_intron_coverage_model* |
| **Per intron**, a separate effect for every intron | *fit_intron_specific_rna_kinetics_model* | *fit_intron_specific_intron_coverage_model* |

We recommend starting with the global and per-gene models of both families, which is also the
default setting in the [dataset_params_template.yaml](dataset_params_template.yaml) file. The global
models tell you whether there is a dataset-wide shift, and they are well powered even for small
experiments, while the per-gene models identify which genes are driving it.

The two per-intron models are disabled by default. They are the most granular, but also the least
powered, since each estimate is based on a single intron; moreover, the per-intron RNA kinetics
model is the most expensive one to compute. These models can be useful if you expect the introns
within a gene to behave differently, otherwise the per-gene models answer the same question with
more data behind each estimate.

The granularity also affects the content of the output. The per-gene and per-intron models perform
an additional regularization step, in which the effect sizes are shrunk towards zero using an
empirical Bayes prior fitted by [ashr](https://github.com/stephens999/ashr) across all genes; their
```test_results.tsv``` file therefore contains both `l2fc_unregularized` and `l2fc_regularized`. The
global models estimate a single effect, so there is nothing to fit the prior on, and no shrinkage is
applied; their output contains a single estimate, `l2fc_mle`.

## Running the pipeline

Besides the FASTQ data, you will need to prepare two files to run the pipeline:

* ```samplesheet.csv```, containing sample annotation.
* ```dataset_params.yaml```, which specifies dataset metadata and several paths on your filesystem.

### Preparing samplesheet

The file ```samplesheet.csv``` needs to contain the following 4 columns:

* *sample*, specifying sample names (the names can be arbitrary).
* *fastq_1*, name of FASTQ file with reads 1 (compressed by gunzip).
* *fastq_2*, same as above for reads 2.
* *strandedness*, specifying the strandedness (read orientation) of the samples. Currently, only values ```forward```
  and ```reverse``` are supported.
  (The most common Illumina RNA-seq protocols use ```reverse``` orientations).

Besides these 4 mandatory columns, the samplesheet can contain arbitrary explanatory variables (e.g., genotype,
intervention, etc.), that can
be used in the dataset parameter file to specify a design formula (see below).

### Preparing dataset parameters

The file ```dataset_params.yaml``` can be created by copying
the [dataset_params_template.yaml](dataset_params_template.yaml) file
and filling it according to the comments. We will now discuss several parameter details:

* **Reference genome and transcriptome** (*gtf_file*, *genome_fasta*, *transcriptome_fasta*, and *gtf_source*): In
  the pipeline, we are aligning reads both to the genome and transcriptome. Therefore, the reference
  files (*gtf_file*, *genome_fasta*, and *transcriptome_fasta*) need to be all compatible (from the same source and
  version).

  We are using [STAR](https://github.com/alexdobin/STAR) for aligning reads to the genome; please follow the
  recommendations
  from the STAR documentation for details on how to choose the reference genome files
  (see section *2: Generating genome indexes* of
  the [manual](https://github.com/alexdobin/STAR/blob/master/doc/STARmanual.pdf)). Typically, the *primary assembly* version of the annotation is the correct one to use.

  Since Ensembl and Gencode annotations slightly differ, the used source needs to be stated in the *gtf_source*
  parameter
  (supported values are ```"ensembl"``` and ```"gencode"```).

  We are using [tximport](https://bioconductor.org/packages/release/bioc/html/tximport.html) package to import transcript abundances from Salmon. Tximport requires
  argument *ignoreTxVersion*, which typically should be true for human or mouse reference annotation from Ensemble, and false 
  for most other references. 
* **STAR and Salmon indices**: You can optionally pre-build the STAR index and/or Salmon index yourself and provide the
  path to it. If no path
  is provided (the parameter is left ```null```), the index will be built from the provided genome/transcriptome files.
* **Design formula**: the parameter *design_formula* uses R syntax to generate a design matrix from the formula (
  see [R documentation](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/formula) for details).
  The explanatory variables used in the formula need to be the columns of the ```samplesheet.csv```.
* **LRT contrasts**: We are using likelihood-ratio tests to assess the significance of LFC parameters. The parameter
  *lrt_contrasts* specifies a list of tests that are going to be performed.

  For categorical variables, please specify (for each test) the *variable* (name of the column in the samplesheet), and 
  comparison groups *group_1* and *group_2* (levels of the variable). The null hypothesis being tested is whether the LFC
  parameters between *group_1* and *group_2* are equal to 0.

  For example, assume that the samplesheet contains column *genotype*, with levels *wild_type*, *knockout_1*, and *knockout_2*.
  To test whether each of the knockout group is different from the wild-type, you can specify:
  ```
  lrt_contrasts:
  - variable: 'genotype'
    group_1: 'knockout_1'
    group_2: 'wild_type'
    
  - variable: 'genotype'
    group_1: 'knockout_2'
    group_2: 'wild_type'
  ```
  You can include as many tests as you wish, e.g., compare also groups in some other explanatory variable, or include also
  the comparison between *knockout_1* and *knockout_2* in the *genotype* variable.

  For a continuous variable, specify only the *variable* parameter, without the comparison groups. 

  We currently support testing only the main-effect term labels from an R terms object, not interaction or other
  higher-order term labels. If you wish to test e.g., an interaction term between two variables, please manually create 
  the interaction variable as a separate column in the *samplesheet*; then, you can add it to the design formula 
  and also include it in the tests.

* **Stage**: Our pipeline consists of 2 workflows: [pre-processing](./workflows/preprocessing.nf)
  and [modeling](./workflows/modeling.nf).
  The pre-processing contains read alignment and related steps, and is supposed to be
  run only once per dataset. The modeling, on the other hand, may be run multiple times, e.g, experimenting with
  different
  design formulas.

  The parameter *stage* specifies whether both or only one workflow should be run. Possible values are
  ```"all"``` (to run both workflows), ```"preprocess"``` and ```"model"```. 
  
  Please note that you can also generally use the ```-resume``` argument for the ```nextflow run``` command. Setting ```stage: 'model'```
  is simply an orthogonal way to re-use preprocessed data, independent of the caching done via  Nextflow work folder.

### Executing the pipeline

Please install [Nextflow](https://www.nextflow.io/) and one of the container engines
[Docker](https://www.docker.com/), [Singularity](https://sylabs.io/singularity/) or
[Apptainer](https://apptainer.org/) on your system. Then, the pipeline can be run by

```commandline
nextflow run main.nf -params-file dataset_params.yaml -profile docker
```

If you are using Singularity or Apptainer, please use ```-profile singularity``` or ```-profile apptainer``` instead.
Our Docker images are used in all cases; Singularity and Apptainer convert them automatically. The images are
fairly large, so we recommend setting ```NXF_SINGULARITY_CACHEDIR``` (or
```NXF_APPTAINER_CACHEDIR```) to a shared folder, so that the conversion is done only once and not for every dataset.

Please note that one of these profiles is always required: without it, Nextflow would try to run the tools directly
on your system, without any container.

We have developed and tested the pipeline with Nextflow 26.04.6. Since newer Nextflow versions occasionally
introduce breaking changes, you can run the pipeline with exactly this version (regardless of the version you have
installed) by setting the ```NXF_VER``` variable; Nextflow will download it automatically:

```commandline
NXF_VER=26.04.6 nextflow run main.nf -params-file dataset_params.yaml -profile docker
```

Please note that the pipeline is computationally demanding. The read alignment is the most costly
part in terms of memory: by default, we request 200 GB of memory for building the STAR and Salmon
indices, and 120 GB for the alignment itself.

The model fitting is demanding as well, although in a different way: it does not need much memory,
but it takes a considerable amount of CPU time, since every gene is fitted separately, and each
likelihood-ratio test requires fitting an additional model. The genes are split into chunks that are
processed in parallel, so this part of the pipeline also benefits from having many cores available.

We therefore recommend running the pipeline on an HPC system. See the Nextflow documentation for
[details](https://www.nextflow.io/docs/latest/executor.html) on how to run the pipeline on your
HPC/cloud system.

For testing purposes, we also provide the *local* profile, which caps the memory requests at 56 GB,
so that the pipeline can be run on a machine with 64 GB of RAM:

```commandline
nextflow run main.nf -params-file dataset_params.yaml -profile local,docker
```

We have tested this profile on a small dataset (C. elegans), but we cannot guarantee that it will be
sufficient for your data. In particular, we have not tested whether the indices of larger genomes
(e.g. human or mouse) can be built within these limits. If the index building runs out of memory,
you can build the STAR and Salmon indices beforehand, and provide the paths to them in the
*star_index* and *salmon_index* parameters. Please note that even then, the alignment itself may
require more memory than the profile allows, depending on the genome and on the size of your FASTQ
files; in that case, we are not aware of any workaround other than using a machine with more memory.

**Note on Slurm**: On our HPC system using Slurm, we noticed the following bug: when several of the
model-fitting processes (that is, any of the processes that run PyTorch) are executed on the same
cluster node, their CPU usages somehow collide, resulting in orders of magnitude slower process
execution. We suspect that this may be related to the underlying BLAS settings and/or our cluster
setup, and we are currently trying to resolve this issue. In the case that you experience similar
behavior, please execute at most one model-fitting process per cluster node.

The crude workaround that we are currently using can be found in the *beyer_cluster* profile in the
[nextflow.config](nextflow.config) file: each of the fitting processes requests substantially more
memory than it actually needs (130 GB, which is somewhat over one half of a node), so that Slurm
cannot place two of them on the same node. Please note that this is only a hack, since the requested
memory is not used, and we are relying on a side effect of the scheduler instead of specifying what
we actually want.

We are using this hack, as the alternatives that we tried were even worse. Setting explicit thread limits
(```OMP_NUM_THREADS```, ```torch.set_num_threads``` and similar) did not resolve the collision in our
tests, and most of the variants made the performance even worse. Requesting the nodes exclusively
(```--exclusive```) does prevent the problem, but it reserves the whole node, while over-requesting
the memory still leaves the remaining cores available for other jobs.

## Contact

If you have any questions or experience any problems with the code,
please open an issue on this repository, or reach out at
[jkoubele@uni-koeln.de](mailto:jkoubele@uni-koeln.de).