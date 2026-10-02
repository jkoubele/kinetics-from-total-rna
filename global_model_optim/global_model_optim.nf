#!/usr/bin/env nextflow

// Runs one global-model fitting experiment per (experiment, model type, dataset) and collects the
// fixed-schema summaries into a single file. That combined file is small, so it is the one worth
// copying off the cluster; the per-run outputs stay behind until a specific case needs a look.

nextflow.enable.dsl = 2

process FitGlobalModelExperiment {
    tag { "${experiment} | ${model_type} | ${dataset_name}" }
    publishDir { "${params.outdir}/${experiment}/${model_type}/${dataset_name}" }, mode: 'copy'
    container params.container_python
    cpus params.fit_cpus
    memory params.fit_memory

    input:
    tuple val(experiment), val(model_type), val(dataset_name), path(dataset_dir)

    output:
    path 'summary.tsv', emit: summary
    path 'model_parameters.tsv'
    path 'test_results.tsv'
    path 'training_log.tsv'
    path 'run_metadata.json'

    script:
    def num_genes_argument = params.num_genes ? "--num_genes ${params.num_genes}" : ''
    def num_introns_argument = params.num_introns ? "--num_introns ${params.num_introns}" : ''
    def num_lrt_tests_argument = params.num_lrt_tests ? "--num_lrt_tests ${params.num_lrt_tests}" : ''
    def verbose_argument = params.verbose ? '--verbose' : ''
    """
    export PYTHONPATH='${projectDir}:${projectDir}/..'\${PYTHONPATH:+:\$PYTHONPATH}
    ${projectDir}/run_experiment.py \\
        --dataset_dir ${dataset_dir} \\
        --dataset_name ${dataset_name} \\
        --model_type ${model_type} \\
        --experiment ${experiment} \\
        --seed ${params.seed} \\
        ${num_genes_argument} \\
        ${num_introns_argument} \\
        ${num_lrt_tests_argument} \\
        ${verbose_argument} \\
        --output_folder .
    """
}

process CollectSummaries {
    publishDir "${params.outdir}", mode: 'copy'
    container params.container_python
    cpus 1
    memory '4 GB'

    input:
    // Every run produces a file called summary.tsv, so they need unique names when staged.
    path summary_files, stageAs: 'summary_*.tsv'

    output:
    path 'all_summaries.tsv'

    script:
    """
    python3 -c "
import pandas as pd, sys
frames = [pd.read_csv(path, sep='\\t') for path in sys.argv[1:]]
combined = pd.concat(frames, ignore_index=True)
combined = combined.sort_values(['experiment', 'model_type', 'dataset', 'test_id', 'tested_parameter'])
combined.to_csv('all_summaries.tsv', sep='\\t', index=False)
print(f'{len(combined)} rows, {int(combined.chi2_is_negative.sum())} with negative chi2')
" ${summary_files}
    """
}

workflow {
    def experiment_names = params.experiments.tokenize(',').collect { it.trim() }
    def model_type_names = params.model_types.tokenize(',').collect { it.trim() }

    dataset_channel = Channel
        .fromList(model_type_names)
        .flatMap { model_type ->
            def dataset_dirs = files("${params.data_dir}/${model_type}_models/*", type: 'dir')
            if (!dataset_dirs) {
                error "No dataset folders found in ${params.data_dir}/${model_type}_models/"
            }
            dataset_dirs.collect { dataset_dir -> [model_type, dataset_dir.name, dataset_dir] }
        }

    run_channel = dataset_channel
        .combine(Channel.fromList(experiment_names))
        .map { model_type, dataset_name, dataset_dir, experiment ->
            [experiment, model_type, dataset_name, dataset_dir]
        }

    FitGlobalModelExperiment(run_channel)
    CollectSummaries(FitGlobalModelExperiment.out.summary.collect())
}
