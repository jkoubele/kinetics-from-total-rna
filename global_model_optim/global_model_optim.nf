#!/usr/bin/env nextflow

// Runs one global-model fitting experiment per (experiment, model type, dataset) and collects the
// fixed-schema summaries into a single file. That combined file is small, so it is the one worth
// copying off the cluster; the per-run outputs stay behind until a specific case needs a look.

nextflow.enable.dsl = 2

process FitGlobalModelExperiment {
    tag { "${experiment} | ${model_type} | ${dataset_name}" }
    publishDir { "${run_dir}/${experiment}/${model_type}/${dataset_name}" }, mode: 'copy'
    container params.container_python
    cpus params.fit_cpus
    memory params.fit_memory

    input:
    tuple val(run_dir), val(experiment), val(model_type), val(dataset_name), path(dataset_dir)

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
    publishDir { "${run_dir}" }, mode: 'copy'
    container params.container_python
    cpus 1
    memory '4 GB'

    input:
    val run_dir
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

    // Each invocation gets its own results_NNN folder, so runs at different settings sit side by
    // side instead of overwriting each other. The settings themselves go into run_spec.json rather
    // than into the folder name. Note that the CLI --overrides are applied after the config is
    // parsed, which is why this is computed here and not in nextflow.config.
    def used_indices = files("${params.output_root}/results_*", type: 'dir')
            .collect { it.name }
            .findAll { it ==~ /results_\d{3}/ }
            .collect { it.substring(8) as int }
    def run_name = params.run_name ?: String.format('results_%03d', (used_indices + [0]).max() + 1)
    def run_dir = "${params.output_root}/${run_name}"

    def run_spec = [
        run_name     : run_name,
        started_at   : new Date().format("yyyy-MM-dd HH:mm:ss"),
        experiments  : experiment_names,
        model_types  : model_type_names,
        data_dir     : params.data_dir.toString(),
        // Values arriving from the command line are strings, so cast them for the json.
        num_genes    : params.num_genes ? params.num_genes as int : null,
        num_introns  : params.num_introns ? params.num_introns as int : null,
        num_lrt_tests: params.num_lrt_tests ? params.num_lrt_tests as int : null,
        seed         : params.seed as int,
        profile      : workflow.profile,
        nextflow_run : workflow.runName,
        // workflow.commitId is only set for pipelines pulled from a git repo, which is not how this
        // one is launched, so ask git directly about the working tree.
        git_revision : ['git', '-C', "${projectDir}", 'rev-parse', '--short', 'HEAD'].execute().text.trim() ?: 'unknown',
        command_line : workflow.commandLine,
    ]
    file(run_dir).mkdirs()
    file("${run_dir}/run_spec.json").text = groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(run_spec))
    log.info "Writing results to ${run_dir}"

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
            [run_dir, experiment, model_type, dataset_name, dataset_dir]
        }

    FitGlobalModelExperiment(run_channel)
    CollectSummaries(run_dir, FitGlobalModelExperiment.out.summary.collect())
}
