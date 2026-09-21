process WriteDesignMetadata {
    publishDir "${params.outdir}/modeling/${params.model_run_id}/design_metadata", mode: 'copy'

    input:
    val design_formula
    val lrt_contrasts

    output:
    path "design.json"
    path "lrt_contrasts.json", emit: lrt_contrasts_json

    script:
    def design_obj = [
        design_formula : design_formula,
    ]

    def design_json_str       = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(design_obj))
    def lrt_contrasts_json_str = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(lrt_contrasts ?: []))

    """
    printf '%s\n' '${design_json_str}'        > design.json
    printf '%s\n' '${lrt_contrasts_json_str}' > lrt_contrasts.json
    """
}



process CreateDesignMatrices {
    publishDir "${params.outdir}/modeling/${params.model_run_id}/design_matrices", mode: 'copy'

    input:
    path samplesheet
    val design_formula
    path lrt_contrasts_json

    output:
    path("design_matrix.tsv"), emit: design_matrix
    path("lrt_metadata.tsv"), emit: lrt_metadata
    path("reduced_design_matrices"), emit: reduced_design_matrices

    script:
    """
    create_design_matrices.R \
    --samplesheet $samplesheet \
    --formula "$design_formula" \
    --lrt_contrasts_json $lrt_contrasts_json
    """
}

process GetModeledGenes {
    publishDir "${params.outdir}/modeling/${params.model_run_id}/modeled_genes", mode: 'copy'

    input:
    tuple(path(gene_names), path(modelable_genes), path(modelable_introns))

    output:
    path("gene_chunks/modeled_genes_chunk_*.tsv"), emit: gene_names_chunks
    path("intron_chunks/modeled_introns_chunk_*.tsv"), emit: intron_chunks
    path("modeled_genes.tsv"), emit: modeled_genes
    path("modeled_introns.tsv"), emit: modeled_introns

    script:
    """
    get_modeled_genes.R \
    --input_gene_names $gene_names \
    --modelable_genes $modelable_genes \
    --modelable_introns $modelable_introns \
    --output_folder .
    """
}


process FitModel {
    input:
    tuple (
        path(modeled_genes_chunk),
        val(chunk_name),
        path(modeled_introns),
        path(design_matrix),
        path(lrt_metadata),
        path(reduced_matrices_folder),
        path(exon_counts_matrix),
        path(intron_counts_matrix),
        path(library_size_factors),
        path(isoform_length_factors),
        path(coverage_parquet_files),
        val(intron_specific_lfc)
    )

    output:
    path("model_parameters*.tsv"), emit: model_parameters_chunk
    path("test_results*.tsv"), emit: test_results_chunk
    tuple path("cache_for_regularization*.pt"), val(chunk_name), emit: cache_for_regularization

//     publishDir "${params.outdir}/modeling/${params.model_run_id}/processing_chunks/model_results_chunks/${chunk_name}", mode: 'copy'

    script:
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    fit_model.py \
    --modeled_genes $modeled_genes_chunk \
    --modeled_introns $modeled_introns \
    --design_matrix $design_matrix \
    --exon_counts $exon_counts_matrix \
    --intron_counts $intron_counts_matrix \
    --library_size_factors $library_size_factors \
    --isoform_length_factors $isoform_length_factors \
    --coverage_data_folder . \
    --lrt_metadata ${lrt_metadata} \
    --reduced_matrices_folder ${reduced_matrices_folder} \
    --intron_specific_lfc ${intron_specific_lfc} \
    --output_folder . \
    --output_name_suffix _${chunk_name}
    """

}

process FitGeneIntronCoverageModel {
    input:
    tuple(
        path(modeled_genes_chunk),
        val(chunk_name),
        path(modeled_introns),
        path(design_matrix),
        path(lrt_metadata),
        path(reduced_matrices_folder),
        path(exon_counts_matrix),
        path(intron_counts_matrix),
        path(library_size_factors),
        path(isoform_length_factors),
        path(coverage_parquet_files)
    )

    output:
    path("model_parameters*.tsv"), emit: model_parameters_chunk
    path("test_results*.tsv"), emit: test_results_chunk
    tuple path("cache_for_regularization*.pt"), val(chunk_name), emit: cache_for_regularization

    script:
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    fit_gene_specific_intron_coverage_model.py \
    --modeled_genes $modeled_genes_chunk \
    --modeled_introns $modeled_introns \
    --design_matrix $design_matrix \
    --exon_counts $exon_counts_matrix \
    --intron_counts $intron_counts_matrix \
    --library_size_factors $library_size_factors \
    --isoform_length_factors $isoform_length_factors \
    --coverage_data_folder . \
    --lrt_metadata ${lrt_metadata} \
    --reduced_matrices_folder ${reduced_matrices_folder} \
    --output_folder . \
    --output_name_suffix _${chunk_name}
    """
}

process FitIntronCoverageModel {
    input:
    tuple(
        path(modeled_introns),
        val(chunk_name),
        path(design_matrix),
        path(lrt_metadata),
        path(reduced_matrices_folder),
        path(library_size_factors),
        path(coverage_parquet_files)
    )

    output:
    path("model_parameters*.tsv"), emit: model_parameters_chunk
    path("test_results*.tsv"), emit: test_results_chunk
    tuple path("cache_for_regularization*.pt"), val(chunk_name), emit: cache_for_regularization

    script:
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    fit_intron_specific_intron_coverage_model.py \
    --modeled_introns $modeled_introns \
    --design_matrix $design_matrix \
    --library_size_factors $library_size_factors \
    --coverage_data_folder . \
    --lrt_metadata ${lrt_metadata} \
    --reduced_matrices_folder ${reduced_matrices_folder} \
    --output_folder . \
    --output_name_suffix _${chunk_name}
    """
}

process FitGlobalIntronCoverageModel {
    publishDir "${params.outdir}/modeling/${params.model_run_id}/global_intron_coverage_model", mode: 'copy'

    input:
    tuple(
        path(modeled_introns),
        path(design_matrix),
        path(lrt_metadata),
        path(reduced_matrices_folder),
        path(library_size_factors),
        path(coverage_parquet_files)
    )

    output:
    path("model_parameters.tsv"), emit: model_parameters
    path("test_results_raw.tsv"), emit: test_results

    script:
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    fit_global_intron_coverage_model.py \
    --modeled_introns $modeled_introns \
    --design_matrix $design_matrix \
    --library_size_factors $library_size_factors \
    --lrt_metadata ${lrt_metadata} \
    --reduced_matrices_folder ${reduced_matrices_folder} \
    --coverage_data_folder .
    mv test_results.tsv test_results_raw.tsv
    """
}

process FitGlobalRnaKineticsModel {
    publishDir "${params.outdir}/modeling/${params.model_run_id}/global_rna_kinetics_model", mode: 'copy'

    input:
    tuple(
        path(modeled_genes),
        path(modeled_introns),
        path(design_matrix),
        path(lrt_metadata),
        path(reduced_matrices_folder),
        path(exon_counts_matrix),
        path(intron_counts_matrix),
        path(library_size_factors),
        path(isoform_length_factors),
        path(coverage_parquet_files)
    )

    output:
    path("model_parameters.tsv"), emit: model_parameters
    path("test_results_raw.tsv"), emit: test_results

    script:
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    fit_global_rna_kinetics_model.py \
    --modeled_genes $modeled_genes \
    --modeled_introns $modeled_introns \
    --design_matrix $design_matrix \
    --exon_counts $exon_counts_matrix \
    --intron_counts $intron_counts_matrix \
    --library_size_factors $library_size_factors \
    --isoform_length_factors $isoform_length_factors \
    --coverage_data_folder . \
    --lrt_metadata ${lrt_metadata} \
    --reduced_matrices_folder ${reduced_matrices_folder}
    mv test_results.tsv test_results_raw.tsv
    """
}

process MergeModelResultChunks {
    publishDir { "${params.outdir}/modeling/${params.model_run_id}/${model_type_subfolder}" },
     mode: 'copy',
     saveAs: { filename ->
            if( filename == 'test_results_before_regularization.tsv' ) null // Not publishing intermediate test results
            else filename
        }

    input:
    path model_parameters_chunks
    path test_results_chunks
    val model_type_subfolder

    output:
    path("test_results_before_regularization.tsv"), emit: test_results_before_regularization
    path("model_parameters.tsv"), emit: model_parameters

    script:
    """
    merge_result_chunks.R \
    --input_folder . \
    --output_folder .
    """

}


process AdaptiveShrinkage {
    publishDir { "${params.outdir}/modeling/${params.model_run_id}/${model_type_subfolder}" }, mode: 'copy'

    input:
    path model_parameters
    val model_type_subfolder

    output:
    path("regularization_coefficients.tsv"), emit: regularization_coefficients

    script:
    """
    adaptive_shrinkage.R \
    --model_parameters $model_parameters
    """
}

process FitRegularizedModel {
    input:
    tuple(
        path(cache_for_regularization),
        val(chunk_name),
        path(regularization_coefficients)
    )

    output:
    path("regularized_model_parameters*.tsv"), emit: regularized_model_parameters_chunk

//     publishDir "${params.outdir}/modeling/${params.model_run_id}/processing_chunks/regularized_model_results_chunks/${chunk_name}", mode: 'copy'

    script:
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    fit_regularized_model.py \
    --cache_for_regularization $cache_for_regularization \
    --regularization_coefficients $regularization_coefficients \
    --output_name_suffix _$chunk_name
    """
}

process FitRegularizedIntronCoverageModel {
    input:
    tuple(
        path(cache_for_regularization),
        val(chunk_name),
        path(regularization_coefficients)
    )

    output:
    path("regularized_model_parameters*.tsv"), emit: regularized_model_parameters_chunk

    script:
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    fit_regularized_intron_coverage_model.py \
    --cache_for_regularization $cache_for_regularization \
    --regularization_coefficients $regularization_coefficients \
    --output_name_suffix _$chunk_name
    """
}

process MergeRegularization {
    publishDir { "${params.outdir}/modeling/${params.model_run_id}/${model_type_subfolder}" }, mode: 'copy'

    input:
    path test_results_before_regularization
    path regularized_model_parameters_chunks
    val model_type_subfolder

    output:
    path("raw_test_results.tsv"), emit: raw_test_results
    path("regularized_model_parameters.tsv")

    script:
    """
    merge_regularization.R \
    --regularization_chunks_folder . \
    --test_results $test_results_before_regularization \
    --output_folder .
    """
}

process PostprocessResultsAndPlot {
    publishDir { "${params.outdir}/modeling/${params.model_run_id}/${model_type_subfolder}" }, mode: 'copy'

    input:
    path raw_test_results
    val model_type
    val model_type_subfolder

    output:
    path("test_results/test_results.tsv"), emit: test_results
    path("test_results/volcano_plots/**")

    script:
    """
    postprocess_results.R \
    --test_results $raw_test_results \
    --model_type $model_type \
    --output_folder test_results
    """
}

process PostprocessGlobalResultsAndPlot {
    publishDir { "${params.outdir}/modeling/${params.model_run_id}/${model_type_subfolder}" }, mode: 'copy'

    input:
    path test_results
    val model_type
    val model_type_subfolder

    output:
    path("test_results/test_results.tsv"), emit: test_results
    path("test_results/plots/**")

    script:
    """
    postprocess_global_results.R \
    --test_results $test_results \
    --model_type $model_type \
    --output_folder test_results
    """
}

workflow rna_kinetics_model_subworkflow {
    take:
        gene_names_chunks
        modeled_introns
        design_matrix
        lrt_metadata
        reduced_design_matrices
        exon_counts
        intron_counts
        library_size_factors
        isoform_length_factors
        coverage_files

    main:
        def model_subfolder = 'gene_specific_rna_kinetics_model'

        def model_input = gene_names_chunks
            .flatten()
            .map{ file -> tuple(file, file.baseName)}
            .combine(modeled_introns)
            .combine(design_matrix)
            .combine(lrt_metadata)
            .combine(reduced_design_matrices)
            .combine(exon_counts)
            .combine(intron_counts)
            .combine(library_size_factors)
            .combine(isoform_length_factors)
            // Wrapping coverage_files in an extra list prevents unwanted flattening behavior in .combine()
            .combine(coverage_files.map { file_list -> tuple([file_list]) })
            .map { it + [false] }

        def fit_model_output = FitModel(model_input)

        def model_result_merged = MergeModelResultChunks(
            fit_model_output.model_parameters_chunk.collect(),
            fit_model_output.test_results_chunk.collect(),
            model_subfolder
        )

        def adaptive_shrinkage_out = AdaptiveShrinkage(model_result_merged.model_parameters, model_subfolder)

        def fit_regularized_model_output = FitRegularizedModel(
            fit_model_output.cache_for_regularization
                .combine(adaptive_shrinkage_out.regularization_coefficients)
        )

        def merge_regularization_output = MergeRegularization(
            model_result_merged.test_results_before_regularization,
            fit_regularized_model_output.regularized_model_parameters_chunk.collect(),
            model_subfolder
        )

        PostprocessResultsAndPlot(merge_regularization_output.raw_test_results, 'rna_kinetics', model_subfolder)
}


workflow intron_specific_rna_kinetics_model_subworkflow {
    take:
        gene_names_chunks
        modeled_introns
        design_matrix
        lrt_metadata
        reduced_design_matrices
        exon_counts
        intron_counts
        library_size_factors
        isoform_length_factors
        coverage_files

    main:
        def model_subfolder = 'intron_specific_rna_kinetics_model'

        def model_input = gene_names_chunks
            .flatten()
            .map{ file -> tuple(file, file.baseName)}
            .combine(modeled_introns)
            .combine(design_matrix)
            .combine(lrt_metadata)
            .combine(reduced_design_matrices)
            .combine(exon_counts)
            .combine(intron_counts)
            .combine(library_size_factors)
            .combine(isoform_length_factors)
            // Wrapping coverage_files in an extra list prevents unwanted flattening behavior in .combine()
            .combine(coverage_files.map { file_list -> tuple([file_list]) })
            .map { it + [true] }

        def fit_model_output = FitModel(model_input)

        def model_result_merged = MergeModelResultChunks(
            fit_model_output.model_parameters_chunk.collect(),
            fit_model_output.test_results_chunk.collect(),
            model_subfolder
        )

        def adaptive_shrinkage_out = AdaptiveShrinkage(model_result_merged.model_parameters, model_subfolder)

        def fit_regularized_model_output = FitRegularizedModel(
            fit_model_output.cache_for_regularization
                .combine(adaptive_shrinkage_out.regularization_coefficients)
        )

        def merge_regularization_output = MergeRegularization(
            model_result_merged.test_results_before_regularization,
            fit_regularized_model_output.regularized_model_parameters_chunk.collect(),
            model_subfolder
        )

        PostprocessResultsAndPlot(merge_regularization_output.raw_test_results, 'rna_kinetics', model_subfolder)
}


workflow gene_specific_intron_coverage_model_subworkflow {
    take:
        gene_names_chunks
        modeled_introns
        design_matrix
        lrt_metadata
        reduced_design_matrices
        exon_counts
        intron_counts
        library_size_factors
        isoform_length_factors
        coverage_files

    main:
        def model_subfolder = 'gene_specific_intron_coverage_model'

        def model_input = gene_names_chunks
            .flatten()
            .map{ file -> tuple(file, file.baseName)}
            .combine(modeled_introns)
            .combine(design_matrix)
            .combine(lrt_metadata)
            .combine(reduced_design_matrices)
            .combine(exon_counts)
            .combine(intron_counts)
            .combine(library_size_factors)
            .combine(isoform_length_factors)
            // Wrapping coverage_files in an extra list prevents unwanted flattening behavior in .combine()
            .combine(coverage_files.map { file_list -> tuple([file_list]) })

        def fit_model_output = FitGeneIntronCoverageModel(model_input)

        def model_result_merged = MergeModelResultChunks(
            fit_model_output.model_parameters_chunk.collect(),
            fit_model_output.test_results_chunk.collect(),
            model_subfolder
        )

        def adaptive_shrinkage_out = AdaptiveShrinkage(model_result_merged.model_parameters, model_subfolder)

        def fit_regularized_model_output = FitRegularizedIntronCoverageModel(
            fit_model_output.cache_for_regularization
                .combine(adaptive_shrinkage_out.regularization_coefficients)
        )

        def merge_regularization_output = MergeRegularization(
            model_result_merged.test_results_before_regularization,
            fit_regularized_model_output.regularized_model_parameters_chunk.collect(),
            model_subfolder
        )

        PostprocessResultsAndPlot(merge_regularization_output.raw_test_results, 'intron_coverage', model_subfolder)
}


workflow intron_specific_intron_coverage_model_subworkflow {
    take:
        intron_chunks
        design_matrix
        lrt_metadata
        reduced_design_matrices
        library_size_factors
        coverage_files

    main:
        def model_subfolder = 'intron_specific_intron_coverage_model'

        def model_input = intron_chunks
            .flatten()
            .map{ file -> tuple(file, file.baseName)}
            .combine(design_matrix)
            .combine(lrt_metadata)
            .combine(reduced_design_matrices)
            .combine(library_size_factors)
            // Wrapping coverage_files in an extra list prevents unwanted flattening behavior in .combine()
            .combine(coverage_files.map { file_list -> tuple([file_list]) })

        def fit_model_output = FitIntronCoverageModel(model_input)

        def model_result_merged = MergeModelResultChunks(
            fit_model_output.model_parameters_chunk.collect(),
            fit_model_output.test_results_chunk.collect(),
            model_subfolder
        )

        def adaptive_shrinkage_out = AdaptiveShrinkage(model_result_merged.model_parameters, model_subfolder)

        def fit_regularized_model_output = FitRegularizedIntronCoverageModel(
            fit_model_output.cache_for_regularization
                .combine(adaptive_shrinkage_out.regularization_coefficients)
        )

        def merge_regularization_output = MergeRegularization(
            model_result_merged.test_results_before_regularization,
            fit_regularized_model_output.regularized_model_parameters_chunk.collect(),
            model_subfolder
        )

        PostprocessResultsAndPlot(merge_regularization_output.raw_test_results, 'intron_coverage', model_subfolder)
}


workflow global_intron_coverage_model_subworkflow {
    take:
        modeled_introns
        design_matrix
        lrt_metadata
        reduced_design_matrices
        library_size_factors
        coverage_files

    main:
        def model_subfolder = 'global_intron_coverage_model'

        def fit_output = FitGlobalIntronCoverageModel(
            modeled_introns
                .combine(design_matrix)
                .combine(lrt_metadata)
                .combine(reduced_design_matrices)
                .combine(library_size_factors)
                // Wrapping coverage_files in an extra list prevents unwanted flattening behavior in .combine()
                .combine(coverage_files.map { file_list -> tuple([file_list]) })
        )

        PostprocessGlobalResultsAndPlot(fit_output.test_results, 'global_intron_coverage', model_subfolder)
}


workflow global_rna_kinetics_model_subworkflow {
    take:
        modeled_genes
        modeled_introns
        design_matrix
        lrt_metadata
        reduced_design_matrices
        exon_counts
        intron_counts
        library_size_factors
        isoform_length_factors
        coverage_files

    main:
        def model_subfolder = 'global_rna_kinetics_model'

        def fit_output = FitGlobalRnaKineticsModel(
            modeled_genes
                .combine(modeled_introns)
                .combine(design_matrix)
                .combine(lrt_metadata)
                .combine(reduced_design_matrices)
                .combine(exon_counts)
                .combine(intron_counts)
                .combine(library_size_factors)
                .combine(isoform_length_factors)
                // Wrapping coverage_files in an extra list prevents unwanted flattening behavior in .combine()
                .combine(coverage_files.map { file_list -> tuple([file_list]) })
        )

        PostprocessGlobalResultsAndPlot(fit_output.test_results, 'global_rna_kinetics', model_subfolder)
}


process StratifiedIntronMetacoveragePlots {
    publishDir "${params.outdir}/modeling/${params.model_run_id}/stratified_intron_metacoverage_plots", mode: 'copy'

    input:
    tuple path(coverage_parquet_files), path(introns_bed_file), path(gene_names_csv), path(samplesheet)
    val design_formula

    output:
    path("*/*")

    script:
    """
    plot_intron_metacoverage.py \
        --coverage_data_folder . \
        --introns_bed_file $introns_bed_file \
        --gene_names $gene_names_csv \
        --samplesheet $samplesheet \
        --design_formula "$design_formula"
    """
}


workflow modeling_workflow {
    take:
        samplesheet_input
        gene_names_file_input
        introns_bed_file_input
        exon_counts_input
        intron_counts_input
        library_size_factors_input
        isoform_length_factors_input
        coverage_files_input
        design_formula
        lrt_contrasts
        modelable_genes_input
        modelable_introns_input
        fit_rna_kinetics_model
        fit_intron_specific_rna_kinetics_model
        fit_global_rna_kinetics_model
        fit_global_intron_coverage_model
        fit_gene_specific_intron_coverage_model
        fit_intron_specific_intron_coverage_model

    main:
        def samplesheet = Channel.value(file(samplesheet_input))

        def stratified_plot_input = coverage_files_input
            .map { files -> tuple([files]) }
            .combine(introns_bed_file_input)
            .combine(gene_names_file_input)
            .combine(samplesheet)

        StratifiedIntronMetacoveragePlots(stratified_plot_input, design_formula)

        def design_metadata = WriteDesignMetadata(design_formula, lrt_contrasts)

        def design_matrices_data = CreateDesignMatrices(
            samplesheet,
            design_formula,
            design_metadata.lrt_contrasts_json
        )

        def modeled_genes = GetModeledGenes(
            gene_names_file_input
                .combine(modelable_genes_input)
                .combine(modelable_introns_input)
        )

        if (fit_rna_kinetics_model) {
            rna_kinetics_model_subworkflow(
                modeled_genes.gene_names_chunks,
                modeled_genes.modeled_introns,
                design_matrices_data.design_matrix,
                design_matrices_data.lrt_metadata,
                design_matrices_data.reduced_design_matrices,
                exon_counts_input,
                intron_counts_input,
                library_size_factors_input,
                isoform_length_factors_input,
                coverage_files_input
            )
        }

        if (fit_intron_specific_rna_kinetics_model) {
            intron_specific_rna_kinetics_model_subworkflow(
                modeled_genes.gene_names_chunks,
                modeled_genes.modeled_introns,
                design_matrices_data.design_matrix,
                design_matrices_data.lrt_metadata,
                design_matrices_data.reduced_design_matrices,
                exon_counts_input,
                intron_counts_input,
                library_size_factors_input,
                isoform_length_factors_input,
                coverage_files_input
            )
        }

        if (fit_global_rna_kinetics_model) {
            global_rna_kinetics_model_subworkflow(
                modeled_genes.modeled_genes,
                modeled_genes.modeled_introns,
                design_matrices_data.design_matrix,
                design_matrices_data.lrt_metadata,
                design_matrices_data.reduced_design_matrices,
                exon_counts_input,
                intron_counts_input,
                library_size_factors_input,
                isoform_length_factors_input,
                coverage_files_input
            )
        }

        if (fit_gene_specific_intron_coverage_model) {
            gene_specific_intron_coverage_model_subworkflow(
                modeled_genes.gene_names_chunks,
                modeled_genes.modeled_introns,
                design_matrices_data.design_matrix,
                design_matrices_data.lrt_metadata,
                design_matrices_data.reduced_design_matrices,
                exon_counts_input,
                intron_counts_input,
                library_size_factors_input,
                isoform_length_factors_input,
                coverage_files_input
            )
        }

        if (fit_intron_specific_intron_coverage_model) {
            intron_specific_intron_coverage_model_subworkflow(
                modeled_genes.intron_chunks,
                design_matrices_data.design_matrix,
                design_matrices_data.lrt_metadata,
                design_matrices_data.reduced_design_matrices,
                library_size_factors_input,
                coverage_files_input
            )
        }

        if (fit_global_intron_coverage_model) {
            global_intron_coverage_model_subworkflow(
                modeled_genes.modeled_introns,
                design_matrices_data.design_matrix,
                design_matrices_data.lrt_metadata,
                design_matrices_data.reduced_design_matrices,
                library_size_factors_input,
                coverage_files_input
            )
        }

}
