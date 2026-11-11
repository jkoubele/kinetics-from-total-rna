process WriteDesignMetadata {
    publishDir "${params.outdir}/modeling/${params.model_run_id}/design_metadata", mode: 'copy'

    input:
    val design_formula
    val contrasts

    output:
    path "design.json"
    path "contrasts.json", emit: contrasts_json

    script:
    def design_obj = [
        design_formula : design_formula,
    ]

    def design_json_str       = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(design_obj))
    def contrasts_json_str = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(contrasts ?: []))

    """
    printf '%s\n' '${design_json_str}'        > design.json
    printf '%s\n' '${contrasts_json_str}' > contrasts.json
    """
}



process CreateDesignMatrices {
    publishDir "${params.outdir}/modeling/${params.model_run_id}/design_matrices", mode: 'copy'

    input:
    path samplesheet
    val design_formula
    path contrasts_json

    output:
    path("design_matrix.tsv"), emit: design_matrix
    path("lrt_metadata.tsv"), emit: lrt_metadata
    path("reduced_design_matrices"), emit: reduced_design_matrices

    script:
    """
    create_design_matrices.R \
    --samplesheet $samplesheet \
    --formula "$design_formula" \
    --contrasts_json $contrasts_json
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


process FitRNAKineticsModel {
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
        val(intron_specific_lfc),
        path(plug_in_coverage_fits)
    )

    output:
    path("model_parameters*.tsv"), emit: model_parameters_chunk
    path("ratio_lfcs*.tsv"), emit: ratio_lfcs_chunk
    path("test_results*.tsv"), emit: test_results_chunk
    tuple path("cache_for_regularization*.pt"), val(chunk_name), emit: cache_for_regularization

//     publishDir "${params.outdir}/modeling/${params.model_run_id}/processing_chunks/model_results_chunks/${chunk_name}", mode: 'copy'

    script:
    // Empty where the joint likelihood is being fitted instead, which stages no file and
    // leaves the argument off, rather than passing a path the script would have to ignore.
    def plug_in_argument = plug_in_coverage_fits ? "--plug_in_coverage_fits ${plug_in_coverage_fits}" : ''
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    fit_rna_kinetics_model.py \
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
    ${plug_in_argument} \
    --output_folder . \
    --output_name_suffix _${chunk_name}
    """

}

process FitGeneSpecificIntronCoverageModel {
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
    path("ratio_lfcs*.tsv"), emit: ratio_lfcs_chunk
    path("test_results*.tsv"), emit: test_results_chunk
    tuple path("cache_for_regularization*.pt"), val(chunk_name), emit: cache_for_regularization
    path("plug_in_coverage_fits*.pt"), emit: plug_in_coverage_fits_chunk

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

process FitIntronSpecificIntronCoverageModel {
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
    path("ratio_lfcs*.tsv"), emit: ratio_lfcs_chunk
    path("test_results*.tsv"), emit: test_results_chunk
    tuple path("cache_for_regularization*.pt"), val(chunk_name), emit: cache_for_regularization
    path("plug_in_coverage_fits*.pt"), emit: plug_in_coverage_fits_chunk

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
    path("plug_in_coverage_fits.pt"), emit: plug_in_coverage_fits

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

process FitGlobalRNAKineticsModel {
    publishDir { "${params.outdir}/modeling/${params.model_run_id}/${model_type_subfolder}" },
     mode: 'copy'

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
        path(coverage_parquet_files),
        path(plug_in_coverage_fits),
        val(model_type_subfolder)
    )

    output:
    path("model_parameters.tsv"), emit: model_parameters
    path("test_results_raw.tsv"), emit: test_results

    script:
    // Empty where the joint likelihood is being fitted instead, which stages no file and
    // leaves the argument off, rather than passing a path the script would have to ignore.
    def plug_in_argument = plug_in_coverage_fits ? "--plug_in_coverage_fits ${plug_in_coverage_fits}" : ''
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
    --reduced_matrices_folder ${reduced_matrices_folder} \
    ${plug_in_argument}
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
    path ratio_lfcs_chunks
    path test_results_chunks
    val model_type_subfolder

    output:
    path("test_results_before_regularization.tsv"), emit: test_results_before_regularization
    path("model_parameters.tsv"), emit: model_parameters
    path("ratio_lfcs.tsv"), emit: ratio_lfcs

    script:
    """
    merge_result_chunks.R \
    --input_folder . \
    --output_folder .
    """

}


process MergePlugInCoverageFits {
    input:
    path plug_in_coverage_fits_chunks

    output:
    path("plug_in_coverage_fits.pt"), emit: plug_in_coverage_fits

    script:
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    merge_plug_in_coverage_fits.py \
    --input_folder . \
    --output_file plug_in_coverage_fits.pt
    """
}


process EstimatePriorVariance {
    publishDir { "${params.outdir}/modeling/${params.model_run_id}/${model_type_subfolder}" }, mode: 'copy'

    input:
    path ratio_lfcs
    val model_type_subfolder

    output:
    path("regularization_coefficients.tsv"), emit: regularization_coefficients

    script:
    """
    estimate_prior_variance.R \
    --ratio_lfcs $ratio_lfcs
    """
}

process FitRegularizedRNAKineticsModel {
    input:
    tuple(
        path(cache_for_regularization),
        val(chunk_name),
        path(regularization_coefficients),
        path(plug_in_coverage_fits)
    )

    output:
    path("regularized_model_parameters*.tsv"), emit: regularized_model_parameters_chunk
    path("regularized_ratio_lfcs*.tsv"), emit: regularized_ratio_lfcs_chunk

//     publishDir "${params.outdir}/modeling/${params.model_run_id}/processing_chunks/regularized_model_results_chunks/${chunk_name}", mode: 'copy'

    script:
    // Empty where the joint likelihood is being fitted instead, which stages no file and
    // leaves the argument off, rather than passing a path the script would have to ignore.
    def plug_in_argument = plug_in_coverage_fits ? "--plug_in_coverage_fits ${plug_in_coverage_fits}" : ''
    """
    export PYTHONPATH='${baseDir}'\${PYTHONPATH:+:\$PYTHONPATH}
    fit_regularized_rna_kinetics_model.py \
    --cache_for_regularization $cache_for_regularization \
    --regularization_coefficients $regularization_coefficients \
    ${plug_in_argument} \
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
    path("regularized_ratio_lfcs*.tsv"), emit: regularized_ratio_lfcs_chunk

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
    path regularized_ratio_lfcs_chunks
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

workflow gene_specific_rna_kinetics_model_subworkflow {
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
        plug_in_coverage_fits
        use_plug_in_coverage

    main:
        // The joint likelihood writes to its own folder, so which estimator produced
        // a set of results is visible from the results themselves.
        def model_subfolder = use_plug_in_coverage ? 'gene_specific_rna_kinetics_model'
            : 'gene_specific_rna_kinetics_model_joint_likelihood'

        def model_input_without_coverage_fits = gene_names_chunks
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
            .map { chunk_input -> chunk_input + [false] }

        // The plug-in route attaches the merged coverage fits. The joint one attaches an empty
        // list, which stages no file and so leaves the argument off the command line.
        def model_input = use_plug_in_coverage
            ? model_input_without_coverage_fits.combine(plug_in_coverage_fits)
            : model_input_without_coverage_fits.map { chunk_input -> chunk_input + [[]] }

        def fit_output = FitRNAKineticsModel(model_input)

        def model_result_merged = MergeModelResultChunks(
            fit_output.model_parameters_chunk.collect(),
            fit_output.ratio_lfcs_chunk.collect(),
            fit_output.test_results_chunk.collect(),
            model_subfolder
        )

        def prior_variance_out = EstimatePriorVariance(model_result_merged.ratio_lfcs, model_subfolder)

        def regularized_input = fit_output.cache_for_regularization
            .combine(prior_variance_out.regularization_coefficients)

        def fit_regularized_output = FitRegularizedRNAKineticsModel(
            use_plug_in_coverage
                ? regularized_input.combine(plug_in_coverage_fits)
                : regularized_input.map { chunk_input -> chunk_input + [[]] }
        )

        def merge_regularization_output = MergeRegularization(
            model_result_merged.test_results_before_regularization,
            fit_regularized_output.regularized_model_parameters_chunk.collect(),
            fit_regularized_output.regularized_ratio_lfcs_chunk.collect(),
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
        plug_in_coverage_fits
        use_plug_in_coverage

    main:
        // The joint likelihood writes to its own folder, so which estimator produced
        // a set of results is visible from the results themselves.
        def model_subfolder = use_plug_in_coverage ? 'intron_specific_rna_kinetics_model'
            : 'intron_specific_rna_kinetics_model_joint_likelihood'

        def model_input_without_coverage_fits = gene_names_chunks
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
            .map { chunk_input -> chunk_input + [true] }

        // The plug-in route attaches the merged coverage fits. The joint one attaches an empty
        // list, which stages no file and so leaves the argument off the command line.
        def model_input = use_plug_in_coverage
            ? model_input_without_coverage_fits.combine(plug_in_coverage_fits)
            : model_input_without_coverage_fits.map { chunk_input -> chunk_input + [[]] }

        def fit_output = FitRNAKineticsModel(model_input)

        def model_result_merged = MergeModelResultChunks(
            fit_output.model_parameters_chunk.collect(),
            fit_output.ratio_lfcs_chunk.collect(),
            fit_output.test_results_chunk.collect(),
            model_subfolder
        )

        def prior_variance_out = EstimatePriorVariance(model_result_merged.ratio_lfcs, model_subfolder)

        def regularized_input = fit_output.cache_for_regularization
            .combine(prior_variance_out.regularization_coefficients)

        def fit_regularized_output = FitRegularizedRNAKineticsModel(
            use_plug_in_coverage
                ? regularized_input.combine(plug_in_coverage_fits)
                : regularized_input.map { chunk_input -> chunk_input + [[]] }
        )

        def merge_regularization_output = MergeRegularization(
            model_result_merged.test_results_before_regularization,
            fit_regularized_output.regularized_model_parameters_chunk.collect(),
            fit_regularized_output.regularized_ratio_lfcs_chunk.collect(),
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

        def fit_output = FitGeneSpecificIntronCoverageModel(model_input)

        def model_result_merged = MergeModelResultChunks(
            fit_output.model_parameters_chunk.collect(),
            fit_output.ratio_lfcs_chunk.collect(),
            fit_output.test_results_chunk.collect(),
            model_subfolder
        )

        def prior_variance_out = EstimatePriorVariance(model_result_merged.ratio_lfcs, model_subfolder)

        def fit_regularized_output = FitRegularizedIntronCoverageModel(
            fit_output.cache_for_regularization
                .combine(prior_variance_out.regularization_coefficients)
        )

        def merge_regularization_output = MergeRegularization(
            model_result_merged.test_results_before_regularization,
            fit_regularized_output.regularized_model_parameters_chunk.collect(),
            fit_regularized_output.regularized_ratio_lfcs_chunk.collect(),
            model_subfolder
        )

        def plug_in_output = MergePlugInCoverageFits(
            fit_output.plug_in_coverage_fits_chunk.collect())

        PostprocessResultsAndPlot(merge_regularization_output.raw_test_results, 'intron_coverage', model_subfolder)

    emit:
        plug_in_coverage_fits = plug_in_output.plug_in_coverage_fits
        raw_test_results = merge_regularization_output.raw_test_results
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

        def fit_output = FitIntronSpecificIntronCoverageModel(model_input)

        def model_result_merged = MergeModelResultChunks(
            fit_output.model_parameters_chunk.collect(),
            fit_output.ratio_lfcs_chunk.collect(),
            fit_output.test_results_chunk.collect(),
            model_subfolder
        )

        def prior_variance_out = EstimatePriorVariance(model_result_merged.ratio_lfcs, model_subfolder)

        def fit_regularized_output = FitRegularizedIntronCoverageModel(
            fit_output.cache_for_regularization
                .combine(prior_variance_out.regularization_coefficients)
        )

        def merge_regularization_output = MergeRegularization(
            model_result_merged.test_results_before_regularization,
            fit_regularized_output.regularized_model_parameters_chunk.collect(),
            fit_regularized_output.regularized_ratio_lfcs_chunk.collect(),
            model_subfolder
        )

        def plug_in_output = MergePlugInCoverageFits(
            fit_output.plug_in_coverage_fits_chunk.collect())

        PostprocessResultsAndPlot(merge_regularization_output.raw_test_results, 'intron_coverage', model_subfolder)

    emit:
        plug_in_coverage_fits = plug_in_output.plug_in_coverage_fits
        raw_test_results = merge_regularization_output.raw_test_results
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

    emit:
        // One fit over the whole dataset, so there are no chunks to merge.
        plug_in_coverage_fits = fit_output.plug_in_coverage_fits
        raw_test_results = fit_output.test_results
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
        plug_in_coverage_fits
        use_plug_in_coverage

    main:
        // The joint likelihood writes to its own folder, so which estimator produced
        // a set of results is visible from the results themselves.
        def model_subfolder = use_plug_in_coverage ? 'global_rna_kinetics_model'
            : 'global_rna_kinetics_model_joint_likelihood'

        def fit_input_without_coverage_fits = modeled_genes
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

        // The plug-in route attaches the single genome-wide coverage fit. The joint one attaches
        // an empty list, which stages no file and so leaves the argument off the command line.
        def fit_output = FitGlobalRNAKineticsModel(
            (use_plug_in_coverage
                ? fit_input_without_coverage_fits.combine(plug_in_coverage_fits)
                : fit_input_without_coverage_fits.map { fit_input -> fit_input + [[]] })
                .map { fit_input -> fit_input + [model_subfolder] }
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


def warn_coverage_switched_on(String coverage_flag) {
    log.warn "Switching on ${coverage_flag}: the RNA kinetics model it serves plugs in its fit " +
        "of lfc_elong_over_splice. Set use_plug_in_coverage to false to fit the joint likelihood " +
        "instead, which we do not recommend."
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
        contrasts
        modelable_genes_input
        modelable_introns_input
        fit_gene_specific_rna_kinetics_model
        fit_intron_specific_rna_kinetics_model
        fit_global_rna_kinetics_model
        fit_global_intron_coverage_model
        fit_gene_specific_intron_coverage_model
        fit_intron_specific_intron_coverage_model
        use_plug_in_coverage

    main:
        def samplesheet = Channel.value(file(samplesheet_input))

        def stratified_plot_input = coverage_files_input
            .map { files -> tuple([files]) }
            .combine(introns_bed_file_input)
            .combine(gene_names_file_input)
            .combine(samplesheet)

        StratifiedIntronMetacoveragePlots(stratified_plot_input, design_formula)

        def design_metadata = WriteDesignMetadata(design_formula, contrasts)

        def design_matrices_data = CreateDesignMatrices(
            samplesheet,
            design_formula,
            design_metadata.contrasts_json
        )

        def modeled_genes = GetModeledGenes(
            gene_names_file_input
                .combine(modelable_genes_input)
                .combine(modelable_introns_input)
        )

        // Each plug-in kinetics model is a stage two, so the coverage model it reads has to run.
        // Switching that on is the only way to honour the request, a plug-in having nothing to
        // plug in otherwise, so the flag is overridden rather than the run refused.
        def run_gene_specific_coverage = fit_gene_specific_intron_coverage_model
        if (use_plug_in_coverage && fit_gene_specific_rna_kinetics_model
                && !fit_gene_specific_intron_coverage_model) {
            warn_coverage_switched_on('fit_gene_specific_intron_coverage_model')
            run_gene_specific_coverage = true
        }
        def run_intron_specific_coverage = fit_intron_specific_intron_coverage_model
        if (use_plug_in_coverage && fit_intron_specific_rna_kinetics_model
                && !fit_intron_specific_intron_coverage_model) {
            warn_coverage_switched_on('fit_intron_specific_intron_coverage_model')
            run_intron_specific_coverage = true
        }
        def run_global_coverage = fit_global_intron_coverage_model
        if (use_plug_in_coverage && fit_global_rna_kinetics_model
                && !fit_global_intron_coverage_model) {
            warn_coverage_switched_on('fit_global_intron_coverage_model')
            run_global_coverage = true
        }

        // Stage one first, so that the kinetics models can read what it produced. A kinetics
        // model fitting the joint likelihood is handed an empty channel it never consumes.
        def gene_specific_coverage_output = null
        if (run_gene_specific_coverage) {
            gene_specific_coverage_output = gene_specific_intron_coverage_model_subworkflow(
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

        def intron_specific_coverage_output = null
        if (run_intron_specific_coverage) {
            intron_specific_coverage_output = intron_specific_intron_coverage_model_subworkflow(
                modeled_genes.intron_chunks,
                design_matrices_data.design_matrix,
                design_matrices_data.lrt_metadata,
                design_matrices_data.reduced_design_matrices,
                library_size_factors_input,
                coverage_files_input
            )
        }

        def global_coverage_output = null
        if (run_global_coverage) {
            global_coverage_output = global_intron_coverage_model_subworkflow(
                modeled_genes.modeled_introns,
                design_matrices_data.design_matrix,
                design_matrices_data.lrt_metadata,
                design_matrices_data.reduced_design_matrices,
                library_size_factors_input,
                coverage_files_input
            )
        }

        if (fit_gene_specific_rna_kinetics_model) {
            gene_specific_rna_kinetics_model_subworkflow(
                modeled_genes.gene_names_chunks,
                modeled_genes.modeled_introns,
                design_matrices_data.design_matrix,
                design_matrices_data.lrt_metadata,
                design_matrices_data.reduced_design_matrices,
                exon_counts_input,
                intron_counts_input,
                library_size_factors_input,
                isoform_length_factors_input,
                coverage_files_input,
                use_plug_in_coverage ? gene_specific_coverage_output.plug_in_coverage_fits
                    : channel.empty(),
                use_plug_in_coverage
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
                coverage_files_input,
                use_plug_in_coverage ? intron_specific_coverage_output.plug_in_coverage_fits
                    : channel.empty(),
                use_plug_in_coverage
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
                coverage_files_input,
                use_plug_in_coverage ? global_coverage_output.plug_in_coverage_fits
                    : channel.empty(),
                use_plug_in_coverage
            )
        }
}
