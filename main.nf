include { preprocessing_workflow } from './workflows/preprocessing.nf'
include { modeling_workflow } from './workflows/modeling.nf'


workflow {

    def preproc_out

    if (!['preprocess', 'model', 'all'].contains(params.stage)) {
        error("Invalid 'stage' value: ${params.stage}. Must be either 'preprocess', 'model' or 'all'.")
    }

    if (['preprocess', 'all'].contains(params.stage)) {
        // Run pre-processing workflow
        if (!params.samplesheet ||
            !params.fastq_dir ||
            !params.gtf_file ||
            !params.genome_fasta ||
            !params.transcriptome_fasta ||
            params.salmon_index_with_decoy == null ||
            !params.gtf_source ||
            params.ignore_tx_version == null) {
            error("Missing required parameters! Please run with -params-file <your_params.yaml>.")
        }

        if (!['ensembl', 'gencode'].contains(params.gtf_source)) {
            error("Invalid 'gtf_source' value: ${params.gtf_source}. Must be 'ensembl' or 'gencode'.")
        }

        preproc_out = preprocessing_workflow(
            params.samplesheet,
            params.fastq_dir,
            params.gtf_file,
            params.genome_fasta,
            params.transcriptome_fasta,
            params.gtf_source,
            params.salmon_index_with_decoy,
            params.star_index,
            params.salmon_index,
            params.ignore_tx_version,
            params.genomic_features_dir)
    }

    if (['model', 'all'].contains(params.stage)) {
        // Run modeling workflow
        def gene_names_file
        def introns_bed_file
        def exon_counts
        def intron_counts
        def library_size_factors
        def isoform_length_factors
        def coverage_files

        if (params.stage == 'model') {
            def preprocessing_output_subfolder = "preprocessing"
            gene_names_file        = Channel.value(file("${params.outdir}/${preprocessing_output_subfolder}/genomic_features/protein_coding_genes.csv"))
            introns_bed_file       = Channel.value(file("${params.outdir}/${preprocessing_output_subfolder}/genomic_features/introns.bed"))
            exon_counts            = Channel.value(file("${params.outdir}/${preprocessing_output_subfolder}/aggregated_counts/exon_counts.tsv"))
            intron_counts          = Channel.value(file("${params.outdir}/${preprocessing_output_subfolder}/aggregated_counts/intron_counts.tsv"))
            library_size_factors   = Channel.value(file("${params.outdir}/${preprocessing_output_subfolder}/aggregated_counts/library_size_factors.tsv"))
            isoform_length_factors = Channel.value(file("${params.outdir}/${preprocessing_output_subfolder}/aggregated_counts/isoform_length_factors.tsv"))
            coverage_files         = Channel.fromPath("${params.outdir}/${preprocessing_output_subfolder}/rescaled_coverage/*.parquet").collect()
            modelable_genes        = Channel.value(file("${params.outdir}/${preprocessing_output_subfolder}/modelable_genes/modelable_genes.tsv"))
            modelable_introns      = Channel.value(file("${params.outdir}/${preprocessing_output_subfolder}/modelable_genes/modelable_introns.tsv"))
        }

        if (params.stage == 'all') {

            gene_names_file        = preproc_out.gene_names_file
            introns_bed_file       = preproc_out.introns_bed_file
            exon_counts            = preproc_out.exon_counts
            intron_counts          = preproc_out.intron_counts
            library_size_factors   = preproc_out.library_size_factors
            isoform_length_factors = preproc_out.isoform_length_factors
            coverage_files         = preproc_out.coverage_files
            modelable_genes        = preproc_out.modelable_genes
            modelable_introns      = preproc_out.modelable_introns

        }

        if (params.custom_gene_list) {
            log.info("Using custom gene list instead of default protein-coding genes: ${params.custom_gene_list}")
            gene_names_file = Channel.value(file(params.custom_gene_list))
        }


        def contrasts = params.contrasts ?: []

        modeling_workflow(
            params.samplesheet,
            gene_names_file,
            introns_bed_file,
            exon_counts,
            intron_counts,
            library_size_factors,
            isoform_length_factors,
            coverage_files,
            params.design_formula,
            contrasts,
            modelable_genes,
            modelable_introns,
            params.fit_gene_specific_rna_kinetics_model,
            params.fit_intron_specific_rna_kinetics_model,
            params.fit_global_rna_kinetics_model,
            params.fit_global_intron_coverage_model,
            params.fit_gene_specific_intron_coverage_model,
            params.fit_intron_specific_intron_coverage_model,
            params.use_plug_in_coverage
        )
    }
}
