// bp trimmed from each intron edge. Shared between the intronic-read overlap
// threshold (ExtractIntronicReads) and the coverage rescaling window
// (RescaleCoverage) so the two stay coupled.
def edge_margin() { 10 }

process FastQC {
    tag "$sample"
    publishDir "${params.outdir}/preprocessing/FastQC", mode: 'copy'

    input:
    tuple val(sample), path(read1), path(read2), val(is_paired)

    output:
    path("*.zip"), emit: fastqc_reports
    path("*.html")

    script:
    def reads = is_paired ? "$read1 $read2" : "$read1"
    """
    fastqc $reads --outdir .
    """
}

process MultiQC {
    publishDir "${params.outdir}/preprocessing/MultiQC", mode: 'copy'

    input:
    path zip_files

    output:
    path("multiqc_report.html")
    path("multiqc_report_data")

    script:
    """
    multiqc . --filename multiqc_report.html
    """
}

process ExtractGenomicFeatures {
    publishDir "${params.outdir}/preprocessing/genomic_features", mode: 'copy'

    input:
    path gtf
    val precomputed_genomic_features_dir

    output:
        path('introns.bed'), emit: introns_bed_file
        path('constitutive_exons.bed'), emit: constitutive_exons_bed_file
        path('introns.gtf'), emit: introns_gtf_file
        path('constitutive_exons.gtf'), emit: constitutive_exons_gtf_file
        // Published for inspection only (not used downstream); optional so the
        // precomputed-features branch, which doesn't produce them, doesn't fail.
        path('blacklisted_introns.bed'), optional: true
        path('blacklisted_introns.gtf'), optional: true
        path("protein_coding_genes.csv"), emit: protein_coding_gene_names
        path("all_genes.csv"), emit: all_genes

    script:
    if (precomputed_genomic_features_dir) {
        log.info "Using provided genomic features directory: ${precomputed_genomic_features_dir}"
    """
    cp ${precomputed_genomic_features_dir}/introns.bed .
    cp ${precomputed_genomic_features_dir}/constitutive_exons.bed .
    cp ${precomputed_genomic_features_dir}/introns.gtf .
    cp ${precomputed_genomic_features_dir}/constitutive_exons.gtf .
    cp ${precomputed_genomic_features_dir}/protein_coding_genes.csv .
    cp ${precomputed_genomic_features_dir}/all_genes.csv .
    """
    } else {
    """
    extract_genomic_features_from_gtf.R \
        --gtf $gtf \
        --threads ${task.cpus}
    """
    }
}


process BuildStarIndex {
    publishDir "${params.outdir}/preprocessing/STAR_index", mode: 'copy'

    input:
    path fasta
    path gtf

    output:
    path("star_index"), emit: star_index_dir

    script:
    """
    mkdir star_index
    STAR \
      --runThreadN ${task.cpus} \
      --runMode genomeGenerate \
      --genomeDir star_index \
      --genomeFastaFiles $fasta \
      --sjdbGTFfile $gtf \
      --limitGenomeGenerateRAM 190000000000
    """
}

process BuildSalmonIndex {
    publishDir "${params.outdir}/preprocessing/Salmon_index", mode: 'copy'

    input:
    path transcriptome_fasta
    path genome_fasta
    path gtf
    val(gtf_source)
    val(salmon_index_with_decoy)

    output:
    path("salmon_index"), emit: salmon_index_dir
    path("decoy_transcriptome/gentrome.fa", optional: true)
    path("decoy_transcriptome/decoys.txt", optional: true)

    script:
    def gencode_flag = gtf_source == 'gencode' ? '--gencode' : ''
    if (salmon_index_with_decoy) {
    """
    generateDecoyTranscriptome.sh \
        -a $gtf \
        -g $genome_fasta \
        -t $transcriptome_fasta \
        -o decoy_transcriptome \
        -j ${task.cpus}

    mkdir salmon_index
    salmon index \
      -t decoy_transcriptome/gentrome.fa \
      -d decoy_transcriptome/decoys.txt \
      -i salmon_index \
      -p ${task.cpus} \
      ${gencode_flag}
    """
    } else {
        """
        mkdir salmon_index
        salmon index \
            -t $transcriptome_fasta \
            -i salmon_index \
            -p ${task.cpus} \
            ${gencode_flag}
        """
    }
}

process PrepareTx2Gene {
    publishDir "${params.outdir}/preprocessing/tx2gene", mode: 'copy'

    input:
    path gtf

    output:
    path("tx2gene.tsv"), emit: tx2gene_file

    script:
    """
    prepare_tx2gene.R --gtf_file $gtf
    """
}

process CreateGenomeFastaIndex {
    publishDir "${params.outdir}/preprocessing/genome_fai", mode: 'copy'

    input:
    path genome_fasta

    output:
    path("*.fai"), emit: genome_fai_file

    script:
    """
    samtools faidx $genome_fasta
    """
}

process STARAlign {
    publishDir { "${params.outdir}/preprocessing/STAR/${sample}" }, mode: 'copy'
    tag "$sample"

    input:
        tuple val(sample), path(read1), path(read2), path(star_index), val(is_paired)

    output:
        tuple val(sample), path("${sample}.Aligned.sortedByCoord.out.bam"), emit: star_bam
        path("${sample}.*")

    script:
    def reads_arg    = is_paired ? "--readFilesIn $read1 $read2" : "--readFilesIn $read1"
    def pe_overlap   = is_paired ? "--peOverlapNbasesMin 10" : ""
    """
    STAR \
      --runThreadN ${task.cpus} \
      --genomeDir $star_index \
      $reads_arg \
      --readFilesCommand zcat \
      --outSAMtype BAM SortedByCoordinate \
      --outSAMattributes All \
      --quantMode GeneCounts \
      $pe_overlap \
      --outFileNamePrefix ${sample}. \
      --limitBAMsortRAM 60000000000
    """
}

process ExtractIntronicReads {
    publishDir { "${params.outdir}/preprocessing/intronic_reads/${sample}" }, mode: 'copy'
    tag "$sample"

    input:
        tuple val(sample), path(bam_file), val(strand), val(is_paired), path(introns_bed_file)

    output:
        tuple val(sample), path("intronic_reads_sorted.bam"), emit:  intronic_bam_files
        tuple val(sample), path("intronic_reads_plus_strand.bed.gz"), path("intronic_reads_minus_strand.bed.gz"),  emit:  intronic_bed_files
        tuple val(sample), path("${sample}.intron_read_counts.tsv"), emit:  intron_read_counts

    script:
    def paired_arg = is_paired ? "" : "--unpaired_sequencing"
    """
    extract_intronic_reads.py \\
        --input_bam $bam_file \\
        --intron_bed_file $introns_bed_file \\
        --strandedness $strand \\
        --edge_margin ${edge_margin()} \\
        $paired_arg

    mv intron_read_counts.tsv ${sample}.intron_read_counts.tsv

    sort -k 1,1 -k 2,2n intronic_reads_plus_strand.bed > tmp_plus_strand.bed
    mv tmp_plus_strand.bed intronic_reads_plus_strand.bed

    sort -k 1,1 -k 2,2n intronic_reads_minus_strand.bed > tmp_minus_strand.bed
    mv tmp_minus_strand.bed intronic_reads_minus_strand.bed

    pigz -p ${task.cpus} -f intronic_reads_plus_strand.bed
    pigz -p ${task.cpus} -f intronic_reads_minus_strand.bed
    """
}

process RemoveIntronicReadsFromFASTQ {
    publishDir { "${params.outdir}/preprocessing/FASTQ_without_intronic_reads/${sample}" }, mode: 'copy'
    tag "$sample"

    input:
        tuple val(sample), path(read1), path(read2), path(bam_introns), val(is_paired)

    output:
        tuple val(sample), path("R*.fastq.gz"), val(is_paired), emit: exonic_fastq

    script:
    if (is_paired)
    """
    remove_intronic_reads_from_fastq.py \\
        --bam_introns $bam_introns \\
        --input_fastq $read1 \\
        --output_fastq R1.fastq
    pigz -p ${task.cpus} -f R1.fastq

    remove_intronic_reads_from_fastq.py \\
        --bam_introns $bam_introns \\
        --input_fastq $read2 \\
        --output_fastq R2.fastq
    pigz -p ${task.cpus} -f R2.fastq
    """
    else
    """
    remove_intronic_reads_from_fastq.py \\
        --bam_introns $bam_introns \\
        --input_fastq $read1 \\
        --output_fastq R1.fastq
    pigz -p ${task.cpus} -f R1.fastq
    """
}

process SalmonQuantification {
    publishDir { "${params.outdir}/preprocessing/Salmon_quantification/${sample}" }, mode: 'copy'
    tag "$sample"

    input:
        tuple val(sample), path(reads), val(is_paired), path(salmon_index)

    output:
        tuple val(sample), path("${sample}.quant.sf"), emit: salmon_quant
        path("**")

    script:
    def reads_arg = is_paired ? "-1 ${reads[0]} -2 ${reads[1]}" : "-r ${reads[0]}"
    """
    salmon quant \
        -i $salmon_index \
        -l A \
        $reads_arg \
        -p ${task.cpus} \
        -o .
    mv quant.sf ${sample}.quant.sf
    """
}

process ComputeCoverage {
    publishDir { "${params.outdir}/preprocessing/bed_graph_intron_coverage/${sample}" }, mode: 'copy'
    tag "$sample"

    input:
        tuple val(sample), path(bed_file_plus), path(bed_file_minus), path(genome_fai_file)

    output:
        tuple val(sample), path("coverage_plus.bedGraph.gz"), path("coverage_minus.bedGraph.gz"), emit: bed_graph_files

    script:
    """
    bedtools genomecov \
        -bga \
        -split \
        -i $bed_file_plus \
        -g $genome_fai_file \
        > coverage_plus.bedGraph
        pigz -p ${task.cpus} -f coverage_plus.bedGraph

    bedtools genomecov \
        -bga \
        -split \
        -i $bed_file_minus \
        -g $genome_fai_file \
        > coverage_minus.bedGraph
        pigz -p ${task.cpus} -f coverage_minus.bedGraph
    """
}

process RescaleCoverage {
    publishDir "${params.outdir}/preprocessing/rescaled_coverage", mode: 'copy'
    tag "$sample"

    input:
        tuple val(sample), path(bedgraph_file_plus), path(bedgraph_file_minus), path(intron_counts_file), path(introns_bed_file)

    output:
    path("${sample}.parquet"), emit: coverage_parquet_file

    script:
    """
    compute_rescaled_coverage.R \
        --bed_graph_plus $bedgraph_file_plus \
        --bed_graph_minus $bedgraph_file_minus \
        --introns_bed_file $introns_bed_file \
        --output_file_basename $sample \
        --introns_count_file $intron_counts_file \
        --edge_margin ${edge_margin()}
    """
}

process IntronMetacoveragePlots {
    publishDir "${params.outdir}/preprocessing/intron_metacoverage_plots", mode: 'copy'

    input:
        tuple path(coverage_parquet_files), path(introns_bed_file), path(gene_names_csv)

    output:
    path("*.png")

    script:
    """
    plot_intron_metacoverage.py \
        --coverage_data_folder . \
        --introns_bed_file $introns_bed_file \
        --gene_names $gene_names_csv \
        --gene_set_label 'protein-coding'
    """
}

process AggregateReadCounts {
    publishDir "${params.outdir}/preprocessing/aggregated_counts", mode: 'copy'

    input:
    tuple val(sample_names), path(exon_quant_files), path(intron_counts_files), path(tx2gene), val(ignore_tx_version), file(protein_coding_genes_csv)

    output:
        path("exon_counts.tsv"), emit: exon_counts
        path("intron_counts.tsv"), emit: intron_counts
        path("library_size_factors.tsv"), emit: library_size_factors
        path("isoform_length_factors.tsv"), emit: isoform_length_factors
        path("*.png")

    script:
    """
    aggregate_read_counts.R \
      --tx2gene $tx2gene \
      --sample_names ${sample_names.join(' ')} \
      --exon_quant_files ${exon_quant_files.join(' ')} \
      --intron_counts_files ${intron_counts_files.join(' ')} \
      --ignore_tx_version $ignore_tx_version \
      --protein_coding_genes_csv $protein_coding_genes_csv
    """
}

process FindModelableGenes {
    publishDir "${params.outdir}/preprocessing/modelable_genes", mode: 'copy'

    input:
        tuple path(exon_counts_tsv), path(intron_counts_tsv), path(all_genes_csv)

    output:
    path("modelable_genes.tsv"), emit: modelable_genes
    path("modelable_introns.tsv"), emit: modelable_introns
    tuple path("non_modelable_genes.tsv"), path("non_modelable_introns.tsv")

    script:
    """
    find_modelable_genes.R \
        --exon_counts_tsv $exon_counts_tsv \
        --intron_counts_tsv $intron_counts_tsv \
        --all_genes_csv $all_genes_csv \
        --output_folder .
    """
}


workflow preprocessing_workflow {
    take:
        samplesheet
        fastq_dir
        gtf_file
        genome_fasta
        transcriptome_fasta
        gtf_source
        salmon_index_with_decoy
        star_index
        salmon_index
        ignore_tx_version
        genomic_features_dir

    main:
        def gtf_channel = Channel.value(file(gtf_file))
        def genome_fasta_channel = Channel.value(file(genome_fasta))
        def transcriptome_fasta_channel = Channel.value(file(transcriptome_fasta))

        def ignore_tx_version_channel = Channel.value(ignore_tx_version)

        def star_index_channel
        if (star_index) {
            log.info "Using provided STAR index: ${star_index}"
            star_index_channel = Channel.value(file(star_index))
        } else {
            log.info "No STAR index provided. Building from genome FASTA and GTF."
            star_index_channel = BuildStarIndex(genome_fasta_channel, gtf_channel).star_index_dir
        }

        def salmon_index_channel
        if (salmon_index) {
            log.info "Using provided Salmon index: ${salmon_index}"
            salmon_index_channel = Channel.value(file(salmon_index))
        } else {
            log.info "No Salmon index provided. Building from transcriptome FASTA."
            salmon_index_channel = BuildSalmonIndex(
                transcriptome_fasta_channel,
                genome_fasta_channel,
                gtf_channel,
                gtf_source,
                salmon_index_with_decoy).salmon_index_dir
        }

        def samples = Channel
            .fromPath(samplesheet)
            .splitCsv(header: true)
            .map { row ->
                def sample    = row.sample
                def fq1       = file("${fastq_dir}/${row.fastq_1}")
                def fq2_str   = row.fastq_2?.trim()
                def is_paired = fq2_str as boolean
                def fq2       = is_paired ? file("${fastq_dir}/${fq2_str}") : []
                def strand    = row.strandedness

                if (!fq1.exists()) error "FASTQ file not found: ${fq1}"
                if (is_paired && !fq2.exists()) error "FASTQ file not found: ${fq2}"
                if (!['forward', 'reverse'].contains(strand)) {
                    error "Invalid strandedness '${strand}' for sample '${sample}'"
                }

                tuple(sample, fq1, fq2, strand, is_paired)
            }

        def tx2gene_out = PrepareTx2Gene(gtf_channel)
        def fai_index = CreateGenomeFastaIndex(genome_fasta_channel).genome_fai_file

        def genomic_features = ExtractGenomicFeatures(gtf_channel, Channel.value(genomic_features_dir ?: ''))

        def fastqc_out = samples.map{sample, fq1, fq2, strand, is_paired -> tuple(sample, fq1, fq2, is_paired)} | FastQC
        def fastqc_out_aggregated = fastqc_out.fastqc_reports.collect()
        fastqc_out_aggregated | MultiQC

        def aligned_bams = samples
        .combine(star_index_channel)
        .map { sample, r1, r2, strand, is_paired, star_index_dir ->
            tuple(sample, r1, r2, star_index_dir, is_paired)
        } | STARAlign

       def extracted_intronic_reads = aligned_bams.star_bam
       .join(samples.map { sample, r1, r2, strand, is_paired -> tuple(sample, strand, is_paired) })
       .combine(genomic_features.introns_bed_file)
       | ExtractIntronicReads

       def exonic_fastq = samples
       .map { sample, r1, r2, strand, is_paired -> tuple(sample, r1, r2, is_paired) }
       .join(extracted_intronic_reads.intronic_bam_files)
       .map { sample, r1, r2, is_paired, bam -> tuple(sample, r1, r2, bam, is_paired) }
       | RemoveIntronicReadsFromFASTQ

       def salmon_quant_out = exonic_fastq.exonic_fastq
       .combine(salmon_index_channel) | SalmonQuantification

       def bed_graph_coverage = extracted_intronic_reads.intronic_bed_files
       .combine(fai_index)| ComputeCoverage

       def rescaled_coverage = bed_graph_coverage.bed_graph_files
       .join(extracted_intronic_reads.intron_read_counts)
       .combine(genomic_features.introns_bed_file)| RescaleCoverage

       def rescaled_coverage_combined = rescaled_coverage.collect()

       def metacoverage_plot_input = rescaled_coverage.coverage_parquet_file
       .collect()
       .map { files -> tuple([files]) }   //  Wrapping in an extra list prevents unwanted flattening behavior in .combine()
       .combine(genomic_features.introns_bed_file)
       .combine(genomic_features.protein_coding_gene_names)

       IntronMetacoveragePlots(metacoverage_plot_input)

       def data_aggregation =  salmon_quant_out.salmon_quant
       .join(extracted_intronic_reads.intron_read_counts)
       .collect(flat: false).map { list_of_tuples ->
            def list_of_tuples_sorted = list_of_tuples.sort { it[0] }
            def sample_names = list_of_tuples_sorted*.getAt(0)
            def quant_files  = list_of_tuples_sorted*.getAt(1)
            def intron_files = list_of_tuples_sorted*.getAt(2)
            tuple(sample_names, quant_files, intron_files)
       }
       .combine(tx2gene_out.tx2gene_file)
       .combine(ignore_tx_version_channel)
       .combine(genomic_features.protein_coding_gene_names) | AggregateReadCounts

       def find_modelable_genes_input =  data_aggregation.exon_counts
       .combine(data_aggregation.intron_counts)
       .combine(genomic_features.all_genes)

       def modelable_genes_output = FindModelableGenes(find_modelable_genes_input)

    emit:
        gene_names_file          = genomic_features.protein_coding_gene_names
        introns_bed_file         = genomic_features.introns_bed_file
        exon_counts              = data_aggregation.exon_counts
        intron_counts            = data_aggregation.intron_counts
        library_size_factors     = data_aggregation.library_size_factors
        isoform_length_factors   = data_aggregation.isoform_length_factors
        coverage_files           = rescaled_coverage_combined
        modelable_genes          = modelable_genes_output.modelable_genes
        modelable_introns        = modelable_genes_output.modelable_introns

}
