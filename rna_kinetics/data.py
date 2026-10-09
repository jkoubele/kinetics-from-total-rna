from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


@dataclass
class DatasetMetadata:
    design_matrix: torch.Tensor
    library_sizes: torch.Tensor
    log_library_sizes: torch.Tensor
    feature_names: list[str]
    sample_names: list[str]
    lrt_metadata: pd.DataFrame
    reduced_matrices: dict[str, torch.Tensor]

    def to(self, device):
        self.design_matrix = self.design_matrix.to(device)
        self.library_sizes = self.library_sizes.to(device)
        self.log_library_sizes = self.log_library_sizes.to(device)
        for name, matrix in self.reduced_matrices.items():
            self.reduced_matrices[name] = matrix.to(device)
        return self


@dataclass
class IntronData:
    intron_name: str
    gene_name: str
    coverage: torch.Tensor  # shape (num_samples, 1, num_coverage_bins)

    @property
    def intron_names(self) -> list[str]:
        return [self.intron_name]

    def to(self, device):
        self.coverage = self.coverage.to(device)
        return self


@dataclass
class GeneData:
    gene_name: str
    intron_names: list[str]
    exon_reads: torch.Tensor  # shape (num_samples,)
    intron_reads: torch.Tensor  # shape (num_samples, num_introns)
    coverage: torch.Tensor  # shape (num_samples, num_introns, num_coverage_bins)
    isoform_length_offset: torch.Tensor  # shape (num_samples,)

    def to(self, device):
        self.exon_reads = self.exon_reads.to(device)
        self.intron_reads = self.intron_reads.to(device)
        self.coverage = self.coverage.to(device)
        self.isoform_length_offset = self.isoform_length_offset.to(device)
        return self


@dataclass
class GlobalGeneData:
    gene_names: list[str]
    intron_names: list[str]
    exon_reads: torch.Tensor  # (num_samples, num_genes)
    intron_reads: torch.Tensor  # (num_samples, num_introns)
    coverage: torch.Tensor  # (num_samples, num_introns, num_bins)
    isoform_length_offset: torch.Tensor  # (num_samples, num_genes)
    gene_idx: torch.Tensor  # (num_introns,) int64 — maps intron index -> gene index

    def to(self, device):
        self.exon_reads = self.exon_reads.to(device)
        self.intron_reads = self.intron_reads.to(device)
        self.coverage = self.coverage.to(device)
        self.isoform_length_offset = self.isoform_length_offset.to(device)
        self.gene_idx = self.gene_idx.to(device)
        return self


def concat_gene_data_list(gene_data_list: list[GeneData]) -> GlobalGeneData:
    gene_idx = torch.cat([
        torch.full((len(g.intron_names),), i, dtype=torch.long)
        for i, g in enumerate(gene_data_list)
    ])
    return GlobalGeneData(
        gene_names=[g.gene_name for g in gene_data_list],
        intron_names=[name for g in gene_data_list for name in g.intron_names],
        exon_reads=torch.stack([g.exon_reads for g in gene_data_list], dim=1),
        intron_reads=torch.cat([g.intron_reads for g in gene_data_list], dim=1),
        coverage=torch.cat([g.coverage for g in gene_data_list], dim=1),
        isoform_length_offset=torch.stack([g.isoform_length_offset for g in gene_data_list], dim=1),
        gene_idx=gene_idx,
    )


def _validate_reduced_matrix(design_matrix: np.ndarray,
                             reduced_matrix: np.ndarray,
                             expected_lrt_df: int,
                             test_id: str) -> None:
    """
    Check that the reduced design is a proper nested submodel of the full one, with the degrees
    of freedom R reported.

    The constant column belongs in both spans: the models carry their own intercepts, so what the
    likelihood can distinguish is each design's contribution up to an additive constant. Without
    it the check would reject valid reduced matrices, since the categorical branch of
    create_design_matrices.R relevels the factor before dropping a column, which leaves the raw
    column spaces non-nested.
    """
    constant = np.ones((design_matrix.shape[0], 1))
    rank_full = np.linalg.matrix_rank(np.hstack([constant, design_matrix]))
    rank_reduced = np.linalg.matrix_rank(np.hstack([constant, reduced_matrix]))
    rank_combined = np.linalg.matrix_rank(np.hstack([constant, design_matrix, reduced_matrix]))

    if rank_combined != rank_full:
        raise ValueError(
            f"Reduced matrix for {test_id} is not nested in the full design: adding its columns "
            f"raises the rank from {rank_full} to {rank_combined}, so the LRT "
            f"would not compare nested models.")
    if rank_full - rank_reduced != expected_lrt_df:
        raise ValueError(
            f"Reduced matrix for {test_id} drops {rank_full - rank_reduced} degrees of freedom, "
            f"but lrt_metadata reports lrt_df={expected_lrt_df}. The p-values would use the wrong "
            f"null distribution.")


def load_dataset_metadata(design_matrix_file: Path,
                          library_size_factors_file: Path,
                          lrt_metadata_file: Path,
                          reduced_matrices_folder: Path) -> DatasetMetadata:
    # Sample names are kept as strings: they are used to select columns of the count tables and to
    # build coverage file names, and purely numeric sample names would otherwise be parsed as ints.
    design_matrix_df = pd.read_csv(design_matrix_file, sep='\t', dtype={'sample': str})
    library_size_factors_df = pd.read_csv(library_size_factors_file, sep='\t', dtype={'sample': str})

    design_matrix_df = design_matrix_df.merge(library_size_factors_df,
                                              left_on='sample',
                                              right_on='sample')
    library_sizes = torch.tensor(design_matrix_df.pop('library_size_factor'),
                                 dtype=torch.float32)
    design_matrix_df = design_matrix_df.set_index('sample')

    lrt_metadata = pd.read_csv(lrt_metadata_file, sep='\t').set_index('test_id', drop=False)
    reduced_matrices: dict[str, torch.Tensor] = {}
    for test_id in lrt_metadata['test_id']:
        reduced_matrix_df = pd.read_csv(reduced_matrices_folder / f"{test_id}.tsv", sep='\t',
                                        dtype={'sample': str}).set_index('sample')
        if not all(design_matrix_df.index == reduced_matrix_df.index):
            raise ValueError(
                f"Reduced matrix index {reduced_matrix_df.index} does not equal to design matrix index {design_matrix_df.index}.")
        _validate_reduced_matrix(design_matrix_df.values, reduced_matrix_df.values,
                                 int(lrt_metadata.loc[test_id, 'lrt_df']), test_id)
        reduced_matrices[test_id] = torch.tensor(reduced_matrix_df.values, dtype=torch.float32)

    dataset_metadata = DatasetMetadata(design_matrix=torch.tensor(design_matrix_df.values, dtype=torch.float32),
                                       library_sizes=library_sizes,
                                       log_library_sizes=torch.log(library_sizes),
                                       feature_names=design_matrix_df.columns.tolist(),
                                       sample_names=design_matrix_df.index.tolist(),
                                       lrt_metadata=lrt_metadata,
                                       reduced_matrices=reduced_matrices)
    return dataset_metadata


def load_gene_data_list(modeled_genes_file: Path,
                        modeled_introns_file: Path,
                        exon_counts_file: Path,
                        intron_counts_file: Path,
                        isoform_length_factors_file: Path,
                        coverage_folder: Path,
                        sample_names: list[str]) -> list[GeneData]:
    modeled_genes_df = pd.read_csv(modeled_genes_file, sep='\t')
    exon_counts_df = pd.read_csv(exon_counts_file, sep='\t').set_index('gene_id')
    exon_counts_df = exon_counts_df[sample_names]

    intron_counts_df = pd.read_csv(intron_counts_file, sep='\t').set_index('intron_id')
    intron_counts_df = intron_counts_df[sample_names]

    isoform_length_factors_df = pd.read_csv(isoform_length_factors_file, sep='\t').set_index('gene_id')
    isoform_length_factors_df = isoform_length_factors_df[sample_names]

    modeled_introns_df = pd.read_csv(modeled_introns_file, sep='\t')

    gene_to_introns = (
        modeled_introns_df
        .groupby("gene_id")["intron_id"]
        .agg(list)
        .to_dict()
    )

    coverage_df_by_sample: dict[str, pd.DataFrame] = {}
    for sample_name in tqdm(sample_names, desc='Loading coverage data'):
        coverage_df = pd.read_parquet(coverage_folder / f'{sample_name}.parquet')
        coverage_df = coverage_df.set_index('intron_name')
        coverage_df_by_sample[sample_name] = coverage_df

    gene_data_list: list[GeneData] = []

    for gene_id in tqdm(modeled_genes_df['gene_id'], desc='Preparing gene data'):
        exon_row = exon_counts_df.loc[gene_id]

        # Sort introns by position in the gene
        gene_intron_names = sorted(
            gene_to_introns[gene_id],
            key=lambda x: int(x.rsplit('_', maxsplit=1)[1]),
        )

        exon_reads = torch.tensor(exon_row.values, dtype=torch.float32)
        intron_reads = torch.tensor(intron_counts_df.loc[gene_intron_names].values, dtype=torch.float32).T

        coverage = torch.tensor(np.stack([
            np.stack([coverage_df_by_sample[sample].loc[intron].values for intron in gene_intron_names])
            for sample in sample_names
        ]), dtype=torch.float32)

        if not torch.allclose(coverage.sum(axis=2), intron_reads):
            raise ValueError(f'Intron coverage for gene {gene_id} does not sum up to its intron read counts.')

        isoform_length_factors = torch.tensor(isoform_length_factors_df.loc[gene_id].values,
                                              dtype=torch.float32)
        isoform_length_offset = torch.log(isoform_length_factors)

        gene_data = GeneData(gene_name=gene_id,
                             intron_names=gene_intron_names,
                             exon_reads=exon_reads,
                             intron_reads=intron_reads,
                             coverage=coverage,
                             isoform_length_offset=isoform_length_offset)
        gene_data_list.append(gene_data)

    return gene_data_list


def load_intron_coverage_list(
        modeled_introns_file: Path,
        coverage_folder: Path,
        sample_names: list[str],
) -> list[IntronData]:
    introns_df = pd.read_csv(modeled_introns_file, sep='\t')

    coverage_df_by_sample: dict[str, pd.DataFrame] = {}
    for sample_name in tqdm(sample_names, desc='Loading coverage data'):
        coverage_df = pd.read_parquet(coverage_folder / f'{sample_name}.parquet')
        coverage_df = coverage_df.set_index('intron_name')
        coverage_df_by_sample[sample_name] = coverage_df

    intron_data_list: list[IntronData] = []
    for _, row in tqdm(introns_df.iterrows(), total=len(introns_df), desc='Preparing intron data'):
        intron_id = row['intron_id']
        gene_id = row['gene_id']
        coverage = torch.tensor(np.stack([
            coverage_df_by_sample[sample].loc[intron_id].values
            for sample in sample_names
        ]), dtype=torch.float32).unsqueeze(1)  # (num_samples, 1, num_coverage_bins)
        intron_data_list.append(IntronData(intron_name=intron_id, gene_name=gene_id, coverage=coverage))

    return intron_data_list
