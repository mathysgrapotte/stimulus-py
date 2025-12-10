
import numpy as np
import pytest
from stimulus.data.interface.dataset_interface import H5adDataset
from stimulus.data.splitting.splitters import TargetGeneSplitter

# Path to the test data
VCC_SUBSET_PATH = "tests/test_data/vcc_subset/vcc_training_subset.h5ad"

@pytest.fixture
def vcc_dataset():
    """Load the VCC subset dataset."""
    return H5adDataset.load_from_disk(VCC_SUBSET_PATH)

def test_target_gene_splitter_with_list(vcc_dataset):
    """Test splitting with a specific list of target genes."""
    target_genes = ["PBX1"]
    splitter = TargetGeneSplitter(target_genes=target_genes, target_gene_col="target_gene")
    
    # We need to extract the target gene column data to pass to get_split_indexes
    # The split API usually handles this, but for unit testing the splitter directly:
    # splitter.get_split_indexes expects a dict of {col_name: list_of_values}
    
    # Let's get the column data directly from the dataset
    target_gene_values = vcc_dataset.get_column("train", "target_gene")
    data = {"target_gene": target_gene_values}
    
    train_indices, test_indices = splitter.get_split_indexes(data)
    
    # Verify split sizes
    assert len(train_indices) + len(test_indices) == len(target_gene_values)
    
    # Create subsets to verify contents
    # H5adDataset.select_split returns a subset object (AnnData view usually)
    # But here we just want to verify the indices against the original list
    
    train_genes = [target_gene_values[i] for i in train_indices]
    test_genes = [target_gene_values[i] for i in test_indices]
    
    # Assert all 'PBX1' are in test
    assert all(gene == "PBX1" for gene in test_genes)
    # Assert no 'PBX1' in train
    assert all(gene != "PBX1" for gene in train_genes)
    # Assert we actually have some test samples (PBX1 exists in dataset)
    assert len(test_genes) > 0

def test_target_gene_splitter_with_ratio(vcc_dataset):
    """Test splitting with a ratio of target genes."""
    # Split 20% of genes to test
    split_ratio = [0.8, 0.2]
    splitter = TargetGeneSplitter(split_ratio=split_ratio, target_gene_col="target_gene", seed=42)
    
    target_gene_values = vcc_dataset.get_column("train", "target_gene")
    data = {"target_gene": target_gene_values}
    
    train_indices, test_indices = splitter.get_split_indexes(data)
    
    # Verify split sizes
    assert len(train_indices) + len(test_indices) == len(target_gene_values)
    
    train_genes = set([target_gene_values[i] for i in train_indices])
    test_genes = set([target_gene_values[i] for i in test_indices])
    
    # Assert no overlap in genes between train and test
    # (assuming genes form a disjoint partition)
    assert train_genes.isdisjoint(test_genes)
    
    # Check rough ratio of GENES (not samples)
    all_genes = set(target_gene_values)
    # Filter out controls if they are special? 
    # The prompt imply simple gene split.
    # But usually 'non-targeting' might be in both or handled specially.
    # For this basic test, we check if the number of unique genes in test is roughly 20% of total
    
    n_total = len(all_genes)
    n_test = len(test_genes)
    
    # It won't be exact due to discreteness and maybe rounding, but should be close
    ratio = n_test / n_total
    assert 0.1 <= ratio <= 0.3 # Broad range check

def test_target_gene_splitter_with_api(vcc_dataset):
    """Test splitting using the high-level stimulus.api.split function."""
    from stimulus.api import split
    
    target_genes = ["PBX1"]
    splitter = TargetGeneSplitter(target_genes=target_genes, target_gene_col="target_gene")
    
    # Use the API to split
    split_dataset = split(vcc_dataset, splitter, ["target_gene"])
    
    # Verify split names
    assert "train" in split_dataset.split_names
    assert "test" in split_dataset.split_names
    
    # Verify content of test split
    test_genes = split_dataset.get_column("test", "target_gene")
    train_genes = split_dataset.get_column("train", "target_gene")
    
    # Assert all 'PBX1' are in test
    assert all(gene == "PBX1" for gene in test_genes)
    # Assert no 'PBX1' in train
    assert all(gene != "PBX1" for gene in train_genes)
    # Assert we actually have some test samples
    assert len(test_genes) > 0

