# Dataset

The datase object is the interface between your data and stimulus. It explains stimulus how to load it from disk, how to split it, convert it to a PyTorch dataset etc.

## Supported Formats

`stimulus-py` primarily supports **Parquet** (`.parquet`). This is the preferred format for tabular data as it is fast, efficient, and typed.

However, you can always extend `stimulus-py` to support your own formats (following the minimal abstraction principle).

### Auto-Detection

`stimulus-py` uses a file extension based auto-detection system. When you provide a path to a dataset, the system:

1.  Recursively searches for the first file in the directory.
2.  Checks its extension against all registered `StimulusDataset` subclasses.
3.  Instantiates the matching class.

To enable this for your dataset, your class must implement the `file_extensions` class method.

### Interface Requirements

To create a custom dataset, inherit from `stimulus.data.interface.dataset_interface.StimulusDataset` and implement the following abstract methods.

#### 1. Registration & Identification

```python
@classmethod
def file_extensions(cls) -> list[str]:
    """Return list of supported extensions, e.g., ['.myfmt']"""
```

#### 2. Metadata & Introspection

```python
@property
def split_names(self) -> list[str]:
    """Return available splits, e.g., ['train', 'val']"""

@property
def column_names(self) -> dict[str, list[str]]:
    """Return columns for each split, e.g., {'train': ['text', 'label']}"""

def get_column(self, split: str, column_name: str) -> Union[list, np.ndarray]:
    """Return all values for a column in a particular split"""
```

#### 3. Core Loading

```python
@classmethod
def load_from_disk(cls, path: str, **kwargs) -> "StimulusDataset":
    """Load the dataset from a file or directory path."""

def save(self, path: str) -> None:
    """Save the dataset to the specified directory.
    
    Should save one to two files, one for the train data and one for the val data (if available)."""
```

#### 4. Interaction with PyTorch

```python
def get_torch_dataset(self, split: Union[str, list[str]]) -> Dataset | dict[str, Dataset]:
    """
    Return a standard torch.utils.data.Dataset.
    If split is a list, return a dictionary of Datasets. 
    E.g. {'train': train_dataset, 'val': val_dataset}
    """
```

#### 5. Splitting Logic

These methods are used by the CLI tools (like `stimulus split`) to partition your data.

```python
def select_split(self, split: str, indices: Any) -> Any:
    """Return a subset of the split data corresponding to the indices."""

def create_from_splits(self, splits: dict[str, Any]) -> "StimulusDataset":
    """Reconstruct a dataset from a dictionary of split objects."""
```

#### 6. Transformations 

Implementation of `map`, `apply`, and `filter` allows your dataset to work with `stimulus transform` and other processing pipelines.

```python
def map(self, function: Callable, ...) -> "StimulusDataset":
    """Apply a function to every sample through parallel processing. (can be a simple for loop)"""

def apply(self, transformation: Callable) -> "StimulusDataset":
    """Apply a transformation to the entire dataset."""

def filter(self, function: Callable, ...) -> "StimulusDataset":
    """Filter samples based on a predicate."""
```

You can find an example of a custom implemented dataset for the .h5ad format in `stimulus.data.data_interface` in the class `H5adDataset`.