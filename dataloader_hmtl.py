import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from typing import List, Tuple, Dict, Optional, Union


class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-windowed time series data.
    Support for hierarchical multi-task learning with intermediate targets.
    """
    def __init__(self, 
                 windowed_features: np.ndarray, 
                 windowed_targets: np.ndarray,
                 windowed_intermediate_targets: Optional[np.ndarray] = None):
        """
        Initialize the dataset with pre-windowed features and targets.
        
        Parameters:
        -----------
        windowed_features : np.ndarray
            Pre-windowed input features array of shape (n_windows, window_size, n_features)
        windowed_targets : np.ndarray
            Pre-windowed target values array of shape:
            - (n_windows, window_size, n_targets) for many-to-many
            - (n_windows, n_targets) for many-to-one
        windowed_intermediate_targets : np.ndarray, optional
            Pre-windowed intermediate target values for hierarchical learning
            Shape: (n_windows, window_size, n_intermediate_targets)
        """
        self.windowed_features = windowed_features
        self.windowed_targets = windowed_targets
        self.windowed_intermediate_targets = windowed_intermediate_targets
        
    def __len__(self):
        # Return the number of pre-created windows
        return len(self.windowed_features)
    
    def __getitem__(self, idx):
        # Return the pre-windowed data directly
        x = self.windowed_features[idx]
        y = self.windowed_targets[idx]
        
        # Include intermediate targets if available
        if self.windowed_intermediate_targets is not None:
            intermediate_y = self.windowed_intermediate_targets[idx]
            # Return features, targets, intermediate targets, and the index
            return (torch.tensor(x, dtype=torch.float32), 
                   torch.tensor(y, dtype=torch.float32),
                   torch.tensor(intermediate_y, dtype=torch.float32),
                   idx)
        else:
            # Return features, targets, and the index (backward compatibility)
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), idx


class FloodDroughtDataLoader:
    """
    Data loader for flood and drought time series data.
    
    This class handles:
    - Loading CSV data
    - Feature selection and target definition
    - Data preprocessing (scaling, etc.)
    - Creating sliding windows with custom stride
    - Split data into train/val/test sets by date ranges
    - Creating PyTorch DataLoader objects
    - Support for both many-to-one and many-to-many sequence modeling
    """
    def __init__(self, 
                 csv_file: str = 'data_processed/KettleRiverModels_hist_scaled_combined.csv', 
                 window_size: int = 30, 
                 stride: int = 1,
                 target_col: Optional[List[str]] = None,
                 feature_cols: Optional[List[str]] = None,
                 intermediate_targets: Optional[List[str]] = None,  # New parameter for HMTL/HSTL
                 train_years: Optional[Tuple[int, int]] = None,  # (start_year, end_year) inclusive
                 val_years: Optional[Tuple[int, int]] = None,    # (start_year, end_year) inclusive
                 test_years: Optional[Tuple[int, int]] = None,   # (start_year, end_year) inclusive
                 val_ratio: float = 0.15,  # Used only if year ranges not specified
                 test_ratio: float = 0.15, # Used only if year ranges not specified
                 batch_size: int = 32,
                 scale_features: bool = True,
                 scale_targets: bool = True,
                 scale_intermediate_targets: bool = True,  # New parameter for intermediate target scaling
                 many_to_many: bool = False,
                 random_seed: int = 42):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        csv_file : str
            Path to the CSV file containing the data
        window_size : int
            Number of time steps to include in each input window
        stride : int
            Step size for the sliding window
        target_col : List[str], optional
            List of column names to use as target variables. If None, defaults to ['streamflow']
            Supports multi-task learning with multiple target variables
        feature_cols : List[str], optional
            List of column names to use as features. If None, all columns except target and Datetime will be used
        intermediate_targets : List[str], optional
            List of column names to use as intermediate targets for hierarchical learning (HMTL/HSTL)
            These are targets that are predicted first and then used as additional features for final target prediction
        train_years : Tuple[int, int], optional
            (start_year, end_year) range for training data (inclusive), e.g., (1980, 2000)
        val_years : Tuple[int, int], optional
            (start_year, end_year) range for validation data (inclusive), e.g., (1975, 1980)
        test_years : Tuple[int, int], optional
            (start_year, end_year) range for test data (inclusive), e.g., (2000, 2005)
        val_ratio : float
            Proportion of data to use for validation (used only if year ranges are not specified)
        test_ratio : float
            Proportion of data to use for testing (used only if year ranges are not specified)
        batch_size : int
            Batch size for DataLoader
        scale_features : bool
            Whether to standardize features
        scale_targets : bool
            Whether to standardize targets
        scale_intermediate_targets : bool
            Whether to standardize intermediate targets
        many_to_many : bool
            If True, use many-to-many sequence modeling (return targets for each timestep)
            If False, use many-to-one sequence modeling (predict the next timestep only)
        random_seed : int
            Random seed for reproducibility
        """
        self.csv_file = csv_file
        self.window_size = window_size
        self.stride = stride
        self.target_col = target_col if target_col is not None else ['streamflow']
        self.feature_cols = feature_cols
        self.intermediate_targets = intermediate_targets if intermediate_targets is not None else []
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.scale_features = scale_features
        self.scale_targets = scale_targets
        self.scale_intermediate_targets = scale_intermediate_targets
        self.many_to_many = many_to_many
        self.random_seed = random_seed
        
        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Placeholders
        self.data = None
        self.feature_scaler = StandardScaler() if scale_features else None
        # For multiple targets, we might need separate scalers or handle them together
        self.target_scaler = StandardScaler() if scale_targets else None
        # Separate scaler for intermediate targets in hierarchical learning
        self.intermediate_target_scaler = StandardScaler() if scale_intermediate_targets and len(self.intermediate_targets) > 0 else None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize
        self._load_data()
        
    def _load_data(self):
        """Load data from CSV and perform initial processing."""
        print(f"Loading data from {self.csv_file}")
        
        # Read the CSV file
        self.data = pd.read_csv(self.csv_file)
        
        # Convert datetime column to datetime format and set as index
        self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
        self.data.set_index('Datetime', inplace=True)
        
        # If feature columns not specified, use all columns except the targets and intermediate targets
        if self.feature_cols is None:
            exclude_cols = self.target_col + self.intermediate_targets
            self.feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        print(f"Data loaded successfully with {len(self.data)} samples")
        print(f"Target columns: {', '.join(self.target_col)}")
        print(f"Feature columns: {', '.join(self.feature_cols)}")
        if self.intermediate_targets:
            print(f"Intermediate target columns: {', '.join(self.intermediate_targets)}")
        
        # Validate that all specified columns exist in the data
        all_required_cols = self.feature_cols + self.target_col + self.intermediate_targets
        missing_cols = [col for col in all_required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
            
        print(f"All required columns found in data")
        
    def prepare_data(self):
        """Prepare data for model training by scaling and creating sliding windows."""
        # Check if we should use date-based splitting
        use_date_split = all(x is not None for x in [self.train_years, self.val_years, self.test_years])
        
        if use_date_split:
            return self._prepare_data_by_years()
        else:
            return self._prepare_data_by_ratio()
    
    def _prepare_data_by_years(self):
        """Prepare data using year-based splitting."""
        print("Using year-based data splitting")
        
        # Create separate datasets for train, validation, and test
        train_data = self._filter_data_by_years(self.train_years[0], self.train_years[1])
        val_data = self._filter_data_by_years(self.val_years[0], self.val_years[1])
        test_data = self._filter_data_by_years(self.test_years[0], self.test_years[1])
        
        # Extract features and targets for each dataset
        train_features = train_data[self.feature_cols].values
        train_targets = train_data[self.target_col].values
        
        val_features = val_data[self.feature_cols].values
        val_targets = val_data[self.target_col].values
        
        test_features = test_data[self.feature_cols].values
        test_targets = test_data[self.target_col].values
        
        # Extract intermediate targets if specified
        train_intermediate_targets = None
        val_intermediate_targets = None
        test_intermediate_targets = None
        
        if self.intermediate_targets:
            train_intermediate_targets = train_data[self.intermediate_targets].values
            val_intermediate_targets = val_data[self.intermediate_targets].values
            test_intermediate_targets = test_data[self.intermediate_targets].values
        
        # Scale features if requested (fit on training data, transform all)
        if self.scale_features:
            self.feature_scaler.fit(train_features)
            train_features = self.feature_scaler.transform(train_features)
            val_features = self.feature_scaler.transform(val_features)
            test_features = self.feature_scaler.transform(test_features)
            print("Features scaled (fit on training data only)")
        
        # Scale targets if requested (fit on training data, transform all)
        if self.scale_targets:
            self.target_scaler.fit(train_targets)
            train_targets = self.target_scaler.transform(train_targets)
            val_targets = self.target_scaler.transform(val_targets)
            test_targets = self.target_scaler.transform(test_targets)
            print("Targets scaled (fit on training data only)")
        
        # Scale intermediate targets if requested (fit on training data, transform all)
        if self.intermediate_targets and self.scale_intermediate_targets:
            self.intermediate_target_scaler.fit(train_intermediate_targets)
            train_intermediate_targets = self.intermediate_target_scaler.transform(train_intermediate_targets)
            val_intermediate_targets = self.intermediate_target_scaler.transform(val_intermediate_targets)
            test_intermediate_targets = self.intermediate_target_scaler.transform(test_intermediate_targets)
            print("Intermediate targets scaled (fit on training data only)")
        
        # Create sliding windows with specified stride for each set
        x_train, y_train, date_train = self._create_sliding_windows(
            train_features, train_targets, self.window_size, self.stride, train_data.index)
        x_val, y_val, date_val = self._create_sliding_windows(
            val_features, val_targets, self.window_size, self.stride, val_data.index)
        x_test, y_test, date_test = self._create_sliding_windows(
            test_features, test_targets, self.window_size, self.stride, test_data.index)
        
        # Create sliding windows for intermediate targets if they exist
        intermediate_train = None
        intermediate_val = None
        intermediate_test = None
        
        if self.intermediate_targets:
            intermediate_train, _, _ = self._create_sliding_windows(
                train_intermediate_targets, train_intermediate_targets, self.window_size, self.stride, train_data.index)
            intermediate_val, _, _ = self._create_sliding_windows(
                val_intermediate_targets, val_intermediate_targets, self.window_size, self.stride, val_data.index)
            intermediate_test, _, _ = self._create_sliding_windows(
                test_intermediate_targets, test_intermediate_targets, self.window_size, self.stride, test_data.index)
        
        print(f"Train data: {len(x_train)} windows ({self.train_years[0]}-{self.train_years[1]})")
        print(f"Validation data: {len(x_val)} windows ({self.val_years[0]}-{self.val_years[1]})")
        print(f"Test data: {len(x_test)} windows ({self.test_years[0]}-{self.test_years[1]})")
        
        return {
            'x_train': x_train,
            'y_train': y_train,
            'date_train': date_train,
            'x_val': x_val,
            'y_val': y_val,
            'date_val': date_val,
            'x_test': x_test,
            'y_test': y_test,
            'date_test': date_test,
            'intermediate_train': intermediate_train,
            'intermediate_val': intermediate_val,
            'intermediate_test': intermediate_test
        }
    
    def _filter_data_by_years(self, start_year, end_year):
        """Filter data by year range."""
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = pd.Timestamp(f"{end_year}-12-31 23:59:59")
        filtered_data = self.data.loc[start_date:end_date].copy()
        print(f"Filtered data for years {start_year}-{end_year}: {len(filtered_data)} samples")
        return filtered_data
    
    def _prepare_data_by_ratio(self):
        """Prepare data using ratio-based splitting."""
        print("Using ratio-based data splitting")
        
        # Extract features and target
        features = self.data[self.feature_cols].values
        targets = self.data[self.target_col].values
        
        n_samples = features.shape[0]
        # Calculate split indices
        test_size = int(n_samples * self.test_ratio)
        val_size = int(n_samples * self.val_ratio)
        train_size = n_samples - test_size - val_size
        print(f"Total samples: {n_samples}, Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")
        train_features = features[:train_size]
        train_targets = targets[:train_size]
        val_features = features[train_size:train_size+val_size]
        val_targets = targets[train_size:train_size+val_size]
        test_features = features[train_size+val_size:]
        test_targets = targets[train_size+val_size:]
        
        # Split dates as well for ratio-based splitting
        train_dates = self.data.index[:train_size]
        val_dates = self.data.index[train_size:train_size+val_size]
        test_dates = self.data.index[train_size+val_size:]
        
        # Print sizes of each split
        print(f"Train features shape: {train_features.shape}, Train targets shape: {train_targets.shape}")
        print(f"Validation features shape: {val_features.shape}, Validation targets shape: {val_targets.shape}")
        print(f"Test features shape: {test_features.shape}, Test targets shape: {test_targets.shape}")      

        # Scale features if requested
        if self.scale_features:
            train_features = self.feature_scaler.fit_transform(train_features)
            val_features = self.feature_scaler.transform(val_features)
            test_features = self.feature_scaler.transform(test_features)
            print("Features scaled (fit on training data only)")

        # Scale targets if requested
        if self.scale_targets:
            train_targets = self.target_scaler.fit_transform(train_targets)
            val_targets = self.target_scaler.transform(val_targets)
            test_targets = self.target_scaler.transform(test_targets)
            print("Targets scaled (fit on training data only)")
            
        # Create sliding windows with specified stride for each set
        x_train, y_train, date_train = self._create_sliding_windows(
            train_features, train_targets, self.window_size, self.stride, train_dates)
        x_val, y_val, date_val = self._create_sliding_windows(
            val_features, val_targets, self.window_size, self.stride, val_dates)
        x_test, y_test, date_test = self._create_sliding_windows(
            test_features, test_targets, self.window_size, self.stride, test_dates)
        

        print(f"Train data: {len(x_train)} windows (ratio={1 - self.test_ratio - self.val_ratio})")
        print(f"Validation data: {len(x_val)} windows (ratio={self.val_ratio})")
        print(f"Test data: {len(x_test)} windows (ratio={self.test_ratio})")
        
        return {
            'x_train': x_train,
            'y_train': y_train,
            'date_train': date_train,
            'x_val': x_val,
            'y_val': y_val,
            'date_val': date_val,
            'x_test': x_test,
            'y_test': y_test,
            'date_test': date_test
        }
    
    def _create_sliding_windows(self, 
                              features: np.ndarray, 
                              targets: np.ndarray,
                              window_size: int, 
                              stride: int,
                              dates: Optional[pd.DatetimeIndex] = None) -> Tuple[np.ndarray, np.ndarray, Optional[List[pd.DatetimeIndex]]]:
        """
        Create sliding windows from features, targets, and optionally dates.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature array of shape (n_samples, n_features)
        targets : np.ndarray
            Target array of shape (n_samples, n_targets)
        window_size : int
            Number of time steps to include in each window
        stride : int
            Step size for the sliding window
        dates : pd.DatetimeIndex, optional
            Datetime index corresponding to the samples
        
        Returns:
        --------
        x_windows : np.ndarray
            Array of windows for features of shape (n_windows, window_size, n_features)
        y_windows : np.ndarray
            Array of corresponding targets of shape:
            - (n_windows, window_size, n_targets) for many-to-many
            - (n_windows, n_targets) for many-to-one
        date_windows : List[pd.DatetimeIndex], optional
            List of DatetimeIndex objects for each window (if dates provided)
        """
        n_samples = features.shape[0]
        n_features = features.shape[1]
        n_targets = targets.shape[1]  # Number of target variables
        
        # Validate minimum requirements and calculate correct number of windows
        if self.many_to_many:
            min_required = window_size
            if n_samples < min_required:
                raise ValueError(f"Need at least {min_required} samples for many-to-many, got {n_samples}")
            # Last valid starting position for many-to-many
            max_start_idx = n_samples - window_size
        else:
            min_required = window_size + 1  # Need extra sample for target
            if n_samples < min_required:
                raise ValueError(f"Need at least {min_required} samples for many-to-one, got {n_samples}")
            # Last valid starting position for many-to-one
            max_start_idx = n_samples - window_size - 1
        
        # Calculate number of valid windows
        n_windows = (max_start_idx // stride) + 1
        
        print(f"Creating {n_windows} windows from {n_samples} samples "
              f"(window_size={window_size}, stride={stride}, many_to_many={self.many_to_many})")
        
        # Initialize arrays for windows
        x_windows = np.zeros((n_windows, window_size, n_features))
        
        # Initialize target windows based on many-to-many or many-to-one
        if self.many_to_many:
            # For many-to-many, targets are the target values for each timestep in window
            y_windows = np.zeros((n_windows, window_size, n_targets))
        else:
            # For many-to-one, target is the next target values after window
            y_windows = np.zeros((n_windows, n_targets))
        
        # Initialize date windows if dates are provided
        date_windows = [] if dates is not None else None
        
        # Create sliding windows
        for i in range(n_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            # Features window - now guaranteed to be valid
            x_windows[i] = features[start_idx:end_idx]
            
            if self.many_to_many:
                # For many-to-many, get targets for each timestep in window
                y_windows[i] = targets[start_idx:end_idx]
                
                # For many-to-many, date window corresponds to the feature window
                if dates is not None:
                    date_windows.append(dates[start_idx:end_idx])
            else:
                # For many-to-one, get target for next timestep after window
                y_windows[i] = targets[end_idx]
                
                # For many-to-one, we need both feature dates and target date
                if dates is not None:
                    # Include feature window dates plus the target date
                    window_dates = dates[start_idx:end_idx + 1]  # +1 to include target date
                    date_windows.append(window_dates)
        
        return x_windows, y_windows, date_windows
    
    def create_data_loaders(self, shuffle_train=True):
        """
        Create PyTorch DataLoader objects for train, validation, and test sets.
        
        Parameters:
        -----------
        shuffle_train : bool
            Whether to shuffle the training data
            
        Returns:
        --------
        dict
            Dictionary containing train, validation, and test DataLoaders
        """
        # Prepare data splits
        data_splits = self.prepare_data()
        
        # Create datasets
        self.train_dataset = TimeSeriesDataset(
            data_splits['x_train'], 
            data_splits['y_train'],
            data_splits.get('intermediate_train')  # Pass intermediate targets if available
        )
        
        self.val_dataset = TimeSeriesDataset(
            data_splits['x_val'], 
            data_splits['y_val'],
            data_splits.get('intermediate_val')  # Pass intermediate targets if available
        )
        
        self.test_dataset = TimeSeriesDataset(
            data_splits['x_test'], 
            data_splits['y_test'],
            data_splits.get('intermediate_test')  # Pass intermediate targets if available
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle_train
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        print(f"Created DataLoaders with batch_size={self.batch_size}")
        print(f"Train: {len(self.train_loader)} batches")
        print(f"Validation: {len(self.val_loader)} batches")
        print(f"Test: {len(self.test_loader)} batches")
        if self.intermediate_targets:
            print(f"Including intermediate targets: {', '.join(self.intermediate_targets)}")
        
        return {
            'train_loader': self.train_loader,
            'date_train': data_splits['date_train'],
            'val_loader': self.val_loader,
            'date_val': data_splits['date_val'],
            'test_loader': self.test_loader,
            'date_test': data_splits['date_test']
        }
    
    def inverse_transform(self, data, scaler):
        """
        Transform scaled data back to original scale.
        
        Parameters:
        -----------
        data : np.ndarray, torch.Tensor, or pd.DataFrame
            Scaled data values
        scaler : sklearn.preprocessing.StandardScaler
            The scaler used for transformation
            
        Returns:
        --------
        np.ndarray or pd.DataFrame
            Data in original scale (same type as input for DataFrame)
        """
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            # Store index and columns for later reconstruction
            original_index = data.index
            original_columns = data.columns
            
            # Convert to numpy array for inverse transform
            data_values = data.values
            
            # Apply inverse transform
            if scaler is not None:
                transformed_values = scaler.inverse_transform(data_values)
            else:
                transformed_values = data_values
            
            # Return as DataFrame with original index and columns
            return pd.DataFrame(
                transformed_values, 
                index=original_index, 
                columns=original_columns
            )
        
        # Handle numpy arrays and tensors
        if isinstance(data, torch.Tensor):
            # Convert torch tensor to numpy if necessary
            data = data.cpu().numpy()
        
        if data.ndim == 1:
            # If 1D, reshape to 2D for inverse transform
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError("Data must be 1D or 2D for inverse transformation")
        
        if scaler is not None:
            return scaler.inverse_transform(data)
        else:
            return data
        
    def get_feature_names(self):
        """Returns the names of the selected features."""
        return self.feature_cols
        
    def get_target_name(self):
        """Returns the names of the target variables."""
        return self.target_col
    
    def get_intermediate_target_names(self):
        """Returns the names of the intermediate target variables."""
        return self.intermediate_targets
    
    def get_intermediate_target_scaler(self):
        """Returns the scaler used for intermediate targets."""
        return self.intermediate_target_scaler
    
    @classmethod
    def from_config(cls, config):
        """
        Create a FloodDroughtDataLoader from a configuration dictionary.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary containing all necessary parameters
            
        Returns:
        --------
        FloodDroughtDataLoader
            Configured data loader instance
        """
        return cls(
            csv_file=config['csv_file'],
            window_size=config['window_size'],
            stride=config['stride'],
            target_col=config['target_cols'],
            feature_cols=config['feature_cols'],
            intermediate_targets=config.get('intermediate_targets'),
            train_years=tuple(config['train_years']),
            val_years=tuple(config['val_years']),
            test_years=tuple(config['test_years']),
            batch_size=config['batch_size'],
            scale_features=config.get('scale_features', True),
            scale_targets=config.get('scale_targets', True),
            scale_intermediate_targets=config.get('scale_intermediate_targets', True),
            many_to_many=True,  # Hierarchical learning typically uses many-to-many
            random_seed=config.get('seed', 42)
        )
    
    def reconstruct_time_series(self, 
                                windowed_data: np.ndarray, 
                                date_windows: List[pd.DatetimeIndex],
                                columns: List[str],
                                aggregation_method: str = 'mean') -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Reconstruct time series from windowed data using date information.
        
        This function takes windowed predictions or targets and reconstructs the original 
        time series by properly handling overlapping windows based on the stride used.
        
        Parameters:
        -----------
        windowed_data : np.ndarray
            Windowed data array of shape:
            - (n_windows, window_size, n_features) for many-to-many
            - (n_windows, n_features) for many-to-one
        date_windows : List[pd.DatetimeIndex]
            List of DatetimeIndex objects for each window
        aggregation_method : str, default='mean'
            Method to aggregate overlapping predictions:
            - 'mean': Average overlapping values
            - 'median': Median of overlapping values
            - 'last': Use the last (most recent) prediction
            - 'first': Use the first prediction
        
        Returns:
        --------
        Tuple of (time_series_df, counts_df)
        """
        if len(windowed_data) != len(date_windows):
            raise ValueError(f"Number of windows ({len(windowed_data)}) must match number of date windows ({len(date_windows)})")
        
        # Determine if this is many-to-many or many-to-one based on data shape
        if windowed_data.ndim == 3:
            # Many-to-many: (n_windows, window_size, n_features)
            is_many_to_many = True
            n_windows, window_size, n_features = windowed_data.shape
        elif windowed_data.ndim == 2:
            # Many-to-one: (n_windows, n_features)
            is_many_to_many = False
            n_windows, n_features = windowed_data.shape
            window_size = None
        else:
            raise ValueError(f"Windowed data must be 2D or 3D, got shape {windowed_data.shape}")
        
        print(f"Reconstructing time series from {n_windows} windows")
        print(f"Mode: {'many-to-many' if is_many_to_many else 'many-to-one'}")
        print(f"Features: {n_features}")
        
        # Collect all unique timestamps and create mapping
        all_timestamps = set()
        for date_window in date_windows:
            if is_many_to_many:
                # For many-to-many, use all dates in the window
                all_timestamps.update(date_window)
            else:
                # For many-to-one, use the last date (target date)
                all_timestamps.add(date_window[-1])
        
        # Convert to sorted list
        all_timestamps = sorted(list(all_timestamps))
        n_timestamps = len(all_timestamps)
        
        print(f"Total unique timestamps: {n_timestamps}")
        print(f"Date range: {all_timestamps[0]} to {all_timestamps[-1]}")
        
        # Initialize arrays to store values and counts
        # Shape: (n_timestamps, n_features)
        values_sum = np.zeros((n_timestamps, n_features))
        values_count = np.zeros((n_timestamps, n_features))
        
        # Create timestamp to index mapping for efficiency
        timestamp_to_idx = {ts: i for i, ts in enumerate(all_timestamps)}
        
        # Process each window
        for window_idx, (data_window, date_window) in enumerate(zip(windowed_data, date_windows)):
            if is_many_to_many:
                # Many-to-many: each timestep in window has a prediction
                for time_step, (values, timestamp) in enumerate(zip(data_window, date_window)):
                    ts_idx = timestamp_to_idx[timestamp]
                    values_sum[ts_idx] += values
                    values_count[ts_idx] += 1
            else:
                # Many-to-one: only the last timestamp has a prediction
                target_timestamp = date_window[-1]
                ts_idx = timestamp_to_idx[target_timestamp]
                values_sum[ts_idx] += data_window
                values_count[ts_idx] += 1
        
        # Handle aggregation
        reconstructed_values = np.zeros((n_timestamps, n_features))
        
        if aggregation_method == 'mean':
            # Avoid division by zero
            mask = values_count > 0
            reconstructed_values[mask] = values_sum[mask] / values_count[mask]
        elif aggregation_method in ['last', 'first']:
            # For last/first, we would need to track order, defaulting to mean
            print(f"Warning: {aggregation_method} aggregation not fully implemented. Using mean instead.")
            mask = values_count > 0
            reconstructed_values[mask] = values_sum[mask] / values_count[mask]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Create DataFrames
        time_series_df = pd.DataFrame(
            reconstructed_values, 
            index=pd.DatetimeIndex(all_timestamps),
            columns=columns
        )
        
        counts_df = pd.DataFrame(
            values_count, 
            index=pd.DatetimeIndex(all_timestamps),
            columns=[f'{col}_count' for col in columns]
        )
        
        # Print summary statistics
        print(f"Reconstruction complete:")
        print(f"  - Total timestamps: {len(time_series_df)}")
        print(f"  - Average overlap per timestamp: {values_count.mean():.2f}")
        print(f"  - Max overlap: {values_count.max()}")
        print(f"  - Min overlap: {values_count.min()}")
        print(f"  - Timestamps with no data: {(values_count == 0).sum()}")
        
        return time_series_df, counts_df

'''
todo:
Dataloader part.
1.Retain the date for at each timestep of the sliding window.
2. Implmente a function to stich the sliding windows back together to get the original time series.

For the local model, don't need to use the Area_ac_total

ET is important, for a variation, we can make the ET as the second target variable, and use the streamflow as the first target variable.
'''

if __name__ == "__main__":

    # Example 1: Using year-based splitting with many-to-many sequence model for multi-task learning
    print("\n----- Example 1: Year-based splitting with many-to-many sequence model (Multi-task) -----")
    data_loader_year = FloodDroughtDataLoader(
        csv_file='data_processed/KettleRiverModels_hist_scaled_combined.csv',
        window_size=14*24,  # 14 days x 24 hours window
        stride=7*24,        # Move window by 7 days x 24 hours each time
        target_col=['streamflow', 'ET'],  # Multi-task learning: predict both streamflow and ET
        feature_cols=['T2', 'DEWPT','PRECIP', 'SWDNB', 'WSPD10', 'LH'],  # Use these features
        train_years=(1980, 2000),  # 1980-2000 for training
        val_years=(1975, 1979),    # 1975-1980 for validation
        test_years=(2001, 2005),   # 2000-2005 for testing
        batch_size=64,
        # Many-to-many sequence model
        scale_features = True,
        scale_targets = True,
        many_to_many=True,
    )
    
    # Example 3: Demonstrating time series reconstruction
    print("\n----- Example 4: Time series reconstruction from windowed data -----")
    
    # Use the test dataset from the year-based splitting example
    data_splits = data_loader_year.prepare_data()
    
    # Reconstruct the test targets to verify the reconstruction function
    print("Reconstructing test targets (ground truth):")
    reconstructed_targets, recontructed_counts = data_loader_year.reconstruct_time_series(
        windowed_data=data_splits['y_test'],
        date_windows=data_splits['date_test'],
        columns=data_loader_year.get_target_name(),
        aggregation_method='mean'
    )

    print(f"Reconstructed targets shape: {reconstructed_targets.shape}")
    print(f"Reconstructed targets columns: {reconstructed_targets.columns.tolist()}")
    print(f"Date range: {reconstructed_targets.index.min()} to {reconstructed_targets.index.max()}")
    print(f"Sample of reconstructed data:")
    print(reconstructed_targets.head())

    reconstructed_targets = data_loader_year.inverse_transform(
        data=reconstructed_targets, 
        scaler=data_loader_year.target_scaler
    )
    
    print(f"Reconstructed targets (original scale) shape: {reconstructed_targets.shape}")
    print(f"Sample of inverse-transformed data:")
    print(reconstructed_targets.head())


    reconstructed_features, recontructed_feature_counts = data_loader_year.reconstruct_time_series(
        windowed_data=data_splits['x_test'],
        date_windows=data_splits['date_test'],
        columns=data_loader_year.get_feature_names(),
        aggregation_method='mean'
    )

    print(f"Reconstructed features shape: {reconstructed_features.shape}")
    print(f"Reconstructed features columns: {reconstructed_features.columns.tolist()}")
    print(f"Date range: {reconstructed_features.index.min()} to {reconstructed_features.index.max()}")
    print(f"Sample of reconstructed features:")
    print(reconstructed_features.head())
    
    # Inverse transform features back to original scale
    reconstructed_features = data_loader_year.inverse_transform(
        data=reconstructed_features, 
        scaler=data_loader_year.feature_scaler
    )
    
    print(f"Reconstructed features (original scale) shape: {reconstructed_features.shape}")
    print(f"Sample of inverse-transformed features:")
    print(reconstructed_features.head())

    # Validation: Compare reconstructed data with original data
    print("\n----- Validation: Comparing reconstructed vs original data -----")
    
    # Get original test data for comparison
    test_data_original = data_loader_year._filter_data_by_years(
        data_loader_year.test_years[0], 
        data_loader_year.test_years[1]
    )
    
    # Compare targets
    print("\n--- Target Validation ---")
    original_targets = test_data_original[data_loader_year.get_target_name()]
    
    # Find overlapping date range for comparison
    overlap_start = max(original_targets.index.min(), reconstructed_targets.index.min())
    overlap_end = min(original_targets.index.max(), reconstructed_targets.index.max())
    
    print(f"Original targets date range: {original_targets.index.min()} to {original_targets.index.max()}")
    print(f"Reconstructed targets date range: {reconstructed_targets.index.min()} to {reconstructed_targets.index.max()}")
    print(f"Overlap date range: {overlap_start} to {overlap_end}")
    
    # Create comparison DataFrame for overlapping period
    original_overlap = original_targets.loc[overlap_start:overlap_end]
    reconstructed_overlap = reconstructed_targets.loc[overlap_start:overlap_end]
    
    # Calculate reconstruction accuracy metrics
    print(f"\nTarget Reconstruction Accuracy:")
    print(f"Original data points in overlap: {len(original_overlap)}")
    print(f"Reconstructed data points in overlap: {len(reconstructed_overlap)}")
    
    if len(original_overlap) > 0 and len(reconstructed_overlap) > 0:
        # Align the data by reindexing
        common_index = original_overlap.index.intersection(reconstructed_overlap.index)
        if len(common_index) > 0:
            orig_aligned = original_overlap.loc[common_index]
            recon_aligned = reconstructed_overlap.loc[common_index]
            
            print(f"Common timestamps: {len(common_index)}")
            
            for col in data_loader_year.get_target_name():
                if col in orig_aligned.columns and col in recon_aligned.columns:
                    # Calculate metrics
                    mae = np.mean(np.abs(orig_aligned[col] - recon_aligned[col]))
                    rmse = np.sqrt(np.mean((orig_aligned[col] - recon_aligned[col])**2))
                    correlation = np.corrcoef(orig_aligned[col], recon_aligned[col])[0,1] if len(orig_aligned) > 1 else np.nan
                    
                    print(f"\n{col}:")
                    print(f"  MAE: {mae:.6f}")
                    print(f"  RMSE: {rmse:.6f}")
                    print(f"  Correlation: {correlation:.6f}")
                    print(f"  Original range: [{orig_aligned[col].min():.3f}, {orig_aligned[col].max():.3f}]")
                    print(f"  Reconstructed range: [{recon_aligned[col].min():.3f}, {recon_aligned[col].max():.3f}]")
    
    # Compare features
    print("\n--- Feature Validation ---")
    original_features = test_data_original[data_loader_year.get_feature_names()]
    
    # Find overlapping date range for features
    feature_overlap_start = max(original_features.index.min(), reconstructed_features.index.min())
    feature_overlap_end = min(original_features.index.max(), reconstructed_features.index.max())
    
    print(f"Original features date range: {original_features.index.min()} to {original_features.index.max()}")
    print(f"Reconstructed features date range: {reconstructed_features.index.min()} to {reconstructed_features.index.max()}")
    print(f"Feature overlap date range: {feature_overlap_start} to {feature_overlap_end}")
    
    # Create comparison for features
    original_features_overlap = original_features.loc[feature_overlap_start:feature_overlap_end]
    reconstructed_features_overlap = reconstructed_features.loc[feature_overlap_start:feature_overlap_end]
    
    print(f"\nFeature Reconstruction Accuracy:")
    print(f"Original feature points in overlap: {len(original_features_overlap)}")
    print(f"Reconstructed feature points in overlap: {len(reconstructed_features_overlap)}")
    
    if len(original_features_overlap) > 0 and len(reconstructed_features_overlap) > 0:
        # Align the feature data
        common_feature_index = original_features_overlap.index.intersection(reconstructed_features_overlap.index)
        if len(common_feature_index) > 0:
            orig_feat_aligned = original_features_overlap.loc[common_feature_index]
            recon_feat_aligned = reconstructed_features_overlap.loc[common_feature_index]
            
            print(f"Common feature timestamps: {len(common_feature_index)}")
            
            # Calculate average metrics across all features
            all_mae = []
            all_rmse = []
            all_corr = []
            
            for col in data_loader_year.get_feature_names()[:3]:  # Show first 3 features to avoid too much output
                if col in orig_feat_aligned.columns and col in recon_feat_aligned.columns:
                    mae = np.mean(np.abs(orig_feat_aligned[col] - recon_feat_aligned[col]))
                    rmse = np.sqrt(np.mean((orig_feat_aligned[col] - recon_feat_aligned[col])**2))
                    correlation = np.corrcoef(orig_feat_aligned[col], recon_feat_aligned[col])[0,1] if len(orig_feat_aligned) > 1 else np.nan
                    
                    all_mae.append(mae)
                    all_rmse.append(rmse)
                    if not np.isnan(correlation):
                        all_corr.append(correlation)
                    
                    print(f"\n{col}:")
                    print(f"  MAE: {mae:.6f}")
                    print(f"  RMSE: {rmse:.6f}")
                    print(f"  Correlation: {correlation:.6f}")
            
            if all_mae:
                print(f"\nAverage across displayed features:")
                print(f"  Average MAE: {np.mean(all_mae):.6f}")
                print(f"  Average RMSE: {np.mean(all_rmse):.6f}")
                if all_corr:
                    print(f"  Average Correlation: {np.mean(all_corr):.6f}")
    
    print("\n----- Validation Complete -----")
    print("Note: Small differences are expected due to:")
    print("- Window overlap averaging in reconstruction")
    print("- Stride effects at boundaries")
    print("- Floating point precision")
    print("- Edge effects in sliding windows")