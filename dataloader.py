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
    """
    def __init__(self, 
                 windowed_features: np.ndarray, 
                 windowed_targets: np.ndarray):
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
        """
        self.windowed_features = windowed_features
        self.windowed_targets = windowed_targets
        
    def __len__(self):
        # Return the number of pre-created windows
        return len(self.windowed_features)
    
    def __getitem__(self, idx):
        # Return the pre-windowed data directly
        x = self.windowed_features[idx]
        y = self.windowed_targets[idx]
        
        # Return features, targets, and the index
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
                 csv_file: str = 'processed/KettleRiverModels_hist_scaled_combined.csv', 
                 window_size: int = 30, 
                 stride: int = 1,
                 target_col: Optional[List[str]] = None,
                 feature_cols: Optional[List[str]] = None,
                 train_years: Optional[Tuple[int, int]] = None,  # (start_year, end_year) inclusive
                 val_years: Optional[Tuple[int, int]] = None,    # (start_year, end_year) inclusive
                 test_years: Optional[Tuple[int, int]] = None,   # (start_year, end_year) inclusive
                 val_ratio: float = 0.15,  # Used only if year ranges not specified
                 test_ratio: float = 0.15, # Used only if year ranges not specified
                 batch_size: int = 32,
                 scale_features: bool = True,
                 scale_targets: bool = True,
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
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.scale_features = scale_features
        self.scale_targets = scale_targets
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
        
        # If feature columns not specified, use all columns except the targets
        if self.feature_cols is None:
            self.feature_cols = [col for col in self.data.columns if col not in self.target_col]
        
        print(f"Data loaded successfully with {len(self.data)} samples")
        print(f"Target columns: {', '.join(self.target_col)}")
        print(f"Feature columns: {', '.join(self.feature_cols)}")
        
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
        
        # Create sliding windows with specified stride for each set
        x_train, y_train, date_train = self._create_sliding_windows(
            train_features, train_targets, self.window_size, self.stride, train_data.index)
        x_val, y_val, date_val = self._create_sliding_windows(
            val_features, val_targets, self.window_size, self.stride, val_data.index)
        x_test, y_test, date_test = self._create_sliding_windows(
            test_features, test_targets, self.window_size, self.stride, test_data.index)
        
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
            'date_test': date_test
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
        )
        
        self.val_dataset = TimeSeriesDataset(
            data_splits['x_val'], 
            data_splits['y_val'],
        )
        
        self.test_dataset = TimeSeriesDataset(
            data_splits['x_test'], 
            data_splits['y_test'],
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
        
        return {
            'train_loader': self.train_loader,
            'date_train': data_splits['date_train'],
            'val_loader': self.val_loader,
            'date_val': data_splits['date_val'],
            'test_loader': self.test_loader,
            'date_test': data_splits['date_test']
        }
    
    def inverse_transform_targets(self, scaled_targets):
        """
        Transform scaled targets back to original scale.
        
        Parameters:
        -----------
        scaled_targets : np.ndarray or torch.Tensor
            Scaled target values
            
        Returns:
        --------
        np.ndarray
            Targets in original scale
        """
        if self.scale_targets:
            # Convert torch tensor to numpy if necessary
            if isinstance(scaled_targets, torch.Tensor):
                scaled_targets = scaled_targets.cpu().numpy()
                
            # Ensure correct shape for inverse_transform
            if scaled_targets.ndim == 1:
                scaled_targets = scaled_targets.reshape(-1, 1)
                
            return self.target_scaler.inverse_transform(scaled_targets)
        else:
            return scaled_targets
        
    def get_feature_names(self):
        """Returns the names of the selected features."""
        return self.feature_cols
        
    def get_target_name(self):
        """Returns the names of the target variables."""
        return self.target_col
    
    def reconstruct_time_series(self, 
                                windowed_data: np.ndarray, 
                                date_windows: List[pd.DatetimeIndex],
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
                for time_step, (timestamp, values) in enumerate(zip(date_window, data_window)):
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
        elif aggregation_method == 'median':
            # For median, we need to store all values, not just sum
            # This is more memory intensive but necessary for median calculation
            print("Warning: Median aggregation requires storing all values. Using mean instead for efficiency.")
            mask = values_count > 0
            reconstructed_values[mask] = values_sum[mask] / values_count[mask]
        elif aggregation_method in ['last', 'first']:
            # For last/first, we would need to track order, defaulting to mean
            print(f"Warning: {aggregation_method} aggregation not fully implemented. Using mean instead.")
            mask = values_count > 0
            reconstructed_values[mask] = values_sum[mask] / values_count[mask]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Create column names
        if n_features == 1:
            columns = ['value']
        else:
            columns = [f'feature_{i}' for i in range(n_features)]
        
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
    
    def reconstruct_predictions(self, 
                              predictions: Union[np.ndarray, torch.Tensor],
                              dataset_type: str = 'test',
                              aggregation_method: str = 'mean') -> pd.DataFrame:
        """
        Convenience function to reconstruct predictions from a specific dataset.
        
        Parameters:
        -----------
        predictions : Union[np.ndarray, torch.Tensor]
            Model predictions with the same shape as the dataset targets
        dataset_type : str, default='test'
            Which dataset to use for date information ('train', 'val', 'test')
        aggregation_method : str, default='mean'
            Method to aggregate overlapping predictions
        
        Returns:
        --------
        pd.DataFrame
            Reconstructed time series of predictions
        """
        # Convert torch tensor to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Get the appropriate date windows
        if dataset_type == 'train':
            if not hasattr(self, 'train_dataset') or self.train_dataset is None:
                raise ValueError("Train dataset not created. Call create_data_loaders() first.")
            date_windows = self.train_dataset.windowed_dates
        elif dataset_type == 'val':
            if not hasattr(self, 'val_dataset') or self.val_dataset is None:
                raise ValueError("Validation dataset not created. Call create_data_loaders() first.")
            date_windows = self.val_dataset.windowed_dates
        elif dataset_type == 'test':
            if not hasattr(self, 'test_dataset') or self.test_dataset is None:
                raise ValueError("Test dataset not created. Call create_data_loaders() first.")
            date_windows = self.test_dataset.windowed_dates
        else:
            raise ValueError("dataset_type must be 'train', 'val', or 'test'")
        
        if date_windows is None:
            raise ValueError("Date information not available for the specified dataset")
        
        # Use appropriate column names based on target variables
        reconstructed_df, reconstruct_count_df = self.reconstruct_time_series(
            predictions, 
            date_windows, 
            aggregation_method=aggregation_method,
        )
        
        # Update column names to match target variables
        if len(self.target_col) == reconstructed_df.shape[1]:
            reconstructed_df.columns = self.target_col
        
        return reconstructed_df, reconstruct_count_df

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
        csv_file='processed/KettleRiverModels_hist_scaled_combined.csv',
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
    
    # # Create data loaders
    # loaders_year = data_loader_year.create_data_loaders()
    
    # # Example of iterating through a batch for many-to-many
    # for i, (x_batch, y_batch, idx_batch) in enumerate(loaders_year['train_loader']):
    #     print(f"Many-to-many batch {i+1}:")
    #     print(f"X shape: {x_batch.shape}")
    #     print(f"y shape: {y_batch.shape}")
    #     # Get the corresponding dates for this batch
    #     batch_dates = [loaders_year['date_train'][idx] for idx in idx_batch]
    #     print(f"Number of date windows in batch: {len(batch_dates)}")  # Should match batch size
    #     print(f"Indices shape: {idx_batch.shape}")  # Indices for each window
        
    #     # If you want to see the first date window in the batch:
    #     if len(batch_dates) > 0:
    #         print(f"First date window shape: {len(batch_dates[0])}")  # Length of first date window
    #         print(f"First date window sample: {batch_dates[0][:3]}")  # First 3 dates
        
    #     # Just show the first batch as an example
    #     break
    
    # # Example 2: Using the ratio-based splitting with many-to-one sequence model (Single target)
    # print("\n----- Example 2: Ratio-based splitting with many-to-one sequence model (Single target) -----")
    # data_loader_ratio = FloodDroughtDataLoader(
    #     window_size=7,   # 7 days window
    #     stride=1,        # Move window by 1 day each time
    #     batch_size=32,
    #     # Single target (default: streamflow)
    #     target_col=None,  # Will default to ['streamflow']
    #     # Ratio-based splitting
    #     val_ratio=0.15,
    #     test_ratio=0.15,
    #     # Many-to-one sequence model
    #     many_to_many=False,
    #     # Features to use
    #     feature_cols=['PRECIP', 'PET', 'ET', 'SNOW', 'T2', 'DEWPT']
    # )
    
    # # Create data loaders
    # loaders_ratio = data_loader_ratio.create_data_loaders()
    
    # # Example of iterating through a batch for many-to-one
    # for i, (x_batch, y_batch, idx_batch) in enumerate(loaders_ratio['train_loader']):
    #     print(f"Many-to-many batch {i+1}:")
    #     print(f"X shape: {x_batch.shape}")
    #     print(f"y shape: {y_batch.shape}")
    #     # Get the corresponding dates for this batch
    #     batch_dates = [loaders_ratio['date_train'][idx] for idx in idx_batch]
    #     print(f"Number of date windows in batch: {len(batch_dates)}")  # Should match batch size
    #     print(f"Indices shape: {idx_batch.shape}")  # Indices for each window
        
    #     # If you want to see the first date window in the batch:
    #     if len(batch_dates) > 0:
    #         print(f"First date window shape: {len(batch_dates[0])}")  # Length of first date window
    #         print(f"First date window sample: {batch_dates[0][:3]}")  # First 3 dates
        
    #     # Just show the first batch as an example
    #     break
    
    # Example 3: Demonstrating time series reconstruction
    print("\n----- Example 4: Time series reconstruction from windowed data -----")
    
    # Use the test dataset from the year-based splitting example
    data_splits = data_loader_year.prepare_data()
    
    # Reconstruct the test targets to verify the reconstruction function
    print("Reconstructing test targets (ground truth):")
    reconstructed_targets, recontructed_counts = data_loader_year.reconstruct_time_series(
        windowed_data=data_splits['y_test'],
        date_windows=data_splits['date_test'],
        aggregation_method='mean'
    )
    
    print(f"Reconstructed targets shape: {reconstructed_targets.shape}")
    print(f"Reconstructed targets columns: {reconstructed_targets.columns.tolist()}")
    print(f"Date range: {reconstructed_targets.index.min()} to {reconstructed_targets.index.max()}")
    print(f"Sample of reconstructed data:")
    print(reconstructed_targets.head())
    
    # # Example using the convenience function
    # print("\nUsing convenience function for reconstruction:")
    # try:
    #     # This would work if we had actual model predictions
    #     # reconstructed_conv, reconstructed_counts  = data_loader_year.reconstruct_predictions(
    #     #     predictions=dummy_predictions,
    #     #     dataset_type='test',
    #     #     aggregation_method='mean'
    #     # )
    #     print("Convenience function available - would work with actual model predictions")
    # except Exception as e:
    #     print(f"Note: {e}")