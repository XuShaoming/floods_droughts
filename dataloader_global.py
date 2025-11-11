import os
from itertools import product
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class GlobalTimeSeriesDataset(Dataset):
    """Dataset that returns (dynamic_window, static_vector, target_window)."""

    def __init__(
        self,
        windowed_features: np.ndarray,
        windowed_targets: np.ndarray,
        static_features: Optional[np.ndarray] = None,
    ):
        self.windowed_features = windowed_features.astype(np.float32)
        self.windowed_targets = windowed_targets.astype(np.float32)
        if static_features is not None:
            self.static_features = static_features.astype(np.float32)
        else:
            # Use empty placeholder tensors if static inputs are disabled
            self.static_features = None

    def __len__(self) -> int:
        return len(self.windowed_features)

    def __getitem__(self, idx: int):
        dynamic_window = torch.from_numpy(self.windowed_features[idx])
        target_window = torch.from_numpy(self.windowed_targets[idx])

        if self.static_features is not None:
            static_vector = torch.from_numpy(self.static_features[idx])
        else:
            static_vector = torch.empty(0, dtype=torch.float32)

        return dynamic_window, static_vector, target_window


class GlobalFloodDroughtDataLoader:
    """
    Data loader that combines multiple watershed/time-period CSVs plus optional
    static watershed attributes for global model training and inference.
    """

    def __init__(
        self,
        data_dir: str,
        watersheds: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        csv_pattern: str = "{watershed}_{scenario}_combined.csv",
        window_size: int = 30,
        stride: int = 1,
        target_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        batch_size: int = 32,
        scale_features: bool = True,
        scale_targets: bool = True,
        many_to_many: bool = True,
        random_seed: int = 42,
        use_static_attributes: bool = True,
        static_attributes_file: Optional[str] = None,
        static_attribute_id_col: str = "characteristic_id",
        static_attribute_value_col: str = "value",
        static_attribute_model_col: str = "model",
        scale_static_attributes: bool = True,
        dataset_splits: Optional[Dict[str, List[Dict]]] = None,
        scenario_date_ranges: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        self.data_dir = data_dir
        self.csv_pattern = csv_pattern
        self.window_size = window_size
        self.stride = stride
        self.target_cols = target_cols if target_cols is not None else ["streamflow"]
        self.feature_cols = feature_cols
        self.batch_size = batch_size
        self.scale_features = scale_features
        self.scale_targets = scale_targets
        self.many_to_many = many_to_many
        self.random_seed = random_seed
        self.use_static_attributes = use_static_attributes
        self.static_attributes_file = static_attributes_file
        self.static_attribute_id_col = static_attribute_id_col
        self.static_attribute_value_col = static_attribute_value_col
        self.static_attribute_model_col = static_attribute_model_col
        self.scale_static_attributes = scale_static_attributes
        self.dataset_splits = dataset_splits or {}
        if not self.dataset_splits:
            raise ValueError("dataset_splits configuration is required for the global data loader.")

        self.available_watersheds = self._collect_unique_from_splits("watersheds")
        self.available_scenarios = self._collect_unique_from_splits("scenarios")

        self.watersheds = self._resolve_requested_entities(watersheds, self.available_watersheds, "watershed")
        self.scenarios = self._resolve_requested_entities(scenarios, self.available_scenarios, "scenario")
        self.scenario_date_ranges = self._prepare_scenario_ranges(scenario_date_ranges)

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.required_combinations = self._determine_required_combinations()
        self.dataset_map = self._load_dynamic_datasets()
        if not self.dataset_map:
            raise ValueError("No dynamic datasets were loaded. Please check paths and naming pattern.")

        if self.feature_cols is None:
            # Infer feature columns by removing targets and metadata columns
            sample_df = next(iter(self.dataset_map.values()))
            sample_columns = sample_df.columns
            exclude_cols = set(self.target_cols + ["Datetime", "watershed", "scenario"])
            self.feature_cols = [col for col in sample_columns if col not in exclude_cols]

        self.dynamic_input_size = len(self.feature_cols)
        self.num_targets = len(self.target_cols)

        self.feature_scaler = StandardScaler() if scale_features else None
        self.target_scaler = StandardScaler() if scale_targets else None
        self.static_scaler = StandardScaler() if (use_static_attributes and scale_static_attributes) else None

        self.static_attribute_lookup = {}
        self.static_attribute_names: List[str] = []
        self.static_input_size = 0

        if self.use_static_attributes:
            if not self.static_attributes_file:
                raise ValueError("static_attributes_file must be provided when use_static_attributes=True")
            self._prepare_static_attributes()

        # Placeholders populated after prepare_data()
        self.prepared_data: Optional[Dict[str, np.ndarray]] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def _collect_unique_from_splits(self, key: str) -> List[str]:
        values = set()
        for selections in self.dataset_splits.values():
            for selection in selections or []:
                for value in selection.get(key, []):
                    values.add(value)
        return sorted(values)

    def _resolve_requested_entities(
        self,
        requested: Optional[List[str]],
        available: List[str],
        label: str,
    ) -> List[str]:
        if requested is None:
            if not available:
                raise ValueError(f"No {label}s found in dataset_splits configuration.")
            return available

        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(
                f"Requested {label}s {missing} are not present in dataset_splits configuration ({available})."
            )
        return requested

    def _prepare_scenario_ranges(self, scenario_date_ranges: Optional[Dict[str, Dict[str, str]]]):
        default_ranges = {
            "hist_scaled": ("1975-01-01 00:00:00", "2005-12-31 23:00:00"),
            "RCP4.5": ("2025-01-01 00:00:00", "2099-12-31 23:00:00"),
            "RCP8.5": ("2025-01-01 00:00:00", "2099-12-31 23:00:00"),
        }
        configured = scenario_date_ranges or {}
        parsed = {}

        for scenario in self.scenarios:
            start = configured.get(scenario, {}).get("start")
            end = configured.get(scenario, {}).get("end")
            if start is None or end is None:
                defaults = default_ranges.get(scenario, (None, None))
                start = start or defaults[0]
                end = end or defaults[1]

            if start is None or end is None:
                parsed[scenario] = None
                continue

            parsed[scenario] = (pd.Timestamp(start), pd.Timestamp(end))
        return parsed

    def _determine_required_combinations(self) -> List[Tuple[str, str]]:
        combos = set()
        for selections in self.dataset_splits.values():
            for selection in selections or []:
                ws_list = selection.get("watersheds", self.available_watersheds)
                sc_list = selection.get("scenarios", self.available_scenarios)
                for watershed in ws_list:
                    if watershed not in self.watersheds:
                        continue
                    for scenario in sc_list:
                        if scenario not in self.scenarios:
                            continue
                        combos.add((watershed, scenario))

        if not combos:
            combos = set(product(self.watersheds, self.scenarios))

        return sorted(combos)

    def _load_dynamic_datasets(self) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Load all requested watershed/scenario combinations."""
        datasets: Dict[Tuple[str, str], pd.DataFrame] = {}

        for watershed, scenario in self.required_combinations:
            relative_path = self.csv_pattern.format(watershed=watershed, scenario=scenario)
            csv_path = os.path.join(self.data_dir, relative_path)

            if not os.path.exists(csv_path):
                warnings.warn(f"CSV not found for {watershed} - {scenario}: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            if "Datetime" not in df.columns:
                raise ValueError(f"'Datetime' column missing in {csv_path}")

            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.sort_values("Datetime").set_index("Datetime")
            df["watershed"] = watershed
            df["scenario"] = scenario

            missing_targets = [col for col in self.target_cols if col not in df.columns]
            if missing_targets:
                raise ValueError(f"Missing target columns {missing_targets} in {csv_path}")

            datasets[(watershed, scenario)] = df

        return datasets

    def _prepare_static_attributes(self):
        """Load and pivot static watershed attributes, then build lookup table."""
        df = pd.read_csv(self.static_attributes_file)
        
        # Handle case where static file uses shortened names (e.g., 'Kettle' vs 'KettleRiverModels')
        # Filter rows where static name matches or is contained in any requested watershed
        available_static_names = set(df[self.static_attribute_model_col].unique())
        
        def matches_any_watershed(static_name: str) -> bool:
            """Check if static_name matches or is contained in any watershed name."""
            for watershed in self.watersheds:
                if static_name == watershed or static_name in watershed:
                    return True
            return False
        
        df = df[df[self.static_attribute_model_col].apply(matches_any_watershed)]

        if df.empty:
            raise ValueError(
                f"No matching rows found in static attribute file for requested watersheds {self.watersheds}. "
                f"Available static names: {sorted(available_static_names)}"
            )

        pivot = df.pivot_table(
            index=self.static_attribute_model_col,
            columns=self.static_attribute_id_col,
            values=self.static_attribute_value_col,
            aggfunc="first",
        )

        # Create mapping from static file names to watershed names
        # Map each static name to the watershed it's contained in
        static_to_watershed_map = {}
        for static_name in pivot.index:
            for watershed in self.watersheds:
                if static_name == watershed or static_name in watershed:
                    static_to_watershed_map[static_name] = watershed
                    break
        
        # Rename index from static file names to watershed names
        pivot.index = pivot.index.map(lambda x: static_to_watershed_map.get(x, x))

        # Ensure every requested watershed has a row
        pivot = pivot.reindex(self.watersheds)

        # Fill missing values column-wise, fallback to zeros
        pivot = pivot.astype(float)
        pivot = pivot.fillna(pivot.mean())
        pivot = pivot.fillna(0.0)

        if self.static_scaler is not None:
            scaled = self.static_scaler.fit_transform(pivot.values)
            pivot.loc[:, :] = scaled

        self.static_attribute_names = pivot.columns.tolist()
        self.static_input_size = pivot.shape[1]

        for watershed in pivot.index:
            values = pivot.loc[watershed].values.astype(np.float32)
            self.static_attribute_lookup[watershed] = values

    def _create_sliding_windows(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ):
        n_samples = features.shape[0]
        n_features = features.shape[1]
        n_targets = targets.shape[1]

        if self.many_to_many:
            min_required = self.window_size
            if n_samples < min_required:
                return np.empty((0, self.window_size, n_features)), np.empty((0, self.window_size, n_targets)), []
            max_start_idx = n_samples - self.window_size
        else:
            min_required = self.window_size + 1
            if n_samples < min_required:
                return np.empty((0, self.window_size, n_features)), np.empty((0, n_targets)), []
            max_start_idx = n_samples - self.window_size - 1

        n_windows = (max_start_idx // self.stride) + 1
        x_windows = np.zeros((n_windows, self.window_size, n_features), dtype=np.float32)
        if self.many_to_many:
            y_windows = np.zeros((n_windows, self.window_size, n_targets), dtype=np.float32)
        else:
            y_windows = np.zeros((n_windows, n_targets), dtype=np.float32)

        date_windows = [] if dates is not None else None

        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size

            x_windows[i] = features[start_idx:end_idx]

            if self.many_to_many:
                y_windows[i] = targets[start_idx:end_idx]
                if date_windows is not None:
                    date_windows.append(dates[start_idx:end_idx])
            else:
                y_windows[i] = targets[end_idx]
                if date_windows is not None:
                    date_windows.append(dates[end_idx])

        return x_windows, y_windows, date_windows

    def _init_prepared_dict(self) -> Dict[str, List]:
        return {
            "x_train": [],
            "y_train": [],
            "static_train": [],
            "x_val": [],
            "y_val": [],
            "static_val": [],
            "x_test": [],
            "y_test": [],
            "static_test": [],
            "date_train": [],
            "date_val": [],
            "date_test": [],
        }

    def _build_prepared_from_cache(self, split_cache: Dict[str, List[Dict]]) -> Dict[str, np.ndarray]:
        prepared = self._init_prepared_dict()

        train_entries = split_cache.get("train", [])
        if self.scale_features:
            train_feats = [entry["features"] for entry in train_entries if entry["features"].size > 0]
            if not train_feats:
                raise ValueError("No training features available to fit feature scaler.")
            self.feature_scaler.fit(np.vstack(train_feats))

        if self.scale_targets:
            train_tgts = [entry["targets"] for entry in train_entries if entry["targets"].size > 0]
            if not train_tgts:
                raise ValueError("No training targets available to fit target scaler.")
            self.target_scaler.fit(np.vstack(train_tgts))

        for split_name in ["train", "val", "test"]:
            for entry in split_cache.get(split_name, []):
                features = entry["features"]
                targets = entry["targets"]
                dates = entry["dates"]
                watershed = entry["watershed"]

                if features.size == 0 or targets.size == 0:
                    continue

                if self.scale_features:
                    features = self.feature_scaler.transform(features)
                if self.scale_targets:
                    targets = self.target_scaler.transform(targets)

                x_windows, y_windows, date_windows = self._create_sliding_windows(features, targets, dates)
                if len(x_windows) == 0:
                    continue

                prepared[f"x_{split_name}"].append(x_windows)
                prepared[f"y_{split_name}"].append(y_windows)
                prepared[f"date_{split_name}"].extend(date_windows if date_windows is not None else [])

                if self.use_static_attributes:
                    static_vector = self.static_attribute_lookup.get(watershed)
                    if static_vector is None:
                        static_vector = np.zeros(self.static_input_size, dtype=np.float32)
                    repeated = np.repeat(static_vector[np.newaxis, :], len(x_windows), axis=0)
                    prepared[f"static_{split_name}"].append(repeated.astype(np.float32))

        # Final concatenation
        for split_name in ["train", "val", "test"]:
            if prepared[f"x_{split_name}"]:
                prepared[f"x_{split_name}"] = np.concatenate(prepared[f"x_{split_name}"], axis=0).astype(np.float32)
            else:
                prepared[f"x_{split_name}"] = np.empty(
                    (0, self.window_size, len(self.feature_cols)), dtype=np.float32
                )

            if prepared[f"y_{split_name}"]:
                prepared[f"y_{split_name}"] = np.concatenate(prepared[f"y_{split_name}"], axis=0).astype(np.float32)
            else:
                empty_shape = (0, self.window_size, len(self.target_cols)) if self.many_to_many else (0, len(self.target_cols))
                prepared[f"y_{split_name}"] = np.empty(empty_shape, dtype=np.float32)

            if self.use_static_attributes:
                if prepared[f"static_{split_name}"]:
                    prepared[f"static_{split_name}"] = np.concatenate(
                        prepared[f"static_{split_name}"], axis=0
                    ).astype(np.float32)
                else:
                    prepared[f"static_{split_name}"] = np.empty((0, self.static_input_size), dtype=np.float32)
            else:
                prepared[f"static_{split_name}"] = None

        return prepared

    def _get_selection_bounds(self, scenario: str, selection: Dict) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        scenario_bounds = self.scenario_date_ranges.get(scenario)
        start = None
        end = None

        years = selection.get("years")
        if years:
            start_year, end_year = years
            start = pd.Timestamp(f"{start_year}-01-01 00:00:00")
            end = pd.Timestamp(f"{end_year}-12-31 23:59:59")
        else:
            start_value = selection.get("start") or selection.get("start_date")
            end_value = selection.get("end") or selection.get("end_date")
            if start_value:
                start = pd.Timestamp(start_value)
            if end_value:
                end = pd.Timestamp(end_value)

        if scenario_bounds:
            scenario_start, scenario_end = scenario_bounds
            if start is None or start < scenario_start:
                start = scenario_start
            if end is None or end > scenario_end:
                end = scenario_end

        return start, end

    def _filter_dataframe_by_selection(self, df: pd.DataFrame, scenario: str, selection: Dict) -> pd.DataFrame:
        start, end = self._get_selection_bounds(scenario, selection)

        if start is None and end is None:
            return df.copy()

        start = start or df.index.min()
        end = end or df.index.max()

        if start > end:
            return pd.DataFrame(columns=df.columns)

        return df.loc[start:end].copy()

    def _prepare_data_from_splits(self) -> Dict[str, np.ndarray]:
        split_cache: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
        if not self.dataset_splits:
            return self._build_prepared_from_cache(split_cache)

        for split_name, selections in self.dataset_splits.items():
            if split_name not in split_cache:
                warnings.warn(f"Unsupported split '{split_name}'. Only train/val/test are used.")
                split_cache[split_name] = []

            for selection in selections or []:
                sel_watersheds = [
                    ws for ws in selection.get("watersheds", self.available_watersheds) if ws in self.watersheds
                ]
                sel_scenarios = [
                    sc for sc in selection.get("scenarios", self.available_scenarios) if sc in self.scenarios
                ]

                if not sel_watersheds or not sel_scenarios:
                    continue

                for watershed in sel_watersheds:
                    for scenario in sel_scenarios:
                        df = self.dataset_map.get((watershed, scenario))
                        if df is None:
                            warnings.warn(f"No data found for {watershed} - {scenario}; skipping.")
                            continue

                        filtered = self._filter_dataframe_by_selection(df, scenario, selection)
                        if filtered.empty:
                            continue

                        split_cache.setdefault(split_name, []).append(
                            {
                                "watershed": watershed,
                                "scenario": scenario,
                                "features": filtered[self.feature_cols].values,
                                "targets": filtered[self.target_cols].values,
                                "dates": filtered.index,
                            }
                        )

        return self._build_prepared_from_cache(split_cache)

    def _prepare_data(self):
        if self.prepared_data is not None:
            return self.prepared_data

        self.prepared_data = self._prepare_data_from_splits()

        return self.prepared_data

    def create_data_loaders(self, shuffle_train: bool = True, num_workers: int = 0):
        data = self._prepare_data()

        self.train_dataset = GlobalTimeSeriesDataset(
            data["x_train"], data["y_train"], data["static_train"]
        )
        self.val_dataset = GlobalTimeSeriesDataset(
            data["x_val"], data["y_val"], data["static_val"]
        )
        self.test_dataset = GlobalTimeSeriesDataset(
            data["x_test"], data["y_test"], data["static_test"]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        return {
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "test_loader": self.test_loader,
        }
