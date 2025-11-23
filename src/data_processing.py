import os
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Union, Dict, Any
import warnings
import time
from dataclasses import dataclass

@dataclass
class DataStats:
    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]
    median: npt.NDArray[np.float64]
    min: npt.NDArray[np.float64]
    max: npt.NDArray[np.float64]
    q25: npt.NDArray[np.float64]
    q75: npt.NDArray[np.float64]
    iqr: npt.NDArray[np.float64]
    skewness: npt.NDArray[np.float64]
    kurtosis: npt.NDArray[np.float64]
    missing_count: npt.NDArray[np.int64]
    outlier_count: npt.NDArray[np.int64]

class NumpyDataProcessor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        self._fitted = False
        self._mean = None
        self._std = None
        self._min = None
        self._max = None
        self._feature_names = None
        
    def load_data(self, filepath: str, delimiter: str = ',', 
                  skip_header: int = 1, target_col: int = -1) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Load data with advanced error handling and memory optimization
        
        Parameters:
        -----------
        filepath : str
            Path to data file
        delimiter : str
            Column delimiter
        skip_header : int
            Number of header rows to skip
        target_col : int
            Index of target column
            
        Returns:
        --------
        X : ndarray
            Feature matrix
        y : ndarray
            Target vector
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            start_time = time.time()

            # Determine number of columns using first data row
            with open(filepath, "r") as f:
                for _ in range(skip_header):
                    next(f)
                first_line = next(f).strip()
            n_cols = len(first_line.split(delimiter))

            # Resolve negative index for target column
            if target_col < 0:
                target_col = n_cols + target_col
            if target_col < 0 or target_col >= n_cols:
                raise ValueError(f"Invalid target column index {target_col} for {n_cols} columns")

            feature_cols = [i for i in range(n_cols) if i != target_col]

            # Load features (float) and target (as string then to int) separately to avoid structured dtypes
            X = np.genfromtxt(
                filepath,
                delimiter=delimiter,
                skip_header=skip_header,
                usecols=feature_cols,
                dtype=np.float64,
            )

            y_raw = np.genfromtxt(
                filepath,
                delimiter=delimiter,
                skip_header=skip_header,
                usecols=target_col,
                dtype=str,
            )
            y = np.array([int(label.strip('"')) for label in np.atleast_1d(y_raw)], dtype=np.int8)

            if X.ndim != 2:
                raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}")
            if y.ndim != 1:
                raise ValueError(f"Expected 1D target vector, got shape {y.shape}")
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Features and target have mismatched samples: {X.shape[0]} vs {y.shape[0]}"
                )

            print(f"Data loaded in {time.time() - start_time:.2f} seconds")
            print(f"Features shape: {X.shape}, Target shape: {y.shape}")
            print(f"Memory usage: {X.nbytes / 1024 / 1024:.2f} MB")

            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            return X, y

        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _optimize_dtypes(self, sample_data: npt.NDArray) -> np.dtype:
        """Optimize data types for memory efficiency"""
        dtypes = []
        
        for col in range(sample_data.shape[1]):
            col_data = sample_data[:, col]
            
            # Handle NaN values
            valid_mask = ~np.isnan(col_data)
            if not np.any(valid_mask):
                dtypes.append(np.float32)
                continue
                
            col_data_clean = col_data[valid_mask]
            
            # Check if integer type is sufficient
            if np.all(col_data_clean == col_data_clean.astype(np.int32)):
                min_val, max_val = np.min(col_data_clean), np.max(col_data_clean)
                
                if min_val >= 0:  # Unsigned
                    if max_val <= 255:
                        dtypes.append(np.uint8)
                    elif max_val <= 65535:
                        dtypes.append(np.uint16)
                    else:
                        dtypes.append(np.uint32)
                else:  # Signed
                    if min_val >= -128 and max_val <= 127:
                        dtypes.append(np.int8)
                    elif min_val >= -32768 and max_val <= 32767:
                        dtypes.append(np.int16)
                    else:
                        dtypes.append(np.int32)
            else:
                # Use float32 for real numbers
                dtypes.append(np.float32)
                
        return np.dtype([(f'col_{i}', dtypes[i]) for i in range(len(dtypes))])
    
    def compute_comprehensive_stats(self, X: npt.NDArray) -> DataStats:
        """
        Compute comprehensive statistical profile of the dataset
        """
        print("Computing comprehensive data statistics...")
        
        # Basic statistics
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        median = np.nanmedian(X, axis=0)
        min_val = np.nanmin(X, axis=0)
        max_val = np.nanmax(X, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        iqr = q75 - q25
        
        # Advanced statistics
        skewness = self._compute_skewness(X, mean, std)
        kurtosis = self._compute_kurtosis(X, mean, std)
        
        # Missing values and outliers
        missing_count = np.sum(np.isnan(X), axis=0)
        outlier_count = self._detect_outliers(X, q25, q75, iqr)
        
        return DataStats(
            mean=mean, std=std, median=median, min=min_val, max=max_val,
            q25=q25, q75=q75, iqr=iqr, skewness=skewness, kurtosis=kurtosis,
            missing_count=missing_count, outlier_count=outlier_count
        )
    
    def _compute_skewness(self, X: npt.NDArray, mean: npt.NDArray, 
                         std: npt.NDArray) -> npt.NDArray:
        """Compute Fisher-Pearson coefficient of skewness"""
        n = X.shape[0]
        # Avoid division by zero
        std_safe = np.where(std == 0, 1e-10, std)
        skew = (np.nansum(((X - mean) / std_safe) ** 3, axis=0) * n / 
               ((n - 1) * (n - 2)))
        return skew
    
    def _compute_kurtosis(self, X: npt.NDArray, mean: npt.NDArray, 
                         std: npt.NDArray) -> npt.NDArray:
        """Compute Fisher's kurtosis (excess kurtosis)"""
        n = X.shape[0]
        std_safe = np.where(std == 0, 1e-10, std)
        kurt = (np.nansum(((X - mean) / std_safe) ** 4, axis=0) * n * (n + 1) / 
               ((n - 1) * (n - 2) * (n - 3)) - 3 * (n - 1) ** 2 / 
               ((n - 2) * (n - 3)))
        return kurt
    
    def _detect_outliers(self, X: npt.NDArray, q25: npt.NDArray, 
                        q75: npt.NDArray, iqr: npt.NDArray) -> npt.NDArray:
        """Detect outliers using IQR method"""
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = np.sum((X < lower_bound) | (X > upper_bound), axis=0)
        return outliers
    
    def analyze_feature_distributions(self, X: npt.NDArray, 
                                    stats: DataStats) -> Dict[str, Any]:
        """
        Advanced analysis of feature distributions
        """
        analysis = {}
        
        for i in range(X.shape[1]):
            feature_analysis = {
                'distribution_type': self._classify_distribution(stats.skewness[i], 
                                                               stats.kurtosis[i]),
                'outlier_percentage': (stats.outlier_count[i] / X.shape[0]) * 100,
                'missing_percentage': (stats.missing_count[i] / X.shape[0]) * 100,
                'variation_coefficient': stats.std[i] / stats.mean[i] if stats.mean[i] != 0 else 0,
                'is_constant': stats.std[i] == 0,
                'is_highly_skewed': abs(stats.skewness[i]) > 2,
                'has_heavy_tails': abs(stats.kurtosis[i]) > 3.5
            }
            analysis[f'feature_{i}'] = feature_analysis
            
        return analysis
    
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution type based on skewness and kurtosis"""
        if abs(skewness) < 0.5:
            if abs(kurtosis) < 1:
                return "Approximately Normal"
            elif kurtosis > 1:
                return "Light-tailed (Platykurtic)"
            else:
                return "Heavy-tailed (Leptokurtic)"
        elif skewness > 0.5:
            return "Right-skewed"
        else:
            return "Left-skewed"
        
    def perform_ttest(self, X: npt.NDArray, y: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Thực hiện Welch's t-test để so sánh sự khác biệt trung bình giữa hai lớp.
        H0: Không có sự khác biệt đáng kể về trung bình của đặc trưng giữa 2 lớp.
        H1: Có sự khác biệt đáng kể.
        
        Parameters:
        -----------
        X : ndarray
            Ma trận đặc trưng
        y : ndarray
            Vector nhãn (0 và 1)
            
        Returns:
        --------
        t_stats : ndarray
            Giá trị t-statistic cho mỗi feature.
        significant_features : ndarray
            Chỉ số (index) của các features có ý nghĩa thống kê (abs(t) > 1.96).
        """
        print("Performing Welch's t-test analysis...")
        
        # Tách dữ liệu theo lớp
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]
        
        # Tính các thống kê cần thiết
        n0 = X_class0.shape[0]
        n1 = X_class1.shape[0]
        
        # Tính mean và variance cho từng feature (axis=0)
        mean0 = np.mean(X_class0, axis=0)
        mean1 = np.mean(X_class1, axis=0)
        
        # ddof=1 để tính phương sai mẫu (sample variance)
        var0 = np.var(X_class0, axis=0, ddof=1)
        var1 = np.var(X_class1, axis=0, ddof=1)
        
        # Tránh chia cho 0
        epsilon = 1e-10
        
        # Công thức Welch's t-statistic
        # t = (mean1 - mean0) / sqrt(var1/n1 + var0/n0)
        numerator = mean1 - mean0
        denominator = np.sqrt((var1 / n1) + (var0 / n0) + epsilon)
        
        t_stats = numerator / denominator
        
        # Xác định các feature quan trọng
        # Ngưỡng 1.96 tương ứng với p-value < 0.05 (độ tin cậy 95%) cho phân phối chuẩn
        significant_mask = np.abs(t_stats) > 1.96
        significant_features = np.where(significant_mask)[0]
        
        print(f"Found {len(significant_features)} significant features (p < 0.05)")
        
        return t_stats, significant_features
        

# Utility functions for advanced analysis
def load_credit_card_data(filepath: str) -> Tuple[npt.NDArray, npt.NDArray]:
    """Convenience wrapper for loading the credit card fraud dataset."""

    processor = NumpyDataProcessor()
    return processor.load_data(filepath, delimiter=",", skip_header=1, target_col=-1)


def load_data_robust(
    filepath: str,
    delimiter: str = ",",
    skip_header: int = 1,
    target_col: int = -1,
    dtype_features: np.dtype = np.float64,
    dtype_target: np.dtype = np.int32,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Standalone loader mirroring NumpyDataProcessor logic for scripts/tests."""

    processor = NumpyDataProcessor()
    return processor.load_data(
        filepath=filepath,
        delimiter=delimiter,
        skip_header=skip_header,
        target_col=target_col,
    )


def compute_correlation_matrix(X: npt.NDArray, 
                             method: str = 'pearson') -> npt.NDArray:
    """
    Compute correlation matrix with different methods
    """
    if method == 'pearson':
        # Using efficient matrix operations for Pearson correlation
        X_clean = np.nan_to_num(X, nan=0.0)
        X_normalized = X_clean - np.mean(X_clean, axis=0)
        cov = np.dot(X_normalized.T, X_normalized) / (X.shape[0] - 1)
        std_devs = np.std(X_clean, axis=0)
        std_devs = np.where(std_devs == 0, 1e-10, std_devs)
        corr = cov / np.outer(std_devs, std_devs)
        return np.clip(corr, -1, 1)
    
    elif method == 'spearman':
        # Spearman's rank correlation
        ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, X)
        return compute_correlation_matrix(ranks, 'pearson')

def detect_highly_correlated_features(corr_matrix: npt.NDArray, 
                                    threshold: float = 0.8) -> list:
    """
    Detect pairs of highly correlated features
    """
    highly_correlated = []
    n_features = corr_matrix.shape[0]
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                highly_correlated.append((i, j, corr_matrix[i, j]))
                
    return sorted(highly_correlated, key=lambda x: abs(x[2]), reverse=True)

def analyze_class_imbalance(y: npt.NDArray) -> Dict[str, float]:
    """
    Comprehensive class imbalance analysis
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    analysis = {
        'class_distribution': dict(zip(unique, counts)),
        'imbalance_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf'),
        'minority_class_percentage': (min(counts) / total) * 100,
        'majority_class_percentage': (max(counts) / total) * 100
    }
    
    return analysis

"""
Advanced Data Processing Module - Extended for Preprocessing
Implementing sophisticated preprocessing techniques using pure NumPy
"""

@dataclass
class PreprocessingConfig:
    """Configuration for advanced preprocessing pipeline"""
    # Missing values handling
    missing_strategy: str = 'advanced_imputation'  # 'mean', 'median', 'knn', 'advanced_imputation'
    knn_k: int = 5
    
    # Outlier handling
    outlier_strategy: str = 'robust_capping'  # 'remove', 'capping', 'robust_capping', 'isolation'
    outlier_threshold: float = 3.0
    
    # Scaling
    scaling_method: str = 'robust'  # 'standard', 'minmax', 'robust', 'power'
    
    # Feature engineering
    create_polynomial: bool = True
    polynomial_degree: int = 2
    create_interactions: bool = True
    interaction_depth: int = 2
    
    # Feature selection
    feature_selection: bool = True
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    
    # Class imbalance
    imbalance_handler: str = 'smote'  # 'smote', 'adasyn', 'undersampling', 'class_weight'
    
    # Advanced options
    use_quantile_transformation: bool = False
    use_box_cox: bool = False
    use_yeo_johnson: bool = False

class AdvancedPreprocessor:
    """
    Advanced Preprocessor with sophisticated NumPy implementations
    Implements state-of-the-art preprocessing techniques from scratch
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self._fitted = False
        self._imputation_values = None
        self._scaling_params = {}
        self._feature_selector = None
        self._transformation_params = {}
        
    def advanced_missing_value_imputation(self, X: npt.NDArray) -> npt.NDArray:
        """
        Advanced missing value imputation using multiple strategies
        """
        print("Performing advanced missing value imputation...")
        
        if np.isnan(X).sum() == 0:
            print("No missing values found.")
            return X
            
        X_imputed = X.copy()
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            missing_mask = np.isnan(feature_data)
            
            if not np.any(missing_mask):
                continue
                
            valid_data = feature_data[~missing_mask]
            
            if self.config.missing_strategy == 'knn':
                # Advanced KNN imputation
                imputed_values = self._knn_imputation(X, feature_idx, missing_mask, k=self.config.knn_k)
                X_imputed[missing_mask, feature_idx] = imputed_values
                
            elif self.config.missing_strategy == 'advanced_imputation':
                # Multi-strategy approach based on data characteristics
                if self._is_categorical(valid_data):
                    # Mode for categorical
                    imputed_value = self._mode(valid_data)
                elif self._is_highly_skewed(valid_data):
                    # Median for skewed data
                    imputed_value = np.median(valid_data)
                else:
                    # Weighted mean considering outliers
                    imputed_value = self._robust_mean(valid_data)
                    
                X_imputed[missing_mask, feature_idx] = imputed_value
                
            else:
                # Basic strategies
                if self.config.missing_strategy == 'mean':
                    imputed_value = np.mean(valid_data)
                elif self.config.missing_strategy == 'median':
                    imputed_value = np.median(valid_data)
                    
                X_imputed[missing_mask, feature_idx] = imputed_value
                
        print(f"Imputed {np.isnan(X).sum()} missing values")
        return X_imputed
    
    def _knn_imputation(self, X: npt.NDArray, target_feature: int, 
                       missing_mask: npt.NDArray, k: int = 5) -> npt.NDArray:
        """Advanced KNN imputation with feature weighting"""
        # Use other features to find similar instances
        other_features = [i for i in range(X.shape[1]) if i != target_feature]
        X_other = X[:, other_features]
        
        # Handle missing values in other features temporarily
        X_other_filled = np.nan_to_num(X_other, nan=0.0)
        
        # Find k nearest neighbors for each missing sample
        imputed_values = []
        
        for sample_idx in np.where(missing_mask)[0]:
            sample = X_other_filled[sample_idx]
            
            # Calculate distances to all samples with valid target feature
            valid_mask = ~np.isnan(X[:, target_feature])
            valid_samples = X_other_filled[valid_mask]
            valid_targets = X[valid_mask, target_feature]
            
            if len(valid_samples) == 0:
                # Fallback to mean if no valid samples
                imputed_values.append(np.nanmean(X[:, target_feature]))
                continue
                
            # Weighted Euclidean distance
            distances = np.sqrt(np.sum((valid_samples - sample) ** 2, axis=1))
            
            # Find k nearest neighbors
            if len(distances) < k:
                k_actual = len(distances)
            else:
                k_actual = k
                
            nearest_indices = np.argpartition(distances, k_actual)[:k_actual]
            nearest_weights = 1.0 / (distances[nearest_indices] + 1e-10)
            
            # Weighted average of neighbors
            weighted_avg = np.average(valid_targets[nearest_indices], weights=nearest_weights)
            imputed_values.append(weighted_avg)
            
        return np.array(imputed_values)
    
    def robust_outlier_handling(self, X: npt.NDArray, 
                              stats: DataStats) -> Tuple[npt.NDArray, Dict[str, Any]]:
        """
        Advanced outlier detection and handling using multiple methods
        """
        print("Performing robust outlier handling...")
        
        X_processed = X.copy()
        outlier_report = {
            'total_outliers': 0,
            'outliers_by_feature': {},
            'treatment_applied': {}
        }
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            
            # Multiple outlier detection methods
            iqr_outliers = self._detect_iqr_outliers(feature_data, stats.q25[feature_idx], 
                                                   stats.q75[feature_idx], stats.iqr[feature_idx])
            zscore_outliers = self._detect_zscore_outliers(feature_data, stats.mean[feature_idx], 
                                                         stats.std[feature_idx])
            modified_zscore_outliers = self._detect_modified_zscore_outliers(feature_data)
            
            # Combine detection methods
            combined_outliers = iqr_outliers | zscore_outliers | modified_zscore_outliers
            n_outliers = np.sum(combined_outliers)
            
            outlier_report['outliers_by_feature'][f'feature_{feature_idx}'] = n_outliers
            outlier_report['total_outliers'] += n_outliers
            
            if n_outliers > 0:
                if self.config.outlier_strategy == 'robust_capping':
                    # Winsorization - cap outliers at specified percentiles
                    lower_bound = np.percentile(feature_data, 1)
                    upper_bound = np.percentile(feature_data, 99)
                    
                    X_processed[combined_outliers & (feature_data < lower_bound), feature_idx] = lower_bound
                    X_processed[combined_outliers & (feature_data > upper_bound), feature_idx] = upper_bound
                    
                    outlier_report['treatment_applied'][f'feature_{feature_idx}'] = 'winsorization'
                    
                elif self.config.outlier_strategy == 'capping':
                    # IQR-based capping
                    lower_bound = stats.q25[feature_idx] - 1.5 * stats.iqr[feature_idx]
                    upper_bound = stats.q75[feature_idx] + 1.5 * stats.iqr[feature_idx]
                    
                    X_processed[feature_data < lower_bound, feature_idx] = lower_bound
                    X_processed[feature_data > upper_bound, feature_idx] = upper_bound
                    
                    outlier_report['treatment_applied'][f'feature_{feature_idx}'] = 'iqr_capping'
                    
        print(f"Treated {outlier_report['total_outliers']} outliers using {self.config.outlier_strategy}")
        return X_processed, outlier_report
    
    def _detect_iqr_outliers(self, data: npt.NDArray, q25: float, q75: float, iqr: float) -> npt.NDArray:
        """Detect outliers using IQR method"""
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        return (data < lower_bound) | (data > upper_bound)
    
    def _detect_zscore_outliers(self, data: npt.NDArray, mean: float, std: float) -> npt.NDArray:
        """Detect outliers using Z-score method"""
        if std == 0:
            return np.zeros_like(data, dtype=bool)
        z_scores = np.abs((data - mean) / std)
        return z_scores > self.config.outlier_threshold
    
    def _detect_modified_zscore_outliers(self, data: npt.NDArray) -> npt.NDArray:
        """Detect outliers using modified Z-score (more robust)"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return np.zeros_like(data, dtype=bool)
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > self.config.outlier_threshold
    
    def advanced_feature_scaling(self, X: npt.NDArray, stats: DataStats) -> npt.NDArray:
        """
        Advanced feature scaling with multiple methods and automatic selection
        """
        print("Performing advanced feature scaling...")
        
        X_scaled = X.copy()
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            
            # Choose scaling method based on data characteristics
            if self.config.scaling_method == 'auto':
                # Automatic method selection
                if self._is_normal_distributed(feature_data):
                    scaling_method = 'standard'
                elif self._has_outliers(feature_data, stats, feature_idx):
                    scaling_method = 'robust'
                else:
                    scaling_method = 'minmax'
            else:
                scaling_method = self.config.scaling_method
            
            # Apply chosen scaling method
            if scaling_method == 'standard':
                scaled_data = (feature_data - stats.mean[feature_idx]) / stats.std[feature_idx]
                
            elif scaling_method == 'minmax':
                min_val, max_val = stats.min[feature_idx], stats.max[feature_idx]
                if max_val - min_val == 0:
                    scaled_data = np.zeros_like(feature_data)
                else:
                    scaled_data = (feature_data - min_val) / (max_val - min_val)
                    
            elif scaling_method == 'robust':
                median = stats.median[feature_idx]
                q75, q25 = stats.q75[feature_idx], stats.q25[feature_idx]
                iqr = q75 - q25
                if iqr == 0:
                    scaled_data = np.zeros_like(feature_data)
                else:
                    scaled_data = (feature_data - median) / iqr
                    
            elif scaling_method == 'power':
                # Yeo-Johnson transformation
                scaled_data = self._yeo_johnson_transform(feature_data)
                
            X_scaled[:, feature_idx] = scaled_data
            
        return X_scaled
    
    def _yeo_johnson_transform(self, data: npt.NDArray) -> npt.NDArray:
        """Yeo-Johnson power transformation"""
        # Simplified implementation
        positive_mask = data >= 0
        negative_mask = data < 0
        
        transformed = np.zeros_like(data)
        
        # For positive values
        if np.any(positive_mask):
            pos_data = data[positive_mask]
            if np.std(pos_data) > 0:
                # Find optimal lambda (simplified)
                lambda_pos = 0.5  # In practice, this should be optimized
                if abs(lambda_pos) < 1e-10:
                    transformed[positive_mask] = np.log1p(pos_data)
                else:
                    transformed[positive_mask] = (np.power(pos_data + 1, lambda_pos) - 1) / lambda_pos
        
        # For negative values  
        if np.any(negative_mask):
            neg_data = data[negative_mask]
            if np.std(neg_data) > 0:
                lambda_neg = 2.0  # Simplified
                if abs(lambda_neg - 2) < 1e-10:
                    transformed[negative_mask] = -np.log1p(-neg_data)
                else:
                    transformed[negative_mask] = -(np.power(-neg_data + 1, 2 - lambda_neg) - 1) / (2 - lambda_neg)
                    
        return transformed
    
    def handle_class_imbalance(self, X: npt.NDArray, y: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Advanced class imbalance handling using SMOTE implementation
        """
        print("Handling class imbalance...")
        
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        if len(unique_classes) != 2:
            print("Multi-class problem detected, returning original data")
            return X, y
            
        minority_class = unique_classes[np.argmin(class_counts)]
        majority_class = unique_classes[np.argmax(class_counts)]
        
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        minority_count = len(minority_indices)
        majority_count = len(majority_indices)
        
        print(f"Before balancing - Minority: {minority_count}, Majority: {majority_count}")
        
        if self.config.imbalance_handler == 'smote':
            X_balanced, y_balanced = self._smote_oversampling(X, y, minority_class, k=5)
        elif self.config.imbalance_handler == 'undersampling':
            X_balanced, y_balanced = self._random_undersampling(X, y, majority_class, minority_count)
        else:
            print("No balancing applied")
            return X, y
            
        print(f"After balancing - Minority: {np.sum(y_balanced == minority_class)}, "
              f"Majority: {np.sum(y_balanced == majority_class)}")
              
        return X_balanced, y_balanced
    
    def _smote_oversampling(self, X: npt.NDArray, y: npt.NDArray, 
                          minority_class: int, k: int = 5) -> Tuple[npt.NDArray, npt.NDArray]:
        """SMOTE implementation from scratch"""
        minority_indices = np.where(y == minority_class)[0]
        X_minority = X[minority_indices]
        
        n_minority = len(minority_indices)
        n_synthetic = len(y) - 2 * n_minority  # Balance to 50-50
        
        if n_synthetic <= 0:
            return X, y
            
        # Find k nearest neighbors within minority class
        synthetic_samples = []
        
        for i in range(n_synthetic):
            # Randomly select a minority sample
            idx = np.random.randint(0, n_minority)
            sample = X_minority[idx]
            
            # Find k nearest neighbors
            distances = np.sqrt(np.sum((X_minority - sample) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[1:k+1]  # Exclude itself
            
            # Randomly select one neighbor
            neighbor_idx = np.random.choice(nearest_indices)
            neighbor = X_minority[neighbor_idx]
            
            # Generate synthetic sample
            gap = np.random.random()
            synthetic = sample + gap * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        # Combine original and synthetic samples
        X_balanced = np.vstack([X, synthetic_samples])
        y_balanced = np.hstack([y, np.full(len(synthetic_samples), minority_class)])
        
        return X_balanced, y_balanced
    
    def _random_undersampling(self, X: npt.NDArray, y: npt.NDArray,
                            majority_class: int, target_count: int) -> Tuple[npt.NDArray, npt.NDArray]:
        """Random undersampling implementation"""
        majority_indices = np.where(y == majority_class)[0]
        
        if len(majority_indices) <= target_count:
            return X, y
            
        # Randomly select subset of majority class
        selected_indices = np.random.choice(majority_indices, target_count, replace=False)
        
        # Combine with all minority samples
        minority_indices = np.where(y != majority_class)[0]
        balanced_indices = np.concatenate([minority_indices, selected_indices])
        
        return X[balanced_indices], y[balanced_indices]
    
    def advanced_feature_engineering(self, X: npt.NDArray) -> npt.NDArray:
        """
        Advanced feature engineering with polynomial features and interactions
        """
        print("Performing advanced feature engineering...")
        
        engineered_features = [X]
        
        if self.config.create_polynomial:
            poly_features = self._create_polynomial_features(X, degree=self.config.polynomial_degree)
            engineered_features.append(poly_features)
            
        if self.config.create_interactions:
            interaction_features = self._create_interaction_features(X, depth=self.config.interaction_depth)
            engineered_features.append(interaction_features)
            
        X_engineered = np.hstack(engineered_features)
        print(f"Feature engineering: {X.shape[1]} -> {X_engineered.shape[1]} features")
        
        return X_engineered
    
    def _create_polynomial_features(self, X: npt.NDArray, degree: int = 2) -> npt.NDArray:
        """Create polynomial features"""
        poly_features = []
        n_features = X.shape[1]
        
        for deg in range(2, degree + 1):
            for i in range(n_features):
                poly_feature = X[:, i] ** deg
                poly_features.append(poly_feature.reshape(-1, 1))
                
        return np.hstack(poly_features) if poly_features else np.empty((X.shape[0], 0))
    
    def _create_interaction_features(self, X: npt.NDArray, depth: int = 2) -> npt.NDArray:
        """Create interaction features between variables"""
        interaction_features = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            for j in range(i + 1, min(i + depth + 1, n_features)):
                interaction = X[:, i] * X[:, j]
                interaction_features.append(interaction.reshape(-1, 1))
                
        return np.hstack(interaction_features) if interaction_features else np.empty((X.shape[0], 0))
    
    def intelligent_feature_selection(self, X: npt.NDArray, y: npt.NDArray, 
                                    corr_matrix: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Intelligent feature selection using multiple criteria
        """
        print("Performing intelligent feature selection...")
        
        if not self.config.feature_selection:
            return X, np.arange(X.shape[1])
            
        selected_features = []
        n_features = X.shape[1]
        
        # 1. Remove low variance features
        variances = np.var(X, axis=0)
        high_variance_mask = variances > self.config.variance_threshold
        selected_features.extend(np.where(high_variance_mask)[0])
        
        # 2. Remove highly correlated features
        correlation_mask = self._remove_highly_correlated(corr_matrix, self.config.correlation_threshold)
        selected_features.extend(np.where(correlation_mask)[0])
        
        # 3. Select features based on mutual information (simplified)
        mi_scores = self._calculate_mutual_info(X, y)
        high_mi_features = np.where(mi_scores > np.median(mi_scores))[0]
        selected_features.extend(high_mi_features)
        
        # Get unique features
        selected_features = np.unique(selected_features)
        
        print(f"Feature selection: {n_features} -> {len(selected_features)} features")
        
        return X[:, selected_features], selected_features
    
    def _remove_highly_correlated(self, corr_matrix: npt.NDArray, threshold: float) -> npt.NDArray:
        """Remove highly correlated features"""
        n_features = corr_matrix.shape[0]
        to_keep = np.ones(n_features, dtype=bool)
        
        for i in range(n_features):
            if to_keep[i]:
                # Find features highly correlated with feature i
                highly_correlated = np.where(np.abs(corr_matrix[i, :]) > threshold)[0]
                # Keep only the first one, remove others
                to_keep[highly_correlated] = False
                to_keep[i] = True
                
        return to_keep
    
    def _calculate_mutual_info(self, X: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
        """Calculate mutual information between features and target (simplified)"""
        # Simplified implementation using correlation as proxy
        n_features = X.shape[1]
        mi_scores = np.zeros(n_features)
        
        for i in range(n_features):
            if np.std(X[:, i]) > 0:
                # Use absolute correlation as proxy for mutual information
                correlation = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                mi_scores[i] = correlation if not np.isnan(correlation) else 0
                
        return mi_scores
    
    # Utility methods
    def _is_categorical(self, data: npt.NDArray) -> bool:
        """Check if data is categorical"""
        unique_ratio = len(np.unique(data)) / len(data)
        return unique_ratio < 0.05  # Less than 5% unique values
    
    def _is_highly_skewed(self, data: npt.NDArray) -> bool:
        """Check if data is highly skewed"""
        if np.std(data) == 0:
            return False
        skewness = np.mean((data - np.mean(data)) ** 3) / (np.std(data) ** 3)
        return abs(skewness) > 1
    
    def _robust_mean(self, data: npt.NDArray) -> float:
        """Calculate robust mean (trimmed mean)"""
        q10, q90 = np.percentile(data, [10, 90])
        trimmed_data = data[(data >= q10) & (data <= q90)]
        return np.mean(trimmed_data) if len(trimmed_data) > 0 else np.mean(data)
    
    def _is_normal_distributed(self, data: npt.NDArray) -> bool:
        """Check if data is normally distributed (simplified)"""
        if np.std(data) == 0:
            return False
        skewness = np.mean((data - np.mean(data)) ** 3) / (np.std(data) ** 3)
        return abs(skewness) < 0.5
    
    def _has_outliers(self, data: npt.NDArray, stats: DataStats, feature_idx: int) -> bool:
        """Check if data has significant outliers"""
        return stats.outlier_count[feature_idx] / len(data) > 0.01  # More than 1% outliers
    
    def _mode(self, data: npt.NDArray) -> float:
        """Calculate mode of data"""
        values, counts = np.unique(data, return_counts=True)
        return values[np.argmax(counts)]