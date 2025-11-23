"""
Advanced Visualization Module for Data Science Projects
Comprehensive plotting utilities using Matplotlib and Seaborn
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
from scipy import stats
import math

# Thiết lập style cho plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

@dataclass
class PlotConfig:
    """Configuration for advanced plotting"""
    figsize: Tuple[int, int] = (12, 8)
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    colormap: str = 'viridis'
    grid_alpha: float = 0.3
    style: str = 'seaborn-v0_8-whitegrid'

class AdvancedVisualizer:
    """
    Advanced visualization class with comprehensive plotting capabilities
    """
    
    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self._setup_style()
        
    def _setup_style(self):
        """Setup matplotlib style and configurations"""
        plt.style.use(self.config.style)
        sns.set_palette("husl")
        
        # Custom colormap for fraud detection
        self.fraud_cmap = LinearSegmentedColormap.from_list(
            'fraud_cmap', ['#2ecc71', '#f39c12', '#e74c3c']
        )
    
    def create_eda_dashboard(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str] = None, 
                           stats: Dict[str, Any] = None) -> plt.Figure:
        """
        Create comprehensive EDA dashboard
        """
        print("Creating Advanced EDA Dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig)
        
        # 1. Distribution of target variable
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_target_distribution(y, ax1)
        
        # 2. Feature correlation heatmap
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_correlation_heatmap(X, ax2, feature_names)
        
        # 3. Missing values heatmap
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_missing_values(X, ax3)
        
        # 4. Feature distributions (first 4 features)
        axes4 = [fig.add_subplot(gs[1, i]) for i in range(4)]
        self._plot_feature_distributions(X, y, axes4, feature_names, num_features=4)
        
        # 5. Outlier analysis
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_outlier_analysis(X, ax5, stats)
        
        # 6. Skewness and Kurtosis analysis
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_skewness_kurtosis(X, ax6)
        
        # 7. Class-wise feature distributions
        ax7 = fig.add_subplot(gs[2, 2:])
        self._plot_classwise_distributions(X, y, ax7)
        
        # 8. PCA visualization (if enough features)
        if X.shape[1] >= 2:
            ax8 = fig.add_subplot(gs[3, :2])
            self._plot_pca_projection(X, y, ax8)
        
        # 9. Feature importance (if available)
        ax9 = fig.add_subplot(gs[3, 2:])
        if stats and 'feature_importance' in stats:
            self._plot_feature_importance(stats['feature_importance'], ax9, feature_names)
        else:
            self._plot_feature_variability(X, ax9, feature_names)
        
        plt.tight_layout()
        return fig
    
    def _plot_target_distribution(self, y: np.ndarray, ax: plt.Axes):
        """Plot target variable distribution"""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        bars = ax.bar(unique, counts, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax.set_title('Target Variable Distribution', fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Add percentage labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}\n({percentage:.2f}%)', ha='center', va='bottom')
        
        # Add imbalance ratio
        if len(counts) == 2:
            imbalance_ratio = max(counts) / min(counts)
            ax.text(0.95, 0.95, f'Imbalance Ratio: {imbalance_ratio:.2f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_correlation_heatmap(self, X: np.ndarray, ax: plt.Axes, 
                                feature_names: List[str] = None, max_features: int = 15):
        """Plot correlation heatmap for features"""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Limit to top features if too many
        if X.shape[1] > max_features:
            # Select features with highest variance
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-max_features:]
            corr_matrix = corr_matrix[top_indices][:, top_indices]
            if feature_names:
                feature_names = [feature_names[i] for i in top_indices]
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={"shrink": .8},
                   annot=True, fmt='.2f', annot_kws={'size': 8})
        
        ax.set_title('Feature Correlation Heatmap', fontsize=self.config.title_fontsize, fontweight='bold')
        
        if feature_names:
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_yticklabels(feature_names, rotation=0)
    
    def _plot_missing_values(self, X: np.ndarray, ax: plt.Axes):
        """Plot missing values analysis"""
        missing_matrix = np.isnan(X)
        missing_per_feature = np.sum(missing_matrix, axis=0)
        missing_per_sample = np.sum(missing_matrix, axis=1)
        
        if np.sum(missing_per_feature) == 0:
            ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Missing Values Analysis', fontsize=self.config.title_fontsize, fontweight='bold')
            return
        
        # Plot missing values per feature
        features_with_missing = np.where(missing_per_feature > 0)[0]
        ax.bar(range(len(features_with_missing)), missing_per_feature[features_with_missing])
        ax.set_title('Missing Values per Feature', fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Missing Count')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Add percentage labels
        total_samples = X.shape[0]
        for i, idx in enumerate(features_with_missing):
            percentage = (missing_per_feature[idx] / total_samples) * 100
            ax.text(i, missing_per_feature[idx] + 0.1, f'{percentage:.1f}%', 
                   ha='center', va='bottom', fontsize=8)
    
    def _plot_feature_distributions(self, X: np.ndarray, y: np.ndarray, 
                                  axes: List[plt.Axes], feature_names: List[str] = None,
                                  num_features: int = 4):
        """Plot distributions of multiple features"""
        n_features = min(num_features, X.shape[1])
        
        for i, ax in enumerate(axes[:n_features]):
            feature_idx = i
            feature_data = X[:, feature_idx]
            
            # Remove outliers for better visualization
            q1, q3 = np.percentile(feature_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            clean_data = feature_data[(feature_data >= lower_bound) & (feature_data <= upper_bound)]
            
            ax.hist(clean_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Feature {feature_idx} Distribution', fontsize=self.config.title_fontsize-2)
            ax.set_xlabel(f'Feature {feature_idx}' if not feature_names else feature_names[feature_idx])
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=self.config.grid_alpha)
            
            # Add statistics
            mean_val = np.mean(feature_data)
            std_val = np.std(feature_data)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.2f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.6)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.6)
            ax.legend(fontsize=8)
        
        # Hide unused axes
        for ax in axes[n_features:]:
            ax.set_visible(False)
    
    def _plot_outlier_analysis(self, X: np.ndarray, ax: plt.Axes, stats: Dict[str, Any] = None):
        """Plot outlier analysis"""
        if stats is None:
            # Calculate basic statistics
            q25 = np.percentile(X, 25, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            iqr = q75 - q25
            lower_bounds = q25 - 1.5 * iqr
            upper_bounds = q75 + 1.5 * iqr
            
            outliers_per_feature = np.sum((X < lower_bounds) | (X > upper_bounds), axis=0)
        else:
            outliers_per_feature = stats.get('outlier_count', np.zeros(X.shape[1]))
        
        # Plot top features with most outliers
        top_outlier_indices = np.argsort(outliers_per_feature)[-10:][::-1]
        top_outlier_counts = outliers_per_feature[top_outlier_indices]
        
        bars = ax.bar(range(len(top_outlier_indices)), top_outlier_counts, 
                     color='coral', alpha=0.8)
        ax.set_title('Top 10 Features by Outlier Count', fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Outlier Count')
        ax.set_xticks(range(len(top_outlier_indices)))
        ax.set_xticklabels([f'Feat_{i}' for i in top_outlier_indices], rotation=45)
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Add percentage labels
        total_samples = X.shape[0]
        for bar, count in zip(bars, top_outlier_counts):
            percentage = (count / total_samples) * 100
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
    
    def _plot_skewness_kurtosis(self, X: np.ndarray, ax: plt.Axes):
        """Plot skewness vs kurtosis scatter plot"""
        skewness = []
        kurtosis = []
        
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            if np.std(feature_data) > 0:
                skew = stats.skew(feature_data)
                kurt = stats.kurtosis(feature_data)
                skewness.append(skew)
                kurtosis.append(kurt)
        
        scatter = ax.scatter(skewness, kurtosis, alpha=0.6, c=range(len(skewness)), 
                           cmap='viridis', s=50)
        ax.set_title('Skewness vs Kurtosis', fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel('Skewness')
        ax.set_ylabel('Kurtosis')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Add reference lines
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Normal Skewness')
        ax.axhline(y=0, color='blue', linestyle='--', alpha=0.5, label='Normal Kurtosis')
        ax.legend()
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Feature Index')
    
    def _plot_classwise_distributions(self, X: np.ndarray, y: np.ndarray, ax: plt.Axes):
        """Plot class-wise distributions for the most discriminative feature"""
        if len(np.unique(y)) != 2:
            ax.text(0.5, 0.5, 'Multi-class not supported', ha='center', va='center', 
                   transform=ax.transAxes)
            return
        
        # Find most discriminative feature
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        
        if len(class_1) == 0:
            ax.text(0.5, 0.5, 'No positive samples', ha='center', va='center', 
                   transform=ax.transAxes)
            return
        
        # Calculate separation score for each feature
        separation_scores = []
        for i in range(X.shape[1]):
            mean_0 = np.mean(class_0[:, i])
            mean_1 = np.mean(class_1[:, i])
            std_0 = np.std(class_0[:, i])
            std_1 = np.std(class_1[:, i])
            
            if std_0 + std_1 > 0:
                score = abs(mean_1 - mean_0) / (std_0 + std_1)
                separation_scores.append(score)
            else:
                separation_scores.append(0)
        
        best_feature = np.argmax(separation_scores)
        
        # Plot distributions
        ax.hist(class_0[:, best_feature], bins=50, alpha=0.7, label='Class 0', 
               color='blue', density=True)
        ax.hist(class_1[:, best_feature], bins=50, alpha=0.7, label='Class 1', 
               color='red', density=True)
        
        ax.set_title(f'Most Discriminative Feature (Index {best_feature})', 
                    fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel(f'Feature {best_feature} Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha)
    
    def _plot_pca_projection(self, X: np.ndarray, y: np.ndarray, ax: plt.Axes):
        """Plot 2D PCA projection"""
        from sklearn.decomposition import PCA
        
        try:
            # Standardize data
            X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            
            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_standardized)
            
            # Plot
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=self.fraud_cmap, 
                               alpha=0.6, s=30)
            ax.set_title('2D PCA Projection', fontsize=self.config.title_fontsize, fontweight='bold')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.grid(True, alpha=self.config.grid_alpha)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Class')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'PCA failed: {str(e)}', ha='center', va='center', 
                   transform=ax.transAxes)
    
    def _plot_feature_importance(self, importance: np.ndarray, ax: plt.Axes, 
                               feature_names: List[str] = None, top_k: int = 15):
        """Plot feature importance"""
        if len(importance) < top_k:
            top_k = len(importance)
        
        indices = np.argsort(importance)[-top_k:][::-1]
        sorted_importance = importance[indices]
        
        if feature_names:
            sorted_names = [feature_names[i] for i in indices]
        else:
            sorted_names = [f'Feature {i}' for i in indices]
        
        bars = ax.barh(range(top_k), sorted_importance, color='lightgreen', alpha=0.8)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Feature Importance', fontsize=self.config.title_fontsize, fontweight='bold')
        ax.grid(True, alpha=self.config.grid_alpha, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center', fontsize=8)
    
    def _plot_feature_variability(self, X: np.ndarray, ax: plt.Axes, 
                                feature_names: List[str] = None, top_k: int = 15):
        """Plot feature variability when importance is not available"""
        variances = np.var(X, axis=0)
        
        if len(variances) < top_k:
            top_k = len(variances)
        
        indices = np.argsort(variances)[-top_k:][::-1]
        sorted_variances = variances[indices]
        
        if feature_names:
            sorted_names = [feature_names[i] for i in indices]
        else:
            sorted_names = [f'Feature {i}' for i in indices]
        
        bars = ax.barh(range(top_k), sorted_variances, color='lightblue', alpha=0.8)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Variance')
        ax.set_title('Top Features by Variability', fontsize=self.config.title_fontsize, fontweight='bold')
        ax.grid(True, alpha=self.config.grid_alpha, axis='x')

    def create_model_performance_dashboard(self, model_results: Dict[str, Any], 
                                         model_names: List[str]) -> plt.Figure:
        """
        Create comprehensive model performance dashboard
        """
        print("Creating Model Performance Dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        # 1. Metrics comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_metrics_comparison(model_results, model_names, ax1)
        
        # 2. Confusion matrices
        axes2 = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
        self._plot_confusion_matrices(model_results, model_names, axes2)
        
        # 3. ROC curves
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_roc_curves(model_results, model_names, ax3)
        
        # 4. Precision-Recall curves
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_precision_recall_curves(model_results, model_names, ax4)
        
        # 5. Training history (for models that have it)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_training_history(model_results, model_names, ax5)
        
        # 6. Feature importance (if available)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_model_feature_importance(model_results, model_names, ax6)
        
        plt.tight_layout()
        return fig
    
    def _plot_metrics_comparison(self, model_results: Dict[str, Any], 
                               model_names: List[str], ax: plt.Axes):
        """Plot comparison of multiple metrics across models"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [model_results[model].get(metric, 0) for model in model_names]
            ax.bar(x + i * width, values, width, label=metric_name, alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison', fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=self.config.grid_alpha, axis='y')
        
        # Add value labels
        for i, model in enumerate(model_names):
            for j, metric in enumerate(metrics):
                value = model_results[model].get(metric, 0)
                ax.text(i + j * width, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=8)
    
    def _plot_confusion_matrices(self, model_results: Dict[str, Any], 
                               model_names: List[str], axes: List[plt.Axes]):
        """Plot confusion matrices for top models"""
        top_models = model_names[:min(2, len(model_names))]
        
        for ax, model_name in zip(axes, top_models):
            if 'confusion_matrix' in model_results[model_name]:
                cm = model_results[model_name]['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           cbar=False, square=True)
                ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=self.config.title_fontsize-2)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
            else:
                ax.text(0.5, 0.5, 'No confusion matrix\navailable', 
                       ha='center', va='center', transform=ax.transAxes)
    
    def _plot_roc_curves(self, model_results: Dict[str, Any], 
                        model_names: List[str], ax: plt.Axes):
        """Plot ROC curves for all models"""
        from sklearn.metrics import roc_curve, auc
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        for model_name in model_names:
            if 'y_true' in model_results[model_name] and 'y_pred_proba' in model_results[model_name]:
                y_true = model_results[model_name]['y_true']
                y_pred_proba = model_results[model_name]['y_pred_proba']
                
                # Handle both 1D and 2D probability arrays
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    y_score = y_pred_proba[:, 1]
                else:
                    y_score = y_pred_proba
                
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves', fontsize=self.config.title_fontsize, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=self.config.grid_alpha)
    
    def _plot_precision_recall_curves(self, model_results: Dict[str, Any], 
                                    model_names: List[str], ax: plt.Axes):
        """Plot Precision-Recall curves for all models"""
        from sklearn.metrics import precision_recall_curve, auc
        
        for model_name in model_names:
            if 'y_true' in model_results[model_name] and 'y_pred_proba' in model_results[model_name]:
                y_true = model_results[model_name]['y_true']
                y_pred_proba = model_results[model_name]['y_pred_proba']
                
                # Handle both 1D and 2D probability arrays
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    y_score = y_pred_proba[:, 1]
                else:
                    y_score = y_pred_proba
                
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                pr_auc = auc(recall, precision)
                
                ax.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})', linewidth=2)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves', fontsize=self.config.title_fontsize, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=self.config.grid_alpha)
    
    def _plot_training_history(self, model_results: Dict[str, Any], 
                             model_names: List[str], ax: plt.Axes):
        """Plot training history for models"""
        for model_name in model_names:
            if 'training_history' in model_results[model_name]:
                history = model_results[model_name]['training_history']
                if 'losses' in history and len(history['losses']) > 0:
                    losses = history['losses']
                    epochs = range(1, len(losses) + 1)
                    ax.plot(epochs, losses, label=f'{model_name} Loss', linewidth=2)
        
        if ax.has_data():
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training History', fontsize=self.config.title_fontsize, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=self.config.grid_alpha)
        else:
            ax.text(0.5, 0.5, 'No training history\navailable', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_model_feature_importance(self, model_results: Dict[str, Any], 
                                     model_names: List[str], ax: plt.Axes):
        """Plot feature importance across models"""
        importance_data = []
        model_labels = []
        
        for model_name in model_names:
            if 'feature_importance' in model_results[model_name]:
                importance = model_results[model_name]['feature_importance']
                if importance is not None and len(importance) > 0:
                    # Take absolute values and normalize
                    abs_importance = np.abs(importance)
                    normalized_importance = abs_importance / np.sum(abs_importance)
                    importance_data.append(normalized_importance)
                    model_labels.append(model_name)
        
        if importance_data:
            importance_matrix = np.array(importance_data)
            im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_yticks(range(len(model_labels)))
            ax.set_yticklabels(model_labels)
            ax.set_xlabel('Feature Index')
            ax.set_title('Feature Importance Heatmap', fontsize=self.config.title_fontsize, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Normalized Importance')
        else:
            ax.text(0.5, 0.5, 'No feature importance\navailable', 
                   ha='center', va='center', transform=ax.transAxes)

    def create_interactive_dashboard(self, X: np.ndarray, y: np.ndarray, 
                                   model_results: Dict[str, Any]) -> go.Figure:
        """
        Create interactive Plotly dashboard (optional)
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Feature Distributions', 'PCA Projection',
                              'Model Performance', 'Confusion Matrix'),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                      [{"type": "xy"}, {"type": "heatmap"}]]
            )
            
            # Feature distributions
            fig.add_trace(
                go.Histogram(x=X[:, 0], name='Feature 0', opacity=0.7),
                row=1, col=1
            )
            
            # PCA projection
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            fig.add_trace(
                go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], mode='markers',
                          marker=dict(color=y, colorscale='Viridis'),
                          name='PCA'),
                row=1, col=2
            )
            
            # Model performance
            model_names = list(model_results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            for metric in metrics:
                values = [model_results[model].get(metric, 0) for model in model_names]
                fig.add_trace(
                    go.Bar(x=model_names, y=values, name=metric),
                    row=2, col=1
                )
            
            # Confusion matrix for first model
            if model_names and 'confusion_matrix' in model_results[model_names[0]]:
                cm = model_results[model_names[0]]['confusion_matrix']
                fig.add_trace(
                    go.Heatmap(z=cm, colorscale='Blues', showscale=False),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, title_text="Interactive Model Dashboard")
            return fig
            
        except Exception as e:
            print(f"Interactive dashboard failed: {e}")
            return None

# Utility functions
def save_plot(fig: plt.Figure, filename: str, dpi: int = 300, 
              bbox_inches: str = 'tight'):
    """Save plot with consistent settings"""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, 
               facecolor='white', edgecolor='none')
    print(f"Plot saved: {filename}")

def plot_statistical_tests(X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
    """Plot statistical test results"""
    from scipy.stats import ttest_ind, mannwhitneyu
    
    if len(np.unique(y)) != 2:
        print("Statistical tests require binary classification")
        return
    
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    t_test_pvalues = []
    mw_test_pvalues = []
    
    for i in range(X.shape[1]):
        # T-test
        t_stat, t_pval = ttest_ind(class_0[:, i], class_1[:, i], equal_var=False)
        t_test_pvalues.append(t_pval)
        
        # Mann-Whitney U test
        mw_stat, mw_pval = mannwhitneyu(class_0[:, i], class_1[:, i])
        mw_test_pvalues.append(mw_pval)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # T-test p-values
    ax1.bar(range(len(t_test_pvalues)), -np.log10(t_test_pvalues))
    ax1.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('-log10(p-value)')
    ax1.set_title('T-test Significance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mann-Whitney p-values
    ax2.bar(range(len(mw_test_pvalues)), -np.log10(mw_test_pvalues))
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Mann-Whitney Test Significance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig