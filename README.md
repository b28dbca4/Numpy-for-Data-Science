# Numpy for Data Science: Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-blue.svg)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-4.0.7-orange.svg)](https://jupyter.org/)


A comprehensive machine learning project implementing **credit card fraud detection** using pure NumPy. This educational project demonstrates the complete ML pipeline: data preprocessing, model training (Logistic Regression, Neural Networks, KNN), evaluation, and advanced visualization – all built from scratch without relying on high-level ML libraries.

## Table of Contents

- [Introduction](#introduction)
  - [Problem Statement](#problem-statement)
  - [Motivation and Real-world Applications](#motivation-and-real-world-applications)
  - [Specific Objectives](#specific-objectives)
- [Dataset](#dataset)
  - [Data Source](#data-source)
  - [Feature Description](#feature-description)
  - [Data Size and Characteristics](#data-size-and-characteristics)
- [Method](#method)
  - [Data Processing Pipeline](#data-processing-pipeline)
  - [Algorithms Used](#algorithms-used)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Challenges & Solutions](#challenges--solutions)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Authors](#authors)
- [Contact](#contact)
- [License](#license)

## Introduction

### Problem Statement

Credit card fraud detection is a critical **binary classification problem** with severe class imbalance. The goal is to identify fraudulent transactions while minimizing false positives that could inconvenience legitimate users. This project addresses the challenge of building effective fraud detection models using only NumPy, focusing on educational value and deep understanding of ML algorithms.

### Motivation and Real-world Applications

- **Financial Security**: Fraudulent transactions cost billions annually; effective detection protects consumers and institutions.
- **Real-time Processing**: Models must process transactions quickly with minimal computational resources.
- **Educational Value**: Implementing ML algorithms from scratch provides deep insights into optimization, regularization, and numerical stability.
- **Scalability**: Pure NumPy implementations can be optimized for production environments without external dependencies.

### Specific Objectives

- Implement a complete ML pipeline using **pure NumPy** for educational purposes.
- Develop robust preprocessing techniques for handling missing values, outliers, and class imbalance.
- Build and compare multiple models: Logistic Regression, Neural Networks, and KNN.
- Create comprehensive visualization tools for model analysis and interpretation.
- Demonstrate best practices in code organization, documentation, and reproducibility.

## Dataset

### Data Source

The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, provided by ULB (Université Libre de Bruxelles). This dataset contains transactions made by credit cards in September 2013 by European cardholders.

**Important**: Ensure you have proper authorization to use this dataset for research/educational purposes.

### Feature Description

- **V1-V28**: Principal components obtained via PCA transformation (anonymized features).
- **Time**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **Amount**: Transaction amount (this feature can be used for cost-sensitive learning).
- **Class**: Target variable (0 = legitimate, 1 = fraudulent).

### Data Size and Characteristics

- **284,807 transactions** over 2 days.
- **492 fraudulent transactions** (0.172% of total).
- Highly imbalanced dataset requiring specialized handling techniques.
- Features are scaled and anonymized for privacy.

## Method

### Data Processing Pipeline

Implemented in `src/data_processing.py` with two main classes:

#### NumpyDataProcessor
- **Data Loading**: Robust CSV loading with error handling and memory optimization.
- **Statistical Analysis**: Comprehensive statistics (mean, std, median, IQR, skewness, kurtosis, missing values, outliers).
- **Feature Analysis**: Distribution classification, correlation analysis, and statistical significance testing (Welch's t-test).
- **Utility Functions**: Correlation matrices, class imbalance analysis, and data validation.

#### AdvancedPreprocessor
- **Missing Value Handling**: KNN imputation (k=5), mean/median imputation based on data characteristics.
- **Outlier Detection**: IQR, Z-score (>3), and modified Z-score methods with robust capping/winsorization.
- **Feature Scaling**: RobustScaler with clipping to [-10, 10] for numerical stability in gradient computations.
- **Class Imbalance**: SMOTE and ADASYN oversampling implementations to balance fraud/normal ratio.
- **Feature Engineering**: 
  - Cyclical time encoding (Time_sin, Time_cos) to capture temporal fraud patterns
  - Polynomial features (degree 2) for non-linear relationships
  - Interaction terms between high-correlation features
- **Feature Selection**: Variance thresholding, correlation-based removal (>0.95), and mutual information approximation.

### Algorithms Used

All models implemented from scratch in `src/models.py`:

#### Logistic Regression
- Mini-batch gradient descent with multiple optimizers (SGD, Momentum, Adam, RMSProp).
- L2 regularization (λ=0.01) and class weighting for imbalanced data.
- Early stopping (patience=20) and learning rate scheduling.
- Numerical stability with logit clipping [-500, 500] and epsilon (1e-15) to prevent overflow/underflow.
- Binary cross-entropy loss with gradient updates.

#### Neural Network (MLP)
- Multi-layer perceptron with configurable hidden layers (64→32→16 neurons).
- ReLU activation (prevents vanishing gradients, sparse activation).
- Xavier initialization for stable training.
- Adam optimizer (learning_rate=0.001) with gradient clipping.
- Batch processing (batch_size=32) for memory efficiency and generalization.
- Backpropagation from scratch with chain rule implementation.

#### K-Nearest Neighbors
- Multiple distance metrics (Euclidean, Manhattan, Cosine, Minkowski).
- Batch processing for memory efficiency (uses 10K subset to avoid 313 GB RAM requirement).
- Distance-weighted voting for better probability estimation.
- Cross-validation for k selection (k ∈ {3, 5, 7, 9, 11, 13}).
- Handles curse of dimensionality through feature scaling.

#### Evaluation Framework
- Comprehensive metrics for imbalanced data: accuracy, precision, recall, F1-score, confusion matrix.
- ROC-AUC and PR-AUC (Precision-Recall curve preferred for imbalanced classification).
- K-Fold cross-validation (k=3) for robust hyperparameter selection.
- Statistical significance testing and performance comparison.
- Error analysis: false positive/negative breakdown, misclassification patterns.
- Feature importance analysis through model weights.

## Installation & Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/b28dbca4/Numpy-for-Data-Science.git
   cd Numpy-for-Data-Science
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   - Place `creditcard.csv` in the `data/raw/` directory.
   - Ensure the file follows the expected format.

## Usage

### Running the Complete Pipeline

Execute the Jupyter notebooks in order:

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`):
   - **Research Questions**: 4 guiding questions for systematic analysis
     - Q1: Statistical properties and distribution analysis
     - Q2: Feature relationships and correlation patterns
     - Q3: Class imbalance impact and fraud patterns
     - Q4: Temporal trends and transaction behavior
   - Comprehensive statistical summaries (mean, median, std, skewness, kurtosis)
   - Visual analysis: distributions, correlations, outliers
   - Data quality assessment and preprocessing recommendations

2. **Preprocessing** (`notebooks/02_preprocessing.ipynb`):
   - **Missing Value Imputation**: KNN-based intelligent imputation
   - **Outlier Treatment**: Multi-method detection (IQR, Z-score, Modified Z-score) with winsorization
   - **Feature Scaling**: RobustScaler + clipping [-10, 10] to prevent gradient explosion
   - **Class Balancing**: SMOTE/ADASYN to address 492:284,315 imbalance ratio
   - **Feature Engineering**: 
     - Cyclical time encoding (sin/cos) for temporal patterns
     - Polynomial features (degree 2) for non-linearity
     - Interaction features between correlated variables
   - **Feature Selection**: Variance thresholding, correlation removal, mutual information

3. **Modeling** (`notebooks/03_modeling.ipynb`):
   - **Model Configuration**: Tailored hyperparameters per algorithm
   - **Training**: Cross-validation with early stopping, learning curves
   - **Evaluation**: Comprehensive metrics (focus on F1, PR-AUC for imbalanced data)
   - **Comparison**: Performance, efficiency, interpretability analysis
   - **Error Analysis**: Misclassification patterns, feature importance
   - **Model Persistence**: Save/load trained models for deployment

### Training Individual Models

```python
from src.data_processing import NumpyDataProcessor
from src.models import LogisticRegression, ModelConfig, compute_metrics, train_test_split

# Load and preprocess data
processor = NumpyDataProcessor()
X, y = processor.load_data("data/raw/creditcard.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Configure and train model
config = ModelConfig(learning_rate=0.001, max_epochs=500, batch_size=256)
model = LogisticRegression(config=config, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
metrics = compute_metrics(y_test, y_pred)
print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
```

### Generating Visualizations

```python
from src.visualization import AdvancedVisualizer

visualizer = AdvancedVisualizer()
fig = visualizer.create_eda_dashboard(X, y)
fig.savefig('eda_dashboard.png', dpi=300, bbox_inches='tight')
```

## Results

Model performance and parameters are saved in `data/processed/`:

- `preprocessed_data.pkl`: Scaled, balanced, engineered features ready for training
- `logistic_regression_params.npy`: Trained weights, biases, training history
- `neural_network_params.npy`: Network parameters (W1, W2, W3, b1, b2, b3), architecture config
- `knn_params.npy`: KNN training data and hyperparameters
- `modeling_results.npy`: Comprehensive evaluation metrics and predictions

### Performance Metrics

**Key Evaluation Metrics** (on test set):
- **F1-Score**: Primary metric for imbalanced classification (balances precision/recall)
- **PR-AUC**: Precision-Recall Area Under Curve (preferred over ROC-AUC for imbalanced data)
- **Recall**: Fraud detection rate (minimize missed frauds)
- **Precision**: False alarm rate (maintain customer trust)

**Expected Performance** (after preprocessing with SMOTE/ADASYN):
- Neural Network: Highest F1-score (~0.85-0.95), best PR-AUC
- Logistic Regression: Fast training, good interpretability (~0.80-0.90 F1)
- KNN: Captures local patterns, slower prediction (~0.75-0.85 F1)

**Key Findings**:
- **Preprocessing is critical**: Feature scaling + clipping prevents gradient issues
- **Cyclical time encoding**: Significantly improves temporal fraud pattern detection
- **Class balancing**: SMOTE/ADASYN essential for learning minority class
- **Model complexity trade-off**: Neural networks achieve best performance but require more tuning
- **Feature importance**: Original PCA components (V1-V28) remain most predictive, Amount and Time features provide additional context

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # Raw dataset
│   └── processed/                   # Processed data and results
│       ├── *.npy                    # Model parameters and results
│       ├── *.pkl                    # Preprocessed data
│       └── *.png                    # Generated visualizations
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Data analysis and EDA
│   ├── 02_preprocessing.ipynb       # Data preprocessing pipeline
│   └── 03_modeling.ipynb           # Model training and evaluation
├── src/
│   ├── __init__.py
│   ├── data_processing.py           # Data loading and preprocessing
│   ├── models.py                    # ML model implementations
│   └── visualization.py             # Plotting and visualization tools
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Challenges & Solutions

### 1. Memory Efficiency
- **Challenge**: Full KNN distance matrix for 280K samples requires 313 GB RAM.
- **Solution**: Implemented batch processing and subset sampling (10K samples), optimized data types (float32), streaming operations.

### 2. Numerical Stability
- **Challenge**: Gradient computations become unstable with extreme values, log(0) and exp(overflow) errors.
- **Solution**: 
  - Clipping: Features to [-10, 10], logits to [-500, 500]
  - Epsilon addition (1e-15) in log operations
  - Xavier initialization for weights
  - Gradient clipping to prevent exploding gradients

### 3. Class Imbalance (Critical)
- **Challenge**: Fraudulent transactions are extremely rare (0.172% = 492/284,807).
- **Solution**: 
  - SMOTE and ADASYN oversampling to balance classes
  - Class weighting in loss function
  - Appropriate metrics (F1, PR-AUC instead of accuracy)
  - Threshold tuning based on business costs

### 4. Curse of Dimensionality (KNN)
- **Challenge**: In 30+ dimensions, all points become equidistant, making KNN less effective.
- **Solution**: Feature scaling to [-10, 10] range, distance-weighted voting, optimal k selection via cross-validation.

### 5. Computational Efficiency
- **Challenge**: Pure NumPy implementations slower than optimized libraries like scikit-learn.
- **Solution**: 
  - Vectorized operations (avoid Python loops)
  - Efficient algorithms (mini-batch gradient descent)
  - Memory-efficient data structures
  - Batch processing for predictions

### 6. Temporal Patterns
- **Challenge**: Linear features (Time) cannot capture cyclical fraud patterns (night vs day, weekend vs weekday).
- **Solution**: Cyclical encoding with sin/cos transformations: Time_sin = sin(2πt/86400), Time_cos = cos(2πt/86400)

## Future Improvements

### Performance Optimization
- **GPU Acceleration**: Implement with CuPy or JAX for 10-100x speedup
- **Parallel Processing**: Multi-core training with joblib/multiprocessing
- **Optimized Kernels**: Cython/Numba for critical loops

### Advanced Models
- **Ensemble Methods**: Stacking, bagging, boosting (combine LR + NN + KNN)
- **Tree-based Models**: Random Forest, Gradient Boosting from scratch
- **Deep Learning**: LSTM for temporal patterns, Autoencoders for anomaly detection
- **Attention Mechanisms**: Self-attention for feature importance

### Production Deployment
- **Containerization**: Docker images for reproducible environments
- **Model Serving**: REST API with Flask/FastAPI for real-time predictions
- **Model Monitoring**: Track performance drift, data distribution changes
- **A/B Testing**: Compare model versions in production

### Automated ML
- **Hyperparameter Optimization**: Grid search, random search, Bayesian optimization
- **Automated Feature Engineering**: Feature generation and selection pipelines
- **Neural Architecture Search**: Optimize network topology automatically

### Real-time Processing
- **Streaming Data**: Apache Kafka integration for real-time fraud detection
- **Online Learning**: Incremental model updates with new transactions
- **Low-latency Inference**: Model quantization and optimization for <100ms latency

### Enhanced Analysis
- **Explainability**: SHAP values, LIME for model interpretation
- **Adversarial Testing**: Robustness against adversarial fraud attacks
- **Threshold Optimization**: Business-cost-aware decision boundaries
- **Temporal Analysis**: Time-series patterns and seasonal trends

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings and type hints to all functions
- Write unit tests for new functionality
- Update documentation for any new features
- Ensure all code works with the existing test suite

## Authors

- **Lang Phu Quy** - *Initial work and implementation*

## Contact

For questions, suggestions, or collaboration:

- **Email**: phuquy.lang@gmail.com
- **GitHub Issues**: [issues](https://github.com/b28dbca4/Numpy-for-Data-Science/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](https://opendatacommons.org/licenses/dbcl/1-0/) file for details.

---

<div align="center">

⭐ **If you find this project helpful, please give it a star!** ⭐

*Built with ❤️ using pure NumPy*

</div>
