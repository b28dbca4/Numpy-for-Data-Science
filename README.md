# Numpy for Data Science: Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-ODbL_1.0-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

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
- **Missing Value Handling**: KNN imputation, mean/median imputation based on data characteristics.
- **Outlier Detection**: IQR, Z-score, and modified Z-score methods with robust capping/winsorization.
- **Feature Scaling**: Standard, Min-Max, Robust, and Power transformations.
- **Class Imbalance**: SMOTE oversampling and random undersampling implementations.
- **Feature Engineering**: Polynomial features, interaction terms, and automated feature selection.
- **Feature Selection**: Variance thresholding, correlation-based removal, and mutual information approximation.

### Algorithms Used

All models implemented from scratch in `src/models.py`:

#### Logistic Regression
- Mini-batch gradient descent with multiple optimizers (SGD, Momentum, Adam, RMSProp).
- L2 regularization and class weighting for imbalanced data.
- Early stopping and learning rate scheduling.

#### Neural Network (MLP)
- Multi-layer perceptron with configurable hidden layers.
- Multiple activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU).
- Batch normalization and dropout support.
- Advanced optimization with Adam and gradient clipping.

#### K-Nearest Neighbors
- Multiple distance metrics (Euclidean, Manhattan, Cosine, Minkowski).
- Batch processing for memory efficiency with large datasets.
- Weighted voting and probability estimation.

#### Evaluation Framework
- Comprehensive metrics: accuracy, precision, recall, F1-score, confusion matrix.
- Cross-validation utilities and statistical significance testing.
- ROC and Precision-Recall curve analysis.

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
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
   - Load and analyze raw data
   - Generate statistical summaries and visualizations
   - Identify data quality issues and preprocessing needs

2. **Preprocessing** (`notebooks/02_preprocessing.ipynb`):
   - Configure and apply advanced preprocessing pipeline
   - Handle missing values, outliers, and scaling
   - Perform feature engineering and selection

3. **Modeling** (`notebooks/03_modeling.ipynb`):
   - Train and evaluate multiple models
   - Compare performance metrics
   - Analyze model errors and generate insights

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

- `logistic_regression_params.npy`: Trained weights and biases
- `neural_network_params.npy`: Network parameters and configuration
- `knn_params.npy`: KNN model parameters
- `modeling_results.npy`: Comprehensive evaluation metrics
- `eda_dashboard.png`: Exploratory data analysis visualizations

**Key Findings**:
- Neural networks typically achieve the highest performance on this task
- Proper handling of class imbalance is crucial for fraud detection
- Feature engineering significantly impacts model performance

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

### Memory Efficiency
- **Challenge**: Large datasets require careful memory management.
- **Solution**: Implemented batch processing, optimized data types, and streaming operations.

### Numerical Stability
- **Challenge**: Gradient computations can become unstable with poor initialization.
- **Solution**: Xavier initialization, gradient clipping, and numerically stable implementations.

### Class Imbalance
- **Challenge**: Fraudulent transactions are extremely rare (0.172%).
- **Solution**: SMOTE oversampling, class weighting, and appropriate evaluation metrics.

### Scalability
- **Challenge**: Pure NumPy implementations may be slower than optimized libraries.
- **Solution**: Vectorized operations, efficient algorithms, and parallel processing where possible.

## Future Improvements

- **Performance Optimization**: Implement GPU acceleration with CuPy or JAX.
- **Advanced Models**: Add ensemble methods, tree-based models, and deep learning architectures.
- **Production Deployment**: Containerization with Docker and model serving APIs.
- **Automated ML**: Hyperparameter optimization and automated feature selection.
- **Real-time Processing**: Streaming data processing and online learning capabilities.

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
