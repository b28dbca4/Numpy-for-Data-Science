import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Union, Dict, Any, List, Callable
from dataclasses import dataclass
import time
from abc import ABC, abstractmethod
from scipy.special import expit, logsumexp
import warnings

@dataclass
class ModelConfig:
    """Configuration for model training"""
    learning_rate: float = 0.01
    max_epochs: int = 1000
    batch_size: int = 32
    patience: int = 10
    tolerance: float = 1e-6
    regularization: float = 0.01
    optimizer: str = 'adam'  # 'sgd', 'momentum', 'adam', 'rmsprop'
    early_stopping: bool = True
    verbose: bool = True

@dataclass
class TrainingHistory:
    """Track training progress"""
    losses: List[float]
    accuracies: List[float]
    gradients_norm: List[float]
    learning_rates: List[float]
    timestamps: List[float]

class BaseModel(ABC):
    """Base class for all models with common functionality"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.history = TrainingHistory([], [], [], [], [])
        self.weights = None
        self.bias = None
        self._fitted = False
        self.best_weights = None
        self.best_bias = None
        
    @abstractmethod
    def _initialize_parameters(self, n_features: int):
        """Initialize model parameters"""
        pass
    
    @abstractmethod
    def _forward_pass(self, X: npt.NDArray) -> npt.NDArray:
        """Forward pass through the model"""
        pass
    
    @abstractmethod
    def _compute_loss(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
        """Compute loss function"""
        pass
    
    @abstractmethod
    def _backward_pass(self, X: npt.NDArray, y_true: npt.NDArray, y_pred: npt.NDArray) -> Tuple[npt.NDArray, float]:
        """Backward pass for gradients"""
        pass
    
    def _update_parameters(self, gradients: Dict[str, npt.NDArray], epoch: int):
        """Update parameters using optimizer"""
        if self.config.optimizer == 'adam':
            self._adam_update(gradients, epoch)
        elif self.config.optimizer == 'momentum':
            self._momentum_update(gradients, epoch)
        elif self.config.optimizer == 'rmsprop':
            self._rmsprop_update(gradients, epoch)
        else:  # sgd
            self._sgd_update(gradients)
    
    def _sgd_update(self, gradients: Dict[str, npt.NDArray]):
        """Stochastic Gradient Descent"""
        self.weights -= self.config.learning_rate * gradients['weights']
        self.bias -= self.config.learning_rate * gradients['bias']
    
    def _momentum_update(self, gradients: Dict[str, npt.NDArray], epoch: int):
        """Momentum optimizer"""
        if not hasattr(self, 'velocity_w'):
            self.velocity_w = np.zeros_like(self.weights)
            self.velocity_b = np.zeros_like(self.bias)
        
        beta = 0.9
        self.velocity_w = beta * self.velocity_w + (1 - beta) * gradients['weights']
        self.velocity_b = beta * self.velocity_b + (1 - beta) * gradients['bias']
        
        self.weights -= self.config.learning_rate * self.velocity_w
        self.bias -= self.config.learning_rate * self.velocity_b
    
    def _rmsprop_update(self, gradients: Dict[str, npt.NDArray], epoch: int):
        """RMSProp optimizer"""
        if not hasattr(self, 'cache_w'):
            self.cache_w = np.zeros_like(self.weights)
            self.cache_b = np.zeros_like(self.bias)
        
        beta = 0.9
        epsilon = 1e-8
        
        self.cache_w = beta * self.cache_w + (1 - beta) * gradients['weights']**2
        self.cache_b = beta * self.cache_b + (1 - beta) * gradients['bias']**2
        
        self.weights -= self.config.learning_rate * gradients['weights'] / (np.sqrt(self.cache_w) + epsilon)
        self.bias -= self.config.learning_rate * gradients['bias'] / (np.sqrt(self.cache_b) + epsilon)
    
    def _adam_update(self, gradients: Dict[str, npt.NDArray], epoch: int):
        """Adam optimizer"""
        if not hasattr(self, 'm_w'):
            self.m_w = np.zeros_like(self.weights)
            self.v_w = np.zeros_like(self.weights)
            self.m_b = np.zeros_like(self.bias)
            self.v_b = np.zeros_like(self.bias)
        
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        # Update biased first moment estimate
        self.m_w = beta1 * self.m_w + (1 - beta1) * gradients['weights']
        self.m_b = beta1 * self.m_b + (1 - beta1) * gradients['bias']
        
        # Update biased second raw moment estimate
        self.v_w = beta2 * self.v_w + (1 - beta2) * (gradients['weights']**2)
        self.v_b = beta2 * self.v_b + (1 - beta2) * (gradients['bias']**2)
        
        # Compute bias-corrected first moment estimate
        m_w_hat = self.m_w / (1 - beta1**(epoch + 1))
        m_b_hat = self.m_b / (1 - beta1**(epoch + 1))
        
        # Compute bias-corrected second raw moment estimate
        v_w_hat = self.v_w / (1 - beta2**(epoch + 1))
        v_b_hat = self.v_b / (1 - beta2**(epoch + 1))
        
        # Update parameters
        self.weights -= self.config.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        self.bias -= self.config.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
    
    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Make predictions"""
        if not self._fitted:
            raise ValueError("Model not fitted yet")
        return self._forward_pass(X)
    
    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        """Calculate accuracy score"""
        predictions = self.predict(X)
        if len(predictions.shape) == 1:
            predictions = (predictions > 0.5).astype(int)
        else:
            predictions = np.argmax(predictions, axis=1)
        return np.mean(predictions == y)

class LogisticRegression(BaseModel):
    """
    Logistic Regression implementation from scratch with advanced features
    """
    
    def __init__(self, config: ModelConfig = None, class_weight: str = None):
        super().__init__(config)
        self.regularization = self.config.regularization
        self.class_weight = class_weight
        self.class_weights = None
        
    def _initialize_parameters(self, n_features: int):
        """Xavier initialization for weights"""
        limit = np.sqrt(6 / (n_features + 1))
        self.weights = np.random.uniform(-limit, limit, n_features)
        self.bias = np.zeros(1)
        
    def _sigmoid(self, z: npt.NDArray) -> npt.NDArray:
        """Numerically stable sigmoid function"""
        # Prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _forward_pass(self, X: npt.NDArray) -> npt.NDArray:
        """Forward pass with logistic function"""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def _compute_loss(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
        """Compute binary cross-entropy loss with regularization and class weights"""
        # Add epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Apply class weights if specified
        if self.class_weights is not None:
            weights = np.where(y_true == 1, self.class_weights[1], self.class_weights[0])
            bce = -np.mean(weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        
        # L2 regularization
        l2_penalty = 0.5 * self.regularization * np.sum(self.weights**2)
        
        return bce + l2_penalty
    
    def _backward_pass(self, X: npt.NDArray, y_true: npt.NDArray, y_pred: npt.NDArray) -> Tuple[npt.NDArray, float]:
        """Compute gradients"""
        n_samples = X.shape[0]
        
        # Gradient of loss w.r.t. predictions
        dloss_dpred = (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-15)
        
        # Gradient of predictions w.r.t. weights and bias
        dw = np.dot(X.T, dloss_dpred) / n_samples + self.regularization * self.weights
        db = np.sum(dloss_dpred) / n_samples
        
        return {'weights': dw, 'bias': db}
    
    def fit(self, X: npt.NDArray, y: npt.NDArray, 
            X_val: npt.NDArray = None, y_val: npt.NDArray = None) -> 'LogisticRegression':
        """
        Train logistic regression model with advanced features
        """
        n_samples, n_features = X.shape
        
        # Compute class weights if specified
        if self.class_weight == 'balanced':
            unique, counts = np.unique(y, return_counts=True)
            total = np.sum(counts)
            self.class_weights = {cls: total / (len(unique) * count) for cls, count in zip(unique, counts)}
            if self.config.verbose:
                print(f"Class weights: {self.class_weights}")
        
        # Initialize parameters
        self._initialize_parameters(n_features)
        
        # Mark as fitted for internal use
        self._fitted = True
        
        # Training variables
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        # Mini-batch training
        n_batches = int(np.ceil(n_samples / self.config.batch_size))
        
        for epoch in range(self.config.max_epochs):
            epoch_loss = 0
            epoch_grad_norm = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            for batch in range(n_batches):
                # Get batch
                start_idx = batch * self.config.batch_size
                end_idx = min((batch + 1) * self.config.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self._forward_pass(X_batch)
                
                # Compute loss
                batch_loss = self._compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Backward pass
                gradients = self._backward_pass(X_batch, y_batch, y_pred)
                epoch_grad_norm += np.linalg.norm(gradients['weights'])
                
                # Update parameters
                self._update_parameters(gradients, epoch)
            
            # Average loss and gradient norm
            epoch_loss /= n_batches
            epoch_grad_norm /= n_batches
            
            # Calculate accuracy
            train_accuracy = self.score(X, y)
            
            # Store history
            self.history.losses.append(epoch_loss)
            self.history.accuracies.append(train_accuracy)
            self.history.gradients_norm.append(epoch_grad_norm)
            self.history.timestamps.append(time.time() - start_time)
            
            # Early stopping check
            if self.config.early_stopping and X_val is not None:
                val_loss = self._compute_loss(y_val, self._forward_pass(X_val))
                if val_loss < best_loss - self.config.tolerance:
                    best_loss = val_loss
                    self.best_weights = self.weights.copy()
                    self.best_bias = self.bias.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.patience:
                    if self.config.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # Print progress
            if self.config.verbose and epoch % 100 == 0:
                val_info = f", Val Loss: {val_loss:.4f}" if X_val is not None else ""
                print(f"Epoch {epoch}: Loss: {epoch_loss:.4f}, Acc: {train_accuracy:.4f}{val_info}")
        
        # Use best weights if early stopping was used
        if self.best_weights is not None:
            self.weights = self.best_weights
            self.bias = self.best_bias
            
        self._fitted = True
        return self

class NeuralNetwork(BaseModel):
    """
    Multi-layer Neural Network implementation from scratch
    """
    
    def __init__(self, hidden_layers: List[int] = [64, 32], 
                 activation: str = 'relu', config: ModelConfig = None):
        super().__init__(config)
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.parameters = {}
        self.cache = {}
        
    def _initialize_parameters(self, n_features: int, n_classes: int = 1):
        """He initialization for deep networks"""
        layer_dims = [n_features] + self.hidden_layers + [n_classes]
        
        for l in range(1, len(layer_dims)):
            # He initialization for ReLU, Xavier for tanh/sigmoid
            if self.activation_name == 'relu':
                scale = np.sqrt(2.0 / layer_dims[l-1])
            else:
                scale = np.sqrt(1.0 / layer_dims[l-1])
                
            self.parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scale
            self.parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    def _activation(self, z: npt.NDArray, derivative: bool = False) -> npt.NDArray:
        """Activation functions and their derivatives"""
        if self.activation_name == 'relu':
            if derivative:
                return (z > 0).astype(float)
            return np.maximum(0, z)
        
        elif self.activation_name == 'sigmoid':
            if derivative:
                s = self._activation(z)
                return s * (1 - s)
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        
        elif self.activation_name == 'tanh':
            if derivative:
                return 1 - np.tanh(z)**2
            return np.tanh(z)
        
        elif self.activation_name == 'leaky_relu':
            alpha = 0.01
            if derivative:
                return np.where(z > 0, 1, alpha)
            return np.where(z > 0, z, alpha * z)
    
    def _softmax(self, z: npt.NDArray) -> npt.NDArray:
        """Numerically stable softmax"""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def _forward_pass(self, X: npt.NDArray) -> npt.NDArray:
        """Forward propagation through all layers"""
        A = X.T  # Transpose for efficient matrix operations
        L = len(self.hidden_layers) + 1  # Total layers
        
        self.cache['A0'] = A
        
        # Hidden layers
        for l in range(1, L):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            A = self._activation(Z)
            
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
        
        # Output layer (sigmoid for binary, softmax for multi-class)
        W_out = self.parameters[f'W{L}']
        b_out = self.parameters[f'b{L}']
        Z_out = np.dot(W_out, A) + b_out
        
        if W_out.shape[0] == 1:  # Binary classification
            A_out = self._activation(Z_out)  # Sigmoid
            return A_out.T.flatten()  # Return 1D array for binary classification
        else:  # Multi-class classification
            A_out = self._softmax(Z_out)
            return A_out.T  # Return 2D array for multi-class
    
    def _compute_loss(self, y_true: npt.NDArray, y_pred: npt.NDArray) -> float:
        """Compute loss (binary cross-entropy or categorical cross-entropy)"""
        n_samples = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:  # Binary classification
            # Binary cross-entropy
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:  # Multi-class classification
            # Categorical cross-entropy
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        # Add L2 regularization
        l2_penalty = 0
        for key in self.parameters:
            if key.startswith('W'):
                l2_penalty += np.sum(self.parameters[key]**2)
        l2_penalty *= 0.5 * self.config.regularization / n_samples
        
        return loss + l2_penalty
    
    def _backward_pass(self, X: npt.NDArray, y_true: npt.NDArray, y_pred: npt.NDArray) -> Dict[str, npt.NDArray]:
        """Backward propagation"""
        gradients = {}
        n_samples = X.shape[0]
        L = len(self.hidden_layers) + 1
        
        # Convert to column vectors for efficient computation
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
            
        Y_true = y_true.T
        Y_pred = y_pred.T
        
        # Output layer gradient
        dZ = Y_pred - Y_true
        gradients[f'dW{L}'] = np.dot(dZ, self.cache[f'A{L-1}'].T) / n_samples
        gradients[f'db{L}'] = np.sum(dZ, axis=1, keepdims=True) / n_samples
        
        # Add regularization gradient
        gradients[f'dW{L}'] += self.config.regularization * self.parameters[f'W{L}'] / n_samples
        
        # Backpropagate through hidden layers
        for l in reversed(range(1, L)):
            dA = np.dot(self.parameters[f'W{l+1}'].T, dZ)
            dZ = dA * self._activation(self.cache[f'Z{l}'], derivative=True)
            
            gradients[f'dW{l}'] = np.dot(dZ, self.cache[f'A{l-1}'].T) / n_samples
            gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / n_samples
            
            # Add regularization gradient
            gradients[f'dW{l}'] += self.config.regularization * self.parameters[f'W{l}'] / n_samples
        
        return gradients
    
    def _update_parameters(self, gradients: Dict[str, npt.NDArray], epoch: int):
        """Update parameters for neural network"""
        L = len(self.hidden_layers) + 1
        
        # Initialize optimizers if needed
        if self.config.optimizer == 'adam' and not hasattr(self, 'm_params'):
            self.m_params = {}
            self.v_params = {}
            for l in range(1, L + 1):
                self.m_params[f'mW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
                self.m_params[f'mb{l}'] = np.zeros_like(self.parameters[f'b{l}'])
                self.v_params[f'vW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
                self.v_params[f'vb{l}'] = np.zeros_like(self.parameters[f'b{l}'])
        
        for l in range(1, L + 1):
            if self.config.optimizer == 'adam':
                # Adam update for each parameter
                beta1, beta2 = 0.9, 0.999
                epsilon = 1e-8
                
                # Update moments
                self.m_params[f'mW{l}'] = beta1 * self.m_params[f'mW{l}'] + (1 - beta1) * gradients[f'dW{l}']
                self.m_params[f'mb{l}'] = beta1 * self.m_params[f'mb{l}'] + (1 - beta1) * gradients[f'db{l}']
                self.v_params[f'vW{l}'] = beta2 * self.v_params[f'vW{l}'] + (1 - beta2) * (gradients[f'dW{l}']**2)
                self.v_params[f'vb{l}'] = beta2 * self.v_params[f'vb{l}'] + (1 - beta2) * (gradients[f'db{l}']**2)
                
                # Bias correction
                mW_hat = self.m_params[f'mW{l}'] / (1 - beta1**(epoch + 1))
                mb_hat = self.m_params[f'mb{l}'] / (1 - beta1**(epoch + 1))
                vW_hat = self.v_params[f'vW{l}'] / (1 - beta2**(epoch + 1))
                vb_hat = self.v_params[f'vb{l}'] / (1 - beta2**(epoch + 1))
                
                # Update parameters
                self.parameters[f'W{l}'] -= self.config.learning_rate * mW_hat / (np.sqrt(vW_hat) + epsilon)
                self.parameters[f'b{l}'] -= self.config.learning_rate * mb_hat / (np.sqrt(vb_hat) + epsilon)
                
            else:
                # Standard SGD update
                self.parameters[f'W{l}'] -= self.config.learning_rate * gradients[f'dW{l}']
                self.parameters[f'b{l}'] -= self.config.learning_rate * gradients[f'db{l}']
    
    def fit(self, X: npt.NDArray, y: npt.NDArray, 
            X_val: npt.NDArray = None, y_val: npt.NDArray = None) -> 'NeuralNetwork':
        """
        Train neural network with advanced features
        """
        n_samples, n_features = X.shape
        
        # Determine output dimension
        if len(y.shape) == 1 or y.shape[1] == 1:
            n_classes = 1
            y = y.reshape(-1, 1)
        else:
            n_classes = y.shape[1]
        
        # Initialize parameters
        self._initialize_parameters(n_features, n_classes)
        
        # Mark as fitted after parameter initialization
        self._fitted = True
        
        # Training variables
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        # Mini-batch training to avoid memory issues
        batch_size = self.config.batch_size
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(self.config.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            
            # Process mini-batches
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self._forward_pass(X_batch)
                
                # Compute loss
                batch_loss = self._compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Backward pass
                gradients = self._backward_pass(X_batch, y_batch, y_pred)
                
                # Update parameters
                self._update_parameters(gradients, epoch)
                
                # Calculate batch accuracy
                batch_acc = self.score(X_batch, y_batch.flatten())
                epoch_acc += batch_acc
            
            # Average metrics over batches
            loss = epoch_loss / n_batches
            accuracy = epoch_acc / n_batches
            
            # Store history
            self.history.losses.append(loss)
            self.history.accuracies.append(accuracy)
            
            # Calculate gradient norm (use last batch gradients as approximation)
            grad_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
            self.history.gradients_norm.append(grad_norm)
            self.history.timestamps.append(time.time() - start_time)
            
            # Early stopping
            if self.config.early_stopping and X_val is not None:
                val_pred = self._forward_pass(X_val)
                val_loss = self._compute_loss(y_val, val_pred)
                
                if val_loss < best_loss - self.config.tolerance:
                    best_loss = val_loss
                    # Deep copy of all parameters
                    self.best_parameters = {k: v.copy() for k, v in self.parameters.items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.patience:
                    if self.config.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # Print progress
            if self.config.verbose and epoch % 100 == 0:
                val_info = f", Val Loss: {val_loss:.4f}" if X_val is not None else ""
                print(f"Epoch {epoch}: Loss: {loss:.4f}, Acc: {accuracy:.4f}{val_info}")
        
        # Restore best parameters if early stopping
        if hasattr(self, 'best_parameters'):
            self.parameters = self.best_parameters
            
        return self

class KNN:
    """
    K-Nearest Neighbors implementation from scratch with optimized distance calculations
    """
    
    def __init__(self, k: int = 5, weights: str = 'uniform', 
                 metric: str = 'euclidean', n_jobs: int = -1):
        self.k = k
        self.weights = weights
        self.metric = metric
        self.n_jobs = n_jobs
        self.X_train = None
        self.y_train = None
        self._fitted = False
        
    def _compute_distance(self, X1: npt.NDArray, X2: npt.NDArray) -> npt.NDArray:
        """Compute distance matrix between two sets of points"""
        if self.metric == 'euclidean':
            # Optimized Euclidean distance using vectorization
            X1_squared = np.sum(X1**2, axis=1, keepdims=True)
            X2_squared = np.sum(X2**2, axis=1)
            cross_term = np.dot(X1, X2.T)
            distances = np.sqrt(X1_squared + X2_squared - 2 * cross_term)
            
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=2)
            
        elif self.metric == 'cosine':
            # Cosine similarity converted to distance
            norm_X1 = np.linalg.norm(X1, axis=1, keepdims=True)
            norm_X2 = np.linalg.norm(X2, axis=1, keepdims=True)
            similarities = np.dot(X1, X2.T) / (norm_X1 * norm_X2.T)
            distances = 1 - similarities
            
        elif self.metric == 'minkowski':
            p = 3  # Minkowski parameter
            distances = np.power(np.sum(np.power(np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), p), axis=2), 1/p)
        
        return distances
    
    def fit(self, X: npt.NDArray, y: npt.NDArray) -> 'KNN':
        """Store training data"""
        self.X_train = X
        self.y_train = y
        self._fitted = True
        return self
    
    def predict(self, X: npt.NDArray, batch_size: int = 1000) -> npt.NDArray:
        """Predict labels for test data with batch processing to avoid memory issues"""
        if not self._fitted:
            raise ValueError("Model not fitted yet")
        
        n_samples = X.shape[0]
        all_predictions = []
        
        # Process in batches to avoid memory overflow
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            
            # Compute distances for this batch only
            distances = self._compute_distance(X_batch, self.X_train)
            
            # Get k nearest neighbors
            nearest_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)
            nearest_labels = self.y_train[nearest_indices]
            
            # Make predictions for this batch
            if self.weights == 'uniform':
                # Majority vote
                batch_predictions = np.array([np.bincount(neighbors.astype(int)).argmax() 
                                            for neighbors in nearest_labels])
            else:
                # Weighted by inverse distance
                weights = 1.0 / (nearest_distances + 1e-8)
                batch_predictions = []
                
                for j in range(len(X_batch)):
                    unique_labels = np.unique(nearest_labels[j])
                    weighted_votes = []
                    for label in unique_labels:
                        mask = nearest_labels[j] == label
                        total_weight = np.sum(weights[j][mask])
                        weighted_votes.append((total_weight, label))
                    batch_predictions.append(max(weighted_votes)[1])
                
                batch_predictions = np.array(batch_predictions)
            
            all_predictions.append(batch_predictions)
        
        return np.concatenate(all_predictions)
    
    def predict_proba(self, X: npt.NDArray, batch_size: int = 1000) -> npt.NDArray:
        """Predict class probabilities with batch processing"""
        if not self._fitted:
            raise ValueError("Model not fitted yet")
        
        n_classes = len(np.unique(self.y_train))
        n_samples = X.shape[0]
        all_probabilities = []
        
        # Process in batches to avoid memory overflow
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            
            # Compute distances for this batch
            distances = self._compute_distance(X_batch, self.X_train)
            nearest_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            nearest_labels = self.y_train[nearest_indices]
            
            batch_probabilities = np.zeros((len(X_batch), n_classes))
            
            if self.weights == 'uniform':
                for j in range(len(X_batch)):
                    counts = np.bincount(nearest_labels[j].astype(int), minlength=n_classes)
                    batch_probabilities[j] = counts / self.k
            else:
                weights = 1.0 / (np.take_along_axis(distances, nearest_indices, axis=1) + 1e-8)
                for j in range(len(X_batch)):
                    for idx, label in enumerate(nearest_labels[j]):
                        batch_probabilities[j, int(label)] += weights[j, idx]
                    batch_probabilities[j] /= np.sum(weights[j])
            
            all_probabilities.append(batch_probabilities)
        
        return np.vstack(all_probabilities)
    
    def score(self, X: npt.NDArray, y: npt.NDArray) -> float:
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Utility functions for model evaluation
def train_test_split(X: npt.NDArray, y: npt.NDArray, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Custom train-test split implementation"""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def k_fold_cross_validation(model: BaseModel, X: npt.NDArray, y: npt.NDArray, 
                           k: int = 5, random_state: int = 42) -> Dict[str, List[float]]:
    """K-fold cross-validation implementation"""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    fold_size = n_samples // k
    
    indices = np.random.permutation(n_samples)
    scores = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for fold in range(k):
        # Create fold indices
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k - 1 else n_samples
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        # Split data
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Train model - create fresh instance with same config
        model_instance = type(model)(config=model.config)
        model_instance.fit(X_train, y_train, X_val, y_val)
        
        # Calculate scores
        train_pred = model_instance.predict(X_train)
        val_pred = model_instance.predict(X_val)
        
        train_loss = model_instance._compute_loss(y_train, train_pred)
        val_loss = model_instance._compute_loss(y_val, val_pred)
        train_acc = model_instance.score(X_train, y_train)
        val_acc = model_instance.score(X_val, y_val)
        
        scores['train_loss'].append(train_loss)
        scores['val_loss'].append(val_loss)
        scores['train_acc'].append(train_acc)
        scores['val_acc'].append(val_acc)
        
        print(f"Fold {fold + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return scores

def compute_metrics(y_true: npt.NDArray, y_pred: npt.NDArray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics"""
    # Confusion matrix
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]]),
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn
    }