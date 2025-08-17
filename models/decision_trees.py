from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
from collections import Counter, defaultdict


class Node:
    """
    A node in the decision tree.
    
    Parameters
    ----------
    feature : Optional[int], default=None
        The feature index to split on (None for leaf nodes).
    threshold : Optional[float], default=None
        The threshold value for the split (None for leaf nodes).
    left : Optional[Node], default=None
        Left child node.
    right : Optional[Node], default=None
        Right child node.
    value : Optional[Any], default=None
        The predicted class for leaf nodes.
    """
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None, 
                 value: Optional[Any] = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.class_counts: Dict[Any, float] = {}
    
    def is_leaf_node(self) -> bool:
        """Check if this node is a leaf node."""
        return self.value is not None


class DecisionTreeClassifier:
    """
    A simple Decision Tree Classifier supporting Gini impurity, optional feature subsampling, and sample weights.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    max_features : Optional[int], default=None
        Number of features to consider when looking for the best split.
    max_leaf_nodes : Optional[int], default=None
        Maximum number of leaf nodes.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    random_seed : int, default=0
        Seed for random number generator for reproducibility.
    """
    def __init__(self, max_depth: int = 5, max_features: Optional[int] = None,
                 max_leaf_nodes: Optional[int] = None, min_samples_split: int = 2, 
                 random_seed: int = 0) -> None:
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.random_seed = random_seed
        self.root: Optional[Node] = None
        self._leaf_count = 0
        self._rng = np.random.default_rng(self.random_seed)
        self._classes: Optional[np.ndarray] = None

    def __call__(self) -> 'DecisionTreeClassifier':
        """Return self for compatibility."""
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None) -> 'DecisionTreeClassifier':
        """
        Build a decision tree classifier from the training set (X, y).
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values (class labels).
        sample_weight : Optional[np.ndarray] of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            
        Returns
        -------
        DecisionTreeClassifier
            Fitted estimator.
        """
        self._classes = np.unique(y)
        self.root = self._build_tree(np.array(X), np.array(y), sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted classes.
            
        Raises
        ------
        ValueError
            If the classifier is not fitted yet.
        """
        if self.root is None:
            raise ValueError("DecisionTreeClassifier instance is not fitted yet.")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
            
        Raises
        ------
        ValueError
            If the classifier is not fitted yet.
        """
        if self.root is None:
            raise ValueError("DecisionTreeClassifier instance is not fitted yet.")
        return np.array([self._traverse_proba(x, self.root) for x in X])

    def _gini(self, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Calculate the Gini impurity for a set of labels.
        
        Parameters
        ----------
        y : np.ndarray
            The target labels.
        sample_weight : Optional[np.ndarray], default=None
            Sample weights for each observation.
            
        Returns
        -------
        float
            The Gini impurity score.
        """
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        weight_per_class = defaultdict(float)
        total_weight = np.sum(sample_weight)
        
        for label, weight in zip(y, sample_weight):
            weight_per_class[label] += weight
            
        impurity = 1.0 - sum((weight_per_class[label] / total_weight) ** 2 
                            for label in weight_per_class)
        return impurity

    def _best_split(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray]) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best split for the given data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values.
        sample_weight : Optional[np.ndarray] of shape (n_samples,)
            Sample weights.
            
        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            The best feature index and threshold value. Returns (None, None) if no split found.
        """
        best_gain = -1.0
        best_feature, best_threshold = None, None
        current_impurity = self._gini(y, sample_weight)
        
        # Select features to consider for splitting
        feature_indices = (np.arange(X.shape[1]) if self.max_features is None else 
                          self._rng.choice(X.shape[1], min(self.max_features, X.shape[1]), replace=False))
        
        for feature in feature_indices:
            x_column = X[:, feature]
            sort_idx = np.argsort(x_column)
            x_sorted, y_sorted = x_column[sort_idx], y[sort_idx]
            w_sorted = sample_weight[sort_idx] if sample_weight is not None else None
            
            # Find valid split points
            diffs = np.diff(x_sorted)
            valid = diffs > 1e-7
            if not np.any(valid):
                continue
                
            thresholds = (x_sorted[:-1][valid] + x_sorted[1:][valid]) / 2
            
            for threshold in thresholds:
                left_idx, right_idx = x_column <= threshold, x_column > threshold
                
                # Check minimum samples constraint
                if (np.sum(left_idx) < self.min_samples_split or 
                    np.sum(right_idx) < self.min_samples_split):
                    continue
                
                y_left, y_right = y[left_idx], y[right_idx]
                w_left = sample_weight[left_idx] if sample_weight is not None else None
                w_right = sample_weight[right_idx] if sample_weight is not None else None
                
                # Calculate information gain
                p = len(y_left) / len(y)
                gain = (current_impurity - 
                       p * self._gini(y_left, w_left) - 
                       (1 - p) * self._gini(y_right, w_right))
                
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
                    
        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: Optional[np.ndarray], depth: int = 0) -> Node:
        """
        Recursively build the decision tree.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values.
        sample_weight : Optional[np.ndarray] of shape (n_samples,)
            Sample weights.
        depth : int, default=0
            Current depth of the tree.
            
        Returns
        -------
        Node
            The root node of the constructed subtree.
        """
        num_samples_per_class = Counter(y)
        predicted_class = num_samples_per_class.most_common(1)[0][0]
        class_probs = self._normalize_counts(num_samples_per_class)
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(num_samples_per_class) == 1 or 
            len(y) < self.min_samples_split or
            (self.max_leaf_nodes is not None and self._leaf_count >= self.max_leaf_nodes)):
            self._leaf_count += 1
            leaf = Node(value=predicted_class)
            leaf.class_counts = class_probs
            return leaf
        
        # Find best split
        feature, threshold = self._best_split(X, y, sample_weight)
        if feature is None:
            self._leaf_count += 1
            leaf = Node(value=predicted_class)
            leaf.class_counts = class_probs
            return leaf
        
        # Create split indices
        left_idx, right_idx = X[:, feature] <= threshold, X[:, feature] > threshold
        
        # Check if split is valid
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            self._leaf_count += 1
            leaf = Node(value=predicted_class)
            leaf.class_counts = class_probs
            return leaf
        
        # Check max leaf nodes constraint before creating children
        if self.max_leaf_nodes is not None and self._leaf_count + 2 > self.max_leaf_nodes:
            self._leaf_count += 1
            leaf = Node(value=predicted_class)
            leaf.class_counts = class_probs
            return leaf
        
        # Recursively build children
        left = self._build_tree(
            X[left_idx], y[left_idx], 
            sample_weight[left_idx] if sample_weight is not None else None, 
            depth + 1
        )
        right = self._build_tree(
            X[right_idx], y[right_idx], 
            sample_weight[right_idx] if sample_weight is not None else None, 
            depth + 1
        )
        
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _traverse_tree(self, x: np.ndarray, node: Optional[Node]) -> Any:
        """
        Traverse the tree to make a prediction for a single sample.
        
        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single input sample.
        node : Optional[Node]
            Current node in the tree.
            
        Returns
        -------
        Any
            The predicted class label.
        """
        if node is None or node.is_leaf_node():
            return node.value if node is not None else None
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _traverse_proba(self, x: np.ndarray, node: Optional[Node]) -> np.ndarray:
        """
        Traverse the tree to get class probabilities for a single sample.
        
        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single input sample.
        node : Optional[Node]
            Current node in the tree.
            
        Returns
        -------
        np.ndarray of shape (n_classes,)
            The class probabilities.
        """
        if node is None or node.is_leaf_node():
            return self._proba_vector(node.class_counts) if node is not None else np.zeros(len(self._classes)) # type: ignore
        if x[node.feature] <= node.threshold:
            return self._traverse_proba(x, node.left)
        return self._traverse_proba(x, node.right)

    def _proba_vector(self, class_counts: Dict[Any, float]) -> np.ndarray:
        """
        Convert class counts to probability vector.
        
        Parameters
        ----------
        class_counts : Dict[Any, float]
            Dictionary mapping class labels to their counts/probabilities.
            
        Returns
        -------
        np.ndarray of shape (n_classes,)
            Probability vector for all classes.
        """
        total = sum(class_counts.values())
        if total == 0:
            return np.zeros(len(self._classes)) # type: ignore
        return np.array([class_counts.get(cls, 0.0) / total for cls in self._classes]) # type: ignore

    @staticmethod
    def _normalize_counts(counter: Counter) -> Dict[Any, float]:
        """
        Normalize class counts to probabilities.
        
        Parameters
        ----------
        counter : Counter
            Counter object with class counts.
            
        Returns
        -------
        Dict[Any, float]
            Dictionary mapping classes to their normalized probabilities.
        """
        total = sum(counter.values())
        return {cls: count / total for cls, count in counter.items()}





class DecisionTreeRegressor:
    """
    A decision tree regressor that uses the CART (Classification and Regression Trees) algorithm.
    
    This implementation supports weighted samples and feature subsampling for improved
    generalization. The tree is built by recursively partitioning the feature space
    to minimize the weighted variance of the target variable within each partition.
    
    Parameters
    ----------
    max_depth : int, default=5
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or contain less than min_samples_split samples.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    max_features : Optional[int], default=None
        The number of features to consider when looking for the best split.
        If None, then all features are considered.
    
    Attributes
    ----------
    root_ : Optional[Node]
        The root node of the tree after fitting.
    n_features_in_ : int
        The number of features seen during fit.
    n_outputs_ : int
        The number of outputs seen during fit.
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, 
                 max_features: Optional[int] = None) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root_: Optional[Node] = None
        self.n_features_in_: int = 0
        self.n_outputs_: int = 0
    
    def __call__(self) -> 'DecisionTreeRegressor':
        """Return self for compatibility with ensemble methods."""
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            sample_weight: Optional[np.ndarray] = None) -> 'DecisionTreeRegressor':
        """
        Build a decision tree regressor from the training set (X, y).
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to dtype=np.float32.
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        sample_weight : Optional[np.ndarray] of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            
        Returns
        -------
        DecisionTreeRegressor
            Fitted estimator.
            
        Raises
        ------
        ValueError
            If X and y have incompatible shapes or if sample_weight has invalid shape.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        
        # Ensure sample_weight is always a numpy array
        if sample_weight is None:
            weight_array: np.ndarray = np.ones(len(y), dtype=np.float32)
        else:
            weight_array = np.asarray(sample_weight, dtype=np.float32)
            if weight_array.shape[0] != X.shape[0]:
                raise ValueError("sample_weight has invalid shape")
            weight_array = weight_array / np.sum(weight_array) * len(weight_array)
        
        n_features_split: int = (
            X.shape[1] if self.max_features is None else min(self.max_features, X.shape[1])
        )
        
        self.root_ = self._build_tree(X, y, weight_array, 0, n_features_split)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to dtype=np.float32.
            
        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
            
        Raises
        ------
        ValueError
            If the regressor is not fitted yet.
        """
        if self.root_ is None:
            raise ValueError("DecisionTreeRegressor instance is not fitted yet.")
        
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has different number of features than during fit")
            
        return np.array([self._predict_one(x, self.root_) for x in X], dtype=np.float32)
    
    def _predict_one(self, x: np.ndarray, node: Optional[Node]) -> float:
        """
        Predict a single sample by traversing the tree.
        
        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single input sample.
        node : Optional[Node]
            Current node in the tree.
            
        Returns
        -------
        float
            The predicted value for this sample.
        """
        if node is None:
            return 0.0
            
        current_node = node
        while not current_node.is_leaf_node(): # type: ignore
            if current_node.feature is None or current_node.threshold is None: # type: ignore
                break
            current_node = current_node.left if x[current_node.feature] <= current_node.threshold else current_node.right # type: ignore
        return current_node.value if current_node.value is not None else 0.0 # type: ignore

    def _calculate_split_gain(self, y: np.ndarray, sample_weight: np.ndarray, 
                            left_mask: np.ndarray) -> float:
        """
        Calculate the gain from a potential split using weighted variance reduction.
        
        Parameters
        ----------
        y : np.ndarray
            The target values for the current node.
        sample_weight : np.ndarray
            The weights for each sample.
        left_mask : np.ndarray
            Boolean mask indicating which samples go to the left child.
            
        Returns
        -------
        float
            The variance reduction achieved by this split. Higher values indicate better splits.
        """
        total_weight: float = float(np.sum(sample_weight))
        if total_weight == 0:
            return -np.inf
        
        left_weight: float = float(np.sum(sample_weight[left_mask]))
        if left_weight == 0 or left_weight == total_weight:
            return -np.inf
        
        right_weight: float = total_weight - left_weight
        if left_weight < self.min_samples_split or right_weight < self.min_samples_split:
            return -np.inf

        left_y, left_w = y[left_mask], sample_weight[left_mask]
        right_mask = ~left_mask
        right_y, right_w = y[right_mask], sample_weight[right_mask]
        
        left_sum: float = float(np.dot(left_y, left_w))
        right_sum: float = float(np.dot(right_y, right_w))
        
        left_mean: float = left_sum / left_weight
        right_mean: float = right_sum / right_weight

        left_var: float = float(np.dot(left_w, left_y ** 2)) / left_weight - left_mean ** 2
        right_var: float = float(np.dot(right_w, right_y ** 2)) / right_weight - right_mean ** 2
        
        total_mean: float = float(np.dot(sample_weight, y)) / total_weight
        current_var: float = float(np.dot(sample_weight, y ** 2)) / total_weight - total_mean ** 2

        weighted_var: float = (left_weight * left_var + right_weight * right_var) / total_weight

        return current_var - weighted_var

    def _best_split(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: np.ndarray, n_features_split: int) -> Tuple[Optional[int], Optional[float]]:
        """
        Find the best split for the given data using variance reduction.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values.
        sample_weight : np.ndarray of shape (n_samples,)
            The weights for each sample.
        n_features_split : int
            Number of features to consider for splitting.
            
        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            The best feature index and threshold value. Returns (None, None) if no valid split found.
        """
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None
        
        best_gain: float = -np.inf
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        
        feature_indices: np.ndarray = (
            np.random.choice(n_features, n_features_split, replace=False)
            if n_features_split < n_features else np.arange(n_features)
        )

        for feature in feature_indices:
            x_column = X[:, feature]
            sort_idx = np.argsort(x_column)
            x_sorted = x_column[sort_idx]
            y_sorted = y[sort_idx]
            w_sorted = sample_weight[sort_idx]

            diffs = np.diff(x_sorted)
            valid = diffs > 1e-7
            if not np.any(valid):
                continue

            thresholds = (x_sorted[:-1][valid] + x_sorted[1:][valid]) / 2

            for threshold in thresholds:
                left_mask = x_column <= threshold
                gain = self._calculate_split_gain(y, sample_weight, left_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, 
                   sample_weight: np.ndarray, depth: int, n_features_split: int) -> Node:
        """
        Recursively build the decision tree using variance reduction.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : np.ndarray of shape (n_samples,)
            The target values.
        sample_weight : np.ndarray of shape (n_samples,)
            The weights for each sample.
        depth : int
            Current depth of the tree.
        n_features_split : int
            Number of features to consider for splitting at each node.
            
        Returns
        -------
        Node
            The root node of the constructed subtree.
        """
        n_samples = X.shape[0]
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.ptp(y) < 1e-7):
            return Node(value=float(np.average(y, weights=sample_weight)))

        feature, threshold = self._best_split(X, y, sample_weight, n_features_split)
        if feature is None:
            return Node(value=float(np.average(y, weights=sample_weight)))
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], sample_weight[left_mask], 
                               depth + 1, n_features_split)
        right = self._build_tree(X[right_mask], y[right_mask], sample_weight[right_mask], 
                                depth + 1, n_features_split)
        return Node(feature=feature, threshold=threshold, left=left, right=right)






