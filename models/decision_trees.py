from collections import Counter, defaultdict
import numpy as np
from typing import Optional, Dict, Any

class Node:
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.class_counts: Optional[Dict[Any, Any]] = None
    def is_leaf_node(self):
        return self.left is None and self.right is None

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, 
                 max_features=None,
                 max_leaf_nodes=None,
                 min_samples_split=2,
                 random_seed=0):

        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.random_seed = random_seed
        self.root = None
        self._leaf_count = 0
        np.random.seed(self.random_seed)
    def __call__(self):

        return self
    def fit(self, X, y, sample_weight=None):
        self._classes = np.unique(y)
        self.root = self._build_tree(X, y, sample_weight)
        return self


    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict_proba(self, X):
        return np.array([self._traverse_proba(x, self.root) for x in X])

    def _gini(self, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(y))

        weight_per_class = defaultdict(float)
        total_weight = np.sum(sample_weight)

        for label, weight in zip(y, sample_weight):
            weight_per_class[label] += weight

        impurity = 1.0
        for label in weight_per_class:
            prob_of_lbl = weight_per_class[label] / total_weight
            impurity -= prob_of_lbl ** 2

        return impurity

    def _best_split(self, X, y, sample_weight):
        best_gain = -1
        best_feature, best_threshold = None, None
        current_impurity = self._gini(y, sample_weight)
        feature_indices = np.arange(X.shape[1]) if self.max_features is None else \
            np.random.choice(X.shape[1], min(self.max_features, X.shape[1]), replace=False)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                    continue

                y_left, y_right = y[left_idx], y[right_idx]
                p = len(y_left) / len(y)
                gain = current_impurity - p * self._gini(y_left, sample_weight) - (1 - p) * self._gini(y_right, sample_weight)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, sample_weight, depth=0, ):
        num_samples_per_class = Counter(y)
        predicted_class = num_samples_per_class.most_common(1)[0][0]
        class_probs = self._normalize_counts(num_samples_per_class)

        if (depth >= self.max_depth or 
            len(num_samples_per_class) == 1 or 
            len(y) < self.min_samples_split or 
            (self.max_leaf_nodes is not None and self._leaf_count >= self.max_leaf_nodes)):
            self._leaf_count += 1
            leaf = Node(value=predicted_class)
            leaf.class_counts = class_probs
            return leaf

        feature, threshold = self._best_split(X, y, sample_weight)
        if feature is None:
            self._leaf_count += 1
            leaf = Node(value=predicted_class)
            leaf.class_counts = class_probs
            return leaf

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            self._leaf_count += 1
            leaf = Node(value=predicted_class)
            leaf.class_counts = class_probs
            return leaf

        if self.max_leaf_nodes is not None and self._leaf_count + 2 > self.max_leaf_nodes:
            self._leaf_count += 1
            leaf = Node(value=predicted_class)
            leaf.class_counts = class_probs
            return leaf

        left = self._build_tree(X[left_idx], y[left_idx], sample_weight, depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], sample_weight, depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _traverse_proba(self, x, node):
        if node.is_leaf_node():
            return self._proba_vector(node.class_counts)
        if x[node.feature] <= node.threshold:
            return self._traverse_proba(x, node.left)
        else:
            return self._traverse_proba(x, node.right)

    def _proba_vector(self, class_counts):
        return np.array([class_counts.get(cls, 0.0) for cls in self._classes])

    def _normalize_counts(self, counter):
        total = sum(counter.values())
        return {cls: count / total for cls, count in counter.items()}




class DecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def __call__(self):
        return self

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=float)
        self.root = self._build_tree(X, y, sample_weight, depth=0)
        return self

    def predict(self, X):
        return np.array([self._traverse_tree_iterative(x, self.root) for x in X])

    def _traverse_tree_iterative(self, x, node):
        while not node.is_leaf_node():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _weighted_mse(self, y, sample_weight):
        if len(y) == 0:
            return 0
        mean = np.average(y, weights=sample_weight)
        return np.average((y - mean) ** 2, weights=sample_weight)

    def _best_split(self, X, y, sample_weight):
        best_mse = float('inf')
        best_feature, best_threshold = None, None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            sorted_idx = np.argsort(X[:, feature])
            X_f = X[sorted_idx, feature]
            y_f = y[sorted_idx]
            w_f = sample_weight[sorted_idx]

            sum_w = np.cumsum(w_f)
            sum_y = np.cumsum(y_f * w_f)
            sum_y2 = np.cumsum((y_f ** 2) * w_f)

            total_weight = sum_w[-1]
            total_y = sum_y[-1]
            total_y2 = sum_y2[-1]

            for i in range(1, n_samples):
                if X_f[i] == X_f[i - 1]:
                    continue

                w_left = sum_w[i - 1]
                y_left = sum_y[i - 1]
                y2_left = sum_y2[i - 1]

                w_right = total_weight - w_left
                y_right = total_y - y_left
                y2_right = total_y2 - y2_left

                if w_left < self.min_samples_split or w_right < self.min_samples_split:
                    continue

                mse_left = (y2_left / w_left) - (y_left / w_left) ** 2
                mse_right = (y2_right / w_right) - (y_right / w_right) ** 2

                mse = (w_left * mse_left + w_right * mse_right) / total_weight

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = (X_f[i] + X_f[i - 1]) / 2

        return best_feature, best_threshold

    def _build_tree(self, X, y, sample_weight, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(value=np.average(y, weights=sample_weight))

        feature, threshold = self._best_split(X, y, sample_weight)
        if feature is None:
            return Node(value=np.average(y, weights=sample_weight))

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left = self._build_tree(X[left_idx], y[left_idx], sample_weight[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], sample_weight[right_idx], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)










