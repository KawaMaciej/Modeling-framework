
from collections import Counter
import numpy as np
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.class_counts = None

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
    def fit(self, X, y):
        self._classes = np.unique(y)
        self.root = self._build_tree(X, y)
        return self


    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict_proba(self, X):
        return np.array([self._traverse_proba(x, self.root) for x in X])

    def _gini(self, y):
        hist = Counter(y)
        impurity = 1.0
        for label in hist:
            prob_of_lbl = hist[label] / len(y)
            impurity -= prob_of_lbl ** 2
        return impurity

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        current_impurity = self._gini(y)
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
                gain = current_impurity - p * self._gini(y_left) - (1 - p) * self._gini(y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
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

        feature, threshold = self._best_split(X, y)
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

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

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

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _mse(self, y):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _best_split(self, X, y):
        best_mse = float('inf')
        best_feature, best_threshold = None, None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                    continue

                y_left, y_right = y[left_idx], y[right_idx]
                mse = (len(y_left) * self._mse(y_left) + len(y_right) * self._mse(y_right)) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(value=np.mean(y))

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=np.mean(y))

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)










