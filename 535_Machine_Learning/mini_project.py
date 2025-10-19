# %%
# Importing libraries and data

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Rotate the images
# https://github.com/pytorch/vision/issues/2630
transform=transforms.Compose([lambda img: torchvision.transforms.functional.rotate(img, -90),
                                transforms.RandomHorizontalFlip(p=1),
                                transforms.ToTensor()])
training_data = datasets.EMNIST(
    root="~/data",
    split="balanced",
    download=True,
    transform=transform
)

# Utility function to extract and preprocess data (from emnist_project.py)
def get_data(data, indices=None, binarize=True, flatten=True):
    N = len(data)
    if indices is None:
        indices = range(0, N)
    X = np.stack([data[i][0].numpy() for i in indices], axis=1).squeeze(0) # (N,28,28)
    if binarize:
        X = (X > 0.5).astype(np.float32)
    if flatten:
        X = X.reshape(len(X), -1)  # (N, 784)
    y = np.array([data[i][1] for i in indices])
    return X, y

# %%
# TASK 1: Display samples from EMNIST dataset in a 5 x C table

num_classes = len(training_data.classes)  # C=47 for balanced split
print(f"Number of classes: {num_classes}")

# Map each label to a list of ids of that label
class_indices = {i: [] for i in range(num_classes)}
for idx, (img, label) in enumerate(training_data):
    class_indices[label].append(idx)

rows, cols = 5, num_classes
figure = plt.figure(figsize=(num_classes * 2, rows * 2))

for class_id in range(num_classes):
    num_samples = 5
    sampled_indices = np.random.choice(class_indices[class_id], size=num_samples, replace=False)
    for row_id in range(num_samples):
        sample_idx = sampled_indices[row_id]
        img, label = training_data[sample_idx]
        # Calculate subplot position
        subplot_idx = row_id * cols + class_id + 1
        figure.add_subplot(rows, cols, subplot_idx)
        # Add title
        if row_id == 0:
            label_name = training_data.classes[label]
            plt.title(label_name, fontsize=50)
        # Remove ticks and display
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img.squeeze(), cmap=plt.cm.binary)

plt.tight_layout()
plt.savefig("task1_5xC_samples.jpg", bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("\nTask 1 complete!")
# %%
# TASK 2: Categorical Naive Bayes Classifier
# Implementing MLE and MAP estimators with scikit-learn's Estimator interface

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class CategoricalNaiveBayes(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=1.0, beta=1.0, estimator_type='mle'):
        self.alpha = alpha
        self.beta = beta
        self.estimator_type = estimator_type

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Initialize parameters
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.class_log_prior_ = np.zeros(self.n_classes_)
        self.feature_log_prob_ = np.zeros((self.n_classes_, self.n_features_))
        N = X.shape[0]  # Total number of samples

        X_binary = (X > 0.5).astype(np.float64) # Convert to binary

        # Count samples per class
        class_counts = np.zeros(self.n_classes_)
        feature_counts = np.zeros((self.n_classes_, self.n_features_))
        for i, c in enumerate(self.classes_):
            X_c = X_binary[y == c]
            class_counts[i] = X_c.shape[0]
            feature_counts[i, :] = X_c.sum(axis=0)

        # MLE: θ_k = N_k / N , θ_{i,k} = C_{i,k} / N_k
        # Add small epsilon to avoid log(0)
        if self.estimator_type == 'mle':
            self.class_probs_ = class_counts / N
            epsilon = 1e-10
            self.feature_probs_ = (feature_counts + epsilon) / (class_counts[:, np.newaxis] + 2 * epsilon)
        # MAP: θ_k = (N_k + α) / (N + C·α) , θ_{i,k} = (C_{i,k} + β) / (N_k + 2β)
        elif self.estimator_type == 'map':
            self.class_probs_ = (class_counts + self.alpha) / (N + self.n_classes_ * self.alpha)
            self.feature_probs_ = (feature_counts + self.beta) / (class_counts[:, np.newaxis] + 2 * self.beta)
        else:
            raise ValueError(f"estimator_type must be 'mle' or 'map', got {self.estimator_type}")

        # Store log probabilities for numerical stability
        self.class_log_prior_ = np.log(self.class_probs_)
        self.feature_log_prob_ = np.log(self.feature_probs_)
        self.feature_log_prob_neg_ = np.log(1 - self.feature_probs_)

        return self

    def predict(self, X):
        # Return class with highest probability
        log_probs = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_binary = (X > 0.5).astype(np.float64) # Convert to binary
        # log P(y=k | x) ∝ log P(y=k) + Σ_i [x_i · log θ_{i,k} + (1-x_i) · log(1 - θ_{i,k})]
        log_prob = np.zeros((X.shape[0], self.n_classes_))
        for k in range(self.n_classes_):
            log_prob[:, k] = self.class_log_prior_[k]
            log_prob[:, k] += np.sum(
                X_binary * self.feature_log_prob_[k, :] +
                (1 - X_binary) * self.feature_log_prob_neg_[k, :],
                axis=1
            )
        return log_prob

    def predict_proba(self, X):
        log_proba = self.predict_log_proba(X)
        # Use log-sum-exp trick for numerical stability
        log_proba_normalized = log_proba - np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba_normalized)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def score(self, X, y):

        check_is_fitted(self)
        X, y = check_X_y(X, y)
        X_binary = (X > 0.5).astype(np.float64) # Convert to binary
        # Compute log P(x, y | Θ) for each sample
        log_likelihood = 0.0

        for i in range(X.shape[0]):
            class_idx_array = np.where(self.classes_ == y[i])[0] # Find class index

            # Handle classes not seen during training (can happen with imbalanced data)
            if len(class_idx_array) == 0:
                # Assign very low log-likelihood for unseen classes
                log_likelihood += -1000.0
                continue
            class_idx = class_idx_array[0]

            # log P(y | Θ_class)
            log_likelihood += self.class_log_prior_[class_idx]
            # log P(x | y, Θ_pixel) = Σ_j [x_j · log θ_{j,k} + (1-x_j) · log(1 - θ_{j,k})]
            log_likelihood += np.sum(
                X_binary[i] * self.feature_log_prob_[class_idx, :] +
                (1 - X_binary[i]) * self.feature_log_prob_neg_[class_idx, :]
            )

        return log_likelihood / X.shape[0]

print("\n" + "="*60)
print("\nTask 2 complete!")
# %%
# TASK 3: Plotting Learning Curves for (almost) Balanced Training Data

X, y = get_data(training_data, binarize=True, flatten=True)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.90, random_state=42)
print("\n" + "="*60)
print(f"\nTask 3 Data Setup:")
print(f"  Training set size: {len(X_train_full)} (10% of data)")
print(f"  Test set size: {len(X_test)} (90% of data)")

# Check class distribution in training set
unique_classes, train_counts = np.unique(y_train_full, return_counts=True)
print(f"\nTraining set class distribution:")
print(f"  Total samples: {len(y_train_full)}")
print(f"  Number of classes: {len(unique_classes)}")
print(f"  Min samples per class: {train_counts.min()}")
print(f"  Max samples per class: {train_counts.max()}")
print(f"  Mean samples per class: {train_counts.mean():.2f}")
print(f"  Std samples per class: {train_counts.std():.2f}")
print(f"\nClass counts: {dict(zip(unique_classes, train_counts))}")

train_sizes = np.linspace(0.1, 1.0, 5)

def plot_learning_curves(models, model_names, X_train, y_train, X_test, y_test,
                         train_sizes, title, filename):
    plt.figure(figsize=(14, 6))
    all_train_scores = []
    all_val_scores = []
    train_sizes_abs = [int(size * len(X_train)) for size in train_sizes]

    for model, name in zip(models, model_names):
        train_scores = []
        val_scores = []

        for n_samples in train_sizes_abs:
            # Train
            model.fit(X_train[:n_samples], y_train[:n_samples])
            # Score
            train_score = model.score(X_train[:n_samples], y_train[:n_samples])
            val_score = model.score(X_test, y_test)
            train_scores.append(train_score)
            val_scores.append(val_score)

        all_train_scores.append(train_scores)
        all_val_scores.append(val_scores)

    # Plot 1: Training scores
    plt.subplot(1, 2, 1)
    for i, name in enumerate(model_names):
        plt.plot(train_sizes_abs, all_train_scores[i], 'o-', label=name, linewidth=2, markersize=6)
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Average Log-Likelihood', fontsize=12)
    plt.title('Training Scores', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # Plot 2: Validation scores
    plt.subplot(1, 2, 2)
    for i, name in enumerate(model_names):
        plt.plot(train_sizes_abs, all_val_scores[i], 'o-', label=name, linewidth=2, markersize=6)
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Average Log-Likelihood', fontsize=12)
    plt.title('Validation Scores', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    print(f"  Saved: {filename}")
    plt.close()

# TASK 3.1: Fix β=1, vary α=(1, 10, 50, 100, 200).
print("\nTask 3.1: Varying α (class prior) with fixed β=1...")
alpha_values = [1, 10, 50, 100, 200]
models_31 = [CategoricalNaiveBayes(alpha=1, beta=1, estimator_type='mle')] + \
            [CategoricalNaiveBayes(alpha=a, beta=1, estimator_type='map')
             for a in alpha_values]
names_31 = ['MLE'] + [f'MAP (α={a})' for a in alpha_values]
plot_learning_curves(models_31, names_31, X_train_full, y_train_full, X_test, y_test,
                    train_sizes,
                    'Task 3.1: Learning Curves - Fixed β=1, Varying α',
                    'task3.1_learning_curves_alpha.jpg')

# TASK 3.2: Fix α=1, vary β=(1, 2, 10, 100).
print("\nTask 3.2: Varying β (pixel prior) with fixed α=1...")
beta_values = [1, 2, 10, 100]
models_32 = [CategoricalNaiveBayes(alpha=1, beta=1, estimator_type='mle')] + \
            [CategoricalNaiveBayes(alpha=1, beta=b, estimator_type='map')
             for b in beta_values]
names_32 = ['MLE'] + [f'MAP (β={b})' for b in beta_values]
plot_learning_curves(models_32, names_32, X_train_full, y_train_full, X_test, y_test,
                    train_sizes,
                    'Task 3.2: Learning Curves - Fixed α=1, Varying β',
                    'task3.2_learning_curves_beta.jpg')

# Optional debug to confirm values are mostly same (cause for overlapping values for plot 3.1 when varying α)
print("\nDEBUG: Training models on full training set and checking class priors:")
for model, name in zip(models_31, names_31):
    model.fit(X_train_full, y_train_full)
    train_score = model.score(X_train_full, y_train_full)
    val_score = model.score(X_test, y_test)
    print(f"{name:20s} | Class prob std: {model.class_probs_.std():.6f} | Train score: {train_score:.4f} | Val score: {val_score:.4f}")

print("\nTask 3 complete!")
# %%
# TASK 4: Plotting Learning Curves for Imbalanced Training Data

def create_imbalanced_dataset(X, y, n_samples, alpha_class, random_state=42):
    np.random.seed(random_state)
    classes = np.unique(y)
    n_classes = len(classes)

    # Sample class proportions from Dirichlet distribution
    class_proportions = np.random.dirichlet([alpha_class] * n_classes)
    samples_per_class = (class_proportions * n_samples).astype(int)
    diff = n_samples - samples_per_class.sum()
    samples_per_class[0] += diff

    # Sample from each class
    X_imbalanced = []
    y_imbalanced = []
    for i, c in enumerate(classes):
        class_indices = np.where(y == c)[0]
        n_needed = samples_per_class[i]
        if n_needed > 0:
            replace = n_needed > len(class_indices)
            sampled_indices = np.random.choice(class_indices, size=n_needed, replace=replace)
            X_imbalanced.append(X[sampled_indices])
            y_imbalanced.append(y[sampled_indices])
    X_imbalanced = np.vstack(X_imbalanced)
    y_imbalanced = np.hstack(y_imbalanced)

    # Shuffle the data
    shuffle_idx = np.random.permutation(len(y_imbalanced))
    X_imbalanced = X_imbalanced[shuffle_idx]
    y_imbalanced = y_imbalanced[shuffle_idx]

    return X_imbalanced, y_imbalanced


def plot_imbalanced_learning_curves(models, model_names, X_train_source, y_train_source,
                                    X_test, y_test, alpha_class_values, train_sizes,
                                    n_train_total, title, filename):
    n_rows = len(alpha_class_values)
    n_cols = 2  # Training and validation
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, alpha_class in enumerate(alpha_class_values):
        # Create imbalanced training data
        X_train_imb, y_train_imb = create_imbalanced_dataset(
            X_train_source, y_train_source,
            n_train_total, alpha_class, random_state=42
        )

        train_sizes_abs = [int(size * len(X_train_imb)) for size in train_sizes]
        all_train_scores = []
        all_val_scores = []

        for model, name in zip(models, model_names):
            train_scores = []
            val_scores = []
            for n_samples in train_sizes_abs:
                # Train
                model.fit(X_train_imb[:n_samples], y_train_imb[:n_samples])
                # Score
                train_score = model.score(X_train_imb[:n_samples], y_train_imb[:n_samples])
                val_score = model.score(X_test, y_test)
                train_scores.append(train_score)
                val_scores.append(val_score)
            all_train_scores.append(train_scores)
            all_val_scores.append(val_scores)

        # Plot training scores
        ax_train = axes[row_idx, 0]
        for i, name in enumerate(model_names):
            ax_train.plot(train_sizes_abs, all_train_scores[i], 'o-', label=name, linewidth=2, markersize=5)
        ax_train.set_xlabel('Training Set Size', fontsize=11)
        ax_train.set_ylabel('Avg Log-Likelihood', fontsize=11)
        ax_train.set_title(f'Training Scores (α_class={alpha_class})', fontsize=12)
        ax_train.legend(loc='best', fontsize=9)
        ax_train.grid(True, alpha=0.3)

        # Plot validation scores
        ax_val = axes[row_idx, 1]
        for i, name in enumerate(model_names):
            ax_val.plot(train_sizes_abs, all_val_scores[i], 'o-', label=name, linewidth=2, markersize=5)
        ax_val.set_xlabel('Training Set Size', fontsize=11)
        ax_val.set_ylabel('Avg Log-Likelihood', fontsize=11)
        ax_val.set_title(f'Validation Scores (α_class={alpha_class})', fontsize=12)
        ax_val.legend(loc='best', fontsize=9)
        ax_val.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    print(f"  Saved: {filename}")
    plt.close()

print("\n" + "="*60)
# TASK 4.1: For MAP model, fix α=1, try β=(1, 1.2, 2, 10, 100) for α_class=(0.1, 0.2, 0.5, 1, 10, 100)
print("\nTask 4.1: Imbalanced data - Varying β with fixed α=1...")
beta_values_41 = [1, 1.2, 2, 10, 100]
alpha_class_values_41 = [0.1, 0.2, 0.5, 1, 10, 100]
models_41 = [CategoricalNaiveBayes(alpha=1, beta=1, estimator_type='mle')] + \
            [CategoricalNaiveBayes(alpha=1, beta=b, estimator_type='map') for b in beta_values_41]
names_41 = ['MLE'] + [f'MAP (β={b})' for b in beta_values_41]
plot_imbalanced_learning_curves(models_41, names_41, X_train_full, y_train_full,
                               X_test, y_test, alpha_class_values_41, train_sizes,
                               len(X_train_full),
                               'Task 4.1: Imbalanced Data - Fixed α=1, Varying β',
                               'task4.1_imbalanced_beta.jpg')

# TASK 4.2: For MAP model, fix β=1, try α=(1, 10, 100, 1000) for α_class=(0.1, 0.2, 0.5, 1, 10, 100)
print("\nTask 4.2: Imbalanced data - Varying α with fixed β=1...")
alpha_values_42 = [1, 10, 100, 1000]
alpha_class_values_42 = [0.1, 0.2, 0.5, 1, 10, 100]
models_42 = [CategoricalNaiveBayes(alpha=1, beta=1, estimator_type='mle')] + \
            [CategoricalNaiveBayes(alpha=a, beta=1, estimator_type='map') for a in alpha_values_42]
names_42 = ['MLE'] + [f'MAP (α={a})' for a in alpha_values_42]
plot_imbalanced_learning_curves(models_42, names_42, X_train_full, y_train_full,
                               X_test, y_test, alpha_class_values_42, train_sizes,
                               len(X_train_full),
                               'Task 4.2: Imbalanced Data - Fixed β=1, Varying α',
                               'task4.2_imbalanced_alpha.jpg')

print("\nTask 4 complete!")
print("\n" + "="*60)