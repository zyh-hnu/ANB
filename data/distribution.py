import numpy as np


def dirichlet_split(dataset, num_clients, alpha=0.5):
    """
    Split dataset indices into Non-IID partitions using Dirichlet distribution.

    Args:
        dataset: PyTorch dataset with targets attribute
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (smaller = more skewed)

    Returns:
        List of lists containing indices for each client
    """
    # Get number of classes
    num_classes = len(set(dataset.targets))

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # For each class, split the indices using Dirichlet
    for class_idx in range(num_classes):
        # Get indices of this class
        class_indices = np.where(np.array(dataset.targets) == class_idx)[0]
        np.random.shuffle(class_indices)

        # Split using Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to counts
        counts = (proportions * len(class_indices)).astype(int)
        # Adjust last count to ensure we use all samples
        counts[-1] = len(class_indices) - sum(counts[:-1])

        # Assign to clients
        start = 0
        for client_id, count in enumerate(counts):
            if count > 0:
                client_indices[client_id].extend(class_indices[start:start+count])
            start += count

    # Shuffle each client's data
    for indices in client_indices:
        np.random.shuffle(indices)

    return client_indices