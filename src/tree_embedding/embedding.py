import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def get_binary_path(tree, x, max_depth=None):
    """
    Returns binary list of decisions (0 for left, 1 for right) from root to leaf for input x.
    """
    node_indicator = tree.decision_path(x.reshape(1, -1))
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    path = []
    depth = max_depth or tree.tree_.max_depth
    for i in range(len(node_index) - 1):
        parent = node_index[i]
        child = node_index[i + 1]
        if tree.tree_.children_left[parent] == child:
            path.append(0)
        else:
            path.append(1)
        if len(path) == depth:
            break

    if len(path) < depth:
        path += [0] * (depth - len(path))
    return path[:depth]


def path_to_decimal(path):
    """Convert binary path list to decimal, MSB=first element."""
    val = 0
    for bit in path:
        val = (val << 1) | bit
    return val


def iterative_ardt(X, y, vocab, embed_dim=1, n_iters=3, max_depth=None, random_state=42):
    """
    Perform iterative ARDT embedding:
    For each iteration:
      - Train a decision tree on X -> y
      - For each sample, extract binary path and convert to decimal embedding
      - Evaluate accuracy of a classifier on these embeddings
      - Replace X with new embeddings as single feature and repeat

    Returns:
        scores: list of accuracy scores per iteration
        embeddings: list of embedding vectors per iteration
    """
    scores = []
    embeddings_list = []
    X_current = X

    for _ in range(n_iters):
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        clf.fit(X_current, y)

        emb = np.array([path_to_decimal(get_binary_path(clf, x, max_depth))
                        for x in X_current])
        
        embeddings_list.append(emb)

        acc = accuracy_score(y, clf.predict(X_current))
        scores.append(acc)

        X_current = emb.reshape(-1, 1)
    return scores, embeddings_list

