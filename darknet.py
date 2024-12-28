#!/usr/bin/env python3
"""
darknet_refined_plus.py

An extended “DarkNet” code with more PDE epochs, anchor repulsion from other classes,
and slightly richer logic in predict. A final attempt to push accuracy higher
while retaining debugging statements.

Author: You
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

# =====================================================
# 1) LOAD & DOWNSAMPLE
# =====================================================

def load_mnist_small_downsample(new_size=(14,14)):
    """
    [DEBUG1] Loading & Subsampling MNIST, then downsample.
    We'll do 2k train, 500 test, downsample from (28,28)->(14,14).
    """
    print("[DEBUG1] Entering load_mnist_small_downsample()...")
    import tensorflow as tf
    (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
    print("[DEBUG2] Original train shape=", X_train_full.shape, "test shape=", X_test_full.shape)

    X_train_full = X_train_full.astype(np.float32)/255.0
    X_test_full  = X_test_full.astype(np.float32)/255.0

    def block_average_downsample(img, new_size=(14,14)):
        old_h, old_w = img.shape
        new_h, new_w = new_size
        ds = np.zeros((new_h, new_w), dtype=np.float32)
        row_scale = old_h // new_h
        col_scale = old_w // new_w
        for r in range(new_h):
            for c in range(new_w):
                r0 = r*row_scale
                r1 = (r+1)*row_scale
                c0 = c*col_scale
                c1 = (c+1)*col_scale
                block = img[r0:r1, c0:c1]
                ds[r,c] = np.mean(block)
        return ds

    train_count = 2000
    test_count  = 500
    idx_train = np.random.choice(len(X_train_full), train_count, replace=False)
    idx_test  = np.random.choice(len(X_test_full),  test_count,  replace=False)
    X_train_small = X_train_full[idx_train]
    y_train_small = y_train_full[idx_train]
    X_test_small  = X_test_full[idx_test]
    y_test_small  = y_test_full[idx_test]

    print(f"[DEBUG3] Downsampling to {new_size}, then flatten.")
    DS_train=[]
    for i in range(train_count):
        ds_img = block_average_downsample(X_train_small[i], new_size).flatten()
        DS_train.append(ds_img)
    DS_train = np.array(DS_train, dtype=np.float32)

    DS_test=[]
    for i in range(test_count):
        ds_img = block_average_downsample(X_test_small[i], new_size).flatten()
        DS_test.append(ds_img)
    DS_test = np.array(DS_test, dtype=np.float32)

    print("[DEBUG4] Done downsampling. New shapes:", DS_train.shape, DS_test.shape)
    return (DS_train, y_train_small), (DS_test, y_test_small)


# =====================================================
# 2) DARKNET CLASS
# =====================================================

class DarkNet:
    """
    Extended ephemeral PDE approach:
      - Weighted adjacency
      - Possibly anchor attraction + anchor repulsion from other classes
      - More PDE epochs
      - Additional debug statements
    """

    def __init__(self, hidden_dim=20, n_anchors=5, alpha=0.01, epochs=30, random_state=SEED):
        print("[DEBUG5] DarkNet.__init__() with hidden_dim={}, n_anchors={}, alpha={}, epochs={}, random_state={}".format(
            hidden_dim, n_anchors, alpha, epochs, random_state))
        self.hidden_dim = hidden_dim
        self.n_anchors = n_anchors
        self.alpha = alpha
        self.epochs = epochs
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.X_ = None
        self.y_ = None
        self.node_states_ = None
        self.anchors_ = {}
        self.n_ = 0
        self.dim_ = 0
        self.knn_ = None

    def fit(self, X, y):
        print("[DEBUG6] DarkNet.fit() invoked. Data shape=", X.shape)
        self.X_ = X
        self.y_ = y
        self.n_ = X.shape[0]
        self.dim_ = X.shape[1]

        classes = np.unique(y)
        print("[DEBUG7] Classes found:", classes)

        # init hidden states
        self.node_states_ = np.random.normal(0,1,(self.n_, self.hidden_dim))

        # anchor states
        for c in classes:
            self.anchors_[c] = np.random.normal(0,1,(self.n_anchors,self.hidden_dim))

        print("[DEBUG8] Building KNN, k=10, Weighted adjacency (with sign).")
        from sklearn.neighbors import NearestNeighbors
        self.knn_ = NearestNeighbors(n_neighbors=10)
        self.knn_.fit(X)

        dist, idx = self.knn_.kneighbors(X, n_neighbors=10)
        # Weighted adjacency => (j, sign, w)
        self.adj_ = []
        for i in range(self.n_):
            row_adjs=[]
            for nn_i, j in enumerate(idx[i]):
                if i==j:
                    continue
                sign = 1 if y[i]==y[j] else -1
                w = np.exp(-dist[i, nn_i]**2)
                row_adjs.append((j, sign, w))
            self.adj_.append(row_adjs)

        print(f"[DEBUG9] PDE iteration total epochs= {self.epochs}")
        for ep in range(self.epochs):
            self._pde_step(anchors=True, classes=classes)
        print("[DEBUG10] fit done PDE.")

    def _pde_step(self, anchors=True, classes=None):
        new_states = np.copy(self.node_states_)
        for i in range(self.n_):
            force = np.zeros(self.hidden_dim,dtype=float)
            for (j, sign, w) in self.adj_[i]:
                force += sign*w*(self.node_states_[j] - self.node_states_[i])
            if anchors:
                # Attract to correct anchors
                c_label = self.y_[i]
                anchor_correct = self.anchors_[c_label]
                for ast in anchor_correct:
                    force += (ast - self.node_states_[i])
                # Optionally repel from other anchors?
                # We'll do a small negative push
                for c in classes:
                    if c==c_label:
                        continue
                    anchor_others = self.anchors_[c]
                    # pick a random anchor from others? or do sum?
                    # We'll do sum but scale by -0.3
                    for ast2 in anchor_others:
                        force -= 0.3*(ast2 - self.node_states_[i])
            new_states[i] = self.node_states_[i] + self.alpha*force
        self.node_states_ = new_states

    def predict(self, X):
        """
        PDE approach for new samples:
          - Use knn_ to find neighbors
          - Weighted PDE steps=3
          - final hidden => measure which anchor is closest in mean
        """
        print("[DEBUG11] DarkNet.predict() with shape=", X.shape)
        dist, idx = self.knn_.kneighbors(X, n_neighbors=10)

        preds=[]
        for i in range(X.shape[0]):
            hidden = np.random.normal(0,1,self.hidden_dim)
            # build ephemeral adjacency for sample i
            row_adjs=[]
            # We'll do label-based sign for neighbors
            # find majority label among neighbors => c => anchor pull
            labs=[]
            for nn_i, j in enumerate(idx[i]):
                labs.append(self.y_[j])
            labs = np.array(labs)
            from collections import Counter
            ccount = Counter(labs)
            maj_label = ccount.most_common(1)[0][0]

            # Weighted adjacency but sign based on y_j vs maj_label?
            # We'll do if y_j==maj_label => +1 else -1
            for nn_i, j in enumerate(idx[i]):
                sign = 1 if self.y_[j]==maj_label else -1
                w = np.exp(-dist[i, nn_i]**2)
                row_adjs.append((j, sign, w))

            for step in range(3):
                force = np.zeros(self.hidden_dim,dtype=float)
                for (jj, sgn, ww) in row_adjs:
                    force += sgn*ww*(self.node_states_[jj] - hidden)
                # anchor => attract to maj_label anchor, repel from others
                for ast in self.anchors_[maj_label]:
                    force += (ast - hidden)
                # repel from other anchors
                for c2, arr_ast in self.anchors_.items():
                    if c2==maj_label:
                        continue
                    for a2 in arr_ast:
                        force -= 0.3*(a2 - hidden)

                hidden += self.alpha*force

            # final => pick anchor set c that is closest
            best_c=None
            best_d=1e15
            for c, anchor_st in self.anchors_.items():
                mean_ast = np.mean(anchor_st, axis=0)
                dd = np.sum((mean_ast - hidden)**2)
                if dd<best_d:
                    best_d=dd
                    best_c=c
            preds.append(best_c)

        return np.array(preds,dtype=int)


# ============================================================
# 3) MAIN
# ============================================================

def main():
    print("[DEBUG12] main() => loading data with downsample, extended version.")
    (X_train,y_train),(X_test,y_test) = load_mnist_small_downsample((14,14))
    print("[DEBUG13] Shapes => train=",X_train.shape,", test=",X_test.shape)

    print("[DEBUG14] Splitting train->val.")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    print("[DEBUG15] X_tr:",X_tr.shape,"X_val:",X_val.shape)

    dn = DarkNet(hidden_dim=20, n_anchors=5, alpha=0.01, epochs=30, random_state=SEED)
    print("[DEBUG16] Created DarkNet. Now fitting...")

    start=time.time()
    dn.fit(X_tr,y_tr)
    train_time=time.time()-start
    print(f"[DEBUG17] Train done in {train_time:.2f}s")

    # Evaluate val
    print("[DEBUG18] Evaluate on val set.")
    start=time.time()
    val_preds = dn.predict(X_val)
    val_t = time.time()-start
    val_acc = accuracy_score(y_val,val_preds)
    print(f"[DEBUG19] val_acc={val_acc:.4f}, val predict time={val_t:.2f}s")

    # Evaluate test
    print("[DEBUG20] Evaluate on test set.")
    start=time.time()
    test_preds = dn.predict(X_test)
    test_t=time.time()-start
    test_acc = accuracy_score(y_test, test_preds)
    print(f"[DEBUG21] test_acc={test_acc:.4f}, test predict time={test_t:.2f}s")

    print("\n[DEBUG22] All done with further refined dark geometry approach.\n")


if __name__=="__main__":
    main()
