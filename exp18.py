import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

doc = [
    "CUTM is located in Paralakhemundi",
    "Paralakhemundi is located in Odisha",
    "Odisha is a state"    
]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(doc)
words = tfidf.get_feature_names_out()
print(words)
print("Original shape = ", X.shape, "\n")
print(X)

#eigen val, vec, cov, mean
X_centered = X - X.mean(axis=0)
print(X_centered, "\n")
cov = np.cov(X_centered, rowvar=False)
print(cov, "\n")

eig_vals, eig_vecs = np.linalg.eigh(cov)
idx  = eig_vals.argsort()[::-1]
eig_vecs = eig_vecs[:, idx[:2]]

X_pca = np.dot(X_centered, eig_vecs)
print("Reduced shape = ", X_pca.shape)
print(X_pca)