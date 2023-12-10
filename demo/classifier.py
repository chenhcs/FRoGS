import numpy as np
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

with open('../data/gene_vec_go_256.csv', mode='r') as infile:
    reader = csv.reader(infile)
    go_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

with open('../data/gene_vec_archs4_256.csv', mode='r') as infile:
    reader = csv.reader(infile)
    archs4_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

tisse_gene = {}
with open('data/tissue_specific.txt') as fr:
    for line in fr:
        items = line.split('\n')[0].split(',')
        if items[0] not in tisse_gene:
            tisse_gene[items[0]] = set(items[1:])
        else:
            tisse_gene[items[0]] = tisse_gene[items[0]].union(set(items[1:]))

#Filter out genes that appear in more than one tissue
common = set([])
for tissue1 in tisse_gene:
    for tissue2 in tisse_gene:
        if tissue1 != tissue2:
            common = common.union(tisse_gene[tissue1].intersection(tisse_gene[tissue2]))

for tissue in tisse_gene:
    tisse_gene[tissue] -= common

#Filter out genes that do not appear in our embedding
print('Number of genes:')
for tissue in tisse_gene:
    tisse_gene[tissue] = tisse_gene[tissue].intersection(set(go_emb.keys()).union(archs4_emb.keys()))
    print(tissue, len(tisse_gene[tissue]))

#Assign embeddings and labels to genes
X = []
y = []
for i,tissue in enumerate(tisse_gene):
    for gid in tisse_gene[tissue]:
        emb = np.zeros(512)
        if gid in go_emb:
            emb[:256]=go_emb[gid]

        if gid in archs4_emb:
            emb[256:]=archs4_emb[gid]
        X.append(emb)
        y.append(tissue)

X = np.array(X)
y = np.array(y)

#Generate the t-SNE plot
model = TSNE(n_components=2, random_state=0)
tsne_pj = model.fit_transform(X)
for tissue in tisse_gene:
    idx = np.where(y == tissue)[0]
    plt.scatter(tsne_pj[idx,0], tsne_pj[idx,1], label=tissue)
plt.legend()
plt.savefig('tsne.png')

#Build the random forest classifer. Using 80% genes for training and 20% genes for test
index = np.arange(len(X))
np.random.shuffle(index)
train_index = index[:int(0.8 * len(X))]
test_index = index[int(0.8 * len(X)):]
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf.fit(X[train_index], y[train_index])
y_hat = clf.predict(X[test_index])
print('Groundtruth', y_hat)
print('Predictions', y[test_index])
print('Accuracy on test:', (y_hat==y[test_index]).sum()/len(y_hat))
