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

common = set([])
for tissue1 in tisse_gene:
    for tissue2 in tisse_gene:
        if tissue1 != tissue2:
            common = common.union(tisse_gene[tissue1].intersection(tisse_gene[tissue2]))

for tissue in tisse_gene:
    tisse_gene[tissue] -= common

for tissue in tisse_gene:
    tisse_gene[tissue] = tisse_gene[tissue].intersection(set(go_emb.keys()).union(archs4_emb.keys()))
    print(tissue, len(tisse_gene[tissue]))

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
        y.append(str(i))

X = np.array(X)
y = np.array(y)

model = TSNE(n_components=2, random_state=0)
tsne_pj = model.fit_transform(X)
plt.scatter(tsne_pj[:,0], tsne_pj[:,1], c=[int(c) for c in y], cmap='tab20b')
plt.savefig('tsne.png')

index = np.arange(len(X))
np.random.shuffle(index)
train_index = index[:int(0.8 * len(X))]
test_index = index[int(0.8 * len(X)):]
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf.fit(X[train_index], y[train_index])
y_hat = clf.predict(X[test_index])
print(y_hat)
print(y[test_index])
print((y_hat==y[test_index]).sum()/len(y_hat))
