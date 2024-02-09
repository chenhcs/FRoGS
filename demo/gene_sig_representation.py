import numpy as np
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from random import sample
from gen_repr_util import gene_sim_mean_std, compute_list_emb
from sklearn.ensemble import RandomForestClassifier

with open('../data/gene_vec_go_256.csv', mode='r') as infile:
    reader = csv.reader(infile)
    go_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

with open('../data/gene_vec_archs4_256.csv', mode='r') as infile:
    reader = csv.reader(infile)
    archs4_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

gene_lists = {}
all_genes = []
with open('data/gene_lists_tissue.txt') as fr:
    for line in fr:
        tissue = line.split(',')[0]
        gene_list = line.split('\n')[0].split(',')[1:]
        gene_list = set(gene_list).intersection(set(go_emb.keys()).union(archs4_emb.keys()))
        #gene_list = sample(gene_list, min(len(gene_list), 6))
        #print(tissue, gene_list)
        if tissue not in gene_lists:
            gene_lists[tissue] = [list(gene_list)]
        else:
            gene_lists[tissue].append(list(gene_list))
        all_genes.extend(gene_list)
all_genes = list(set(all_genes))

#Generate the t-SNE plot
def classify(X, y, output, method):
    plt.clf()
    plt.figure(figsize=(4,4))
    plt.axis('equal')
    model = TSNE(n_components=2, random_state=0)
    tsne_pj = model.fit_transform(X)
    for tissue in gene_lists:
        idx = np.where(y == tissue)[0]
        plt.scatter(tsne_pj[idx,0], tsne_pj[idx,1], label=tissue)
    plt.legend()
    plt.savefig(output)
    plt.title(method)
    plt.savefig(output, bbox_inches='tight')

    #Build the random forest classifer. Using 80% genes for training and 20% genes for test
    acc = []
    for i in range(100):
        train_index = []
        test_index = []
        for tissue in set(y):
            idx = np.where(y == tissue)[0]
            np.random.shuffle(idx)
            train_index.extend(idx[:int(0.1 * len(idx))])
            test_index.extend(idx[int(0.1 * len(idx)):])
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
        clf.fit(X[train_index], y[train_index])
        y_hat = clf.predict(X[test_index])
        acc.append((y_hat==y[test_index]).sum()/len(y_hat))
        print(method, 'repeat', i, 'acc', (y_hat==y[test_index]).sum()/len(y_hat))
    print(method, 'Accuracy on test:', np.mean(acc), 'std', np.std(acc))

#Generate gene list embeddings and assign labels to lists
mean_std_dict_go, mean_std_dict_archs4 = gene_sim_mean_std(go_emb, archs4_emb)
X = []
y = []
for i,tissue in enumerate(gene_lists):
    for glist in gene_lists[tissue]:
        emb = compute_list_emb(glist, go_emb, archs4_emb, mean_std_dict_go, mean_std_dict_archs4)
        X.append(emb)
        y.append(tissue)
X = np.array(X)
y = np.array(y)
print(X.shape)
classify(X, y, 'tsne_signature_FRoGS.png', 'FRoGS')

X = []
y = []
one_hot = np.diag(np.ones(len(all_genes)))
gene_encode = {gene:one_hot[i] for i,gene in enumerate(all_genes)}
for i,tissue in enumerate(gene_lists):
    for glist in gene_lists[tissue]:
        vec = np.zeros(len(all_genes))
        for gene in glist:
            vec += gene_encode[gene]
        X.append(vec)
        y.append(tissue)
X = np.array(X)
y = np.array(y)
classify(X, y, 'tsne_signature_onehot.png', 'One-hot')
