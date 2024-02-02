import numpy as np
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gen_repr_util import gene_sim_mean_std, compute_list_emb

with open('../data/gene_vec_go_256.csv', mode='r') as infile:
    reader = csv.reader(infile)
    go_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

with open('../data/gene_vec_archs4_256.csv', mode='r') as infile:
    reader = csv.reader(infile)
    archs4_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

gene_lists = {}
with open('data/gene_lists_tissue.txt') as fr:
    for line in fr:
        tissue = line.split(',')[0]
        gene_list = line.split('\n')[0].split(',')[1:]
        gene_list = set(gene_list).intersection(set(go_emb.keys()).union(archs4_emb.keys()))
        if tissue not in gene_lists:
            gene_lists[tissue] = [list(gene_list)]
        else:
            gene_lists[tissue].append(list(gene_list))

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

#Generate the t-SNE plot
model = TSNE(n_components=2, random_state=0)
tsne_pj = model.fit_transform(X)
for tissue in gene_lists:
    idx = np.where(y == tissue)[0]
    plt.scatter(tsne_pj[idx,0], tsne_pj[idx,1], label=tissue)
plt.legend()
plt.savefig('tsne_gene_signature.png')
