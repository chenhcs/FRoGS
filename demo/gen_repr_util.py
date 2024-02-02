import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def gene_sim_mean_std(go_emb, archs4_emb):
    '''Compute the mean and standard diviation of each gene's cosine
       similarities with all the genes in the dataset
    '''
    all_genes=set(list(go_emb.keys()) + list(archs4_emb.keys()))

    def sim_mean_std(emb):
        emb_mat=[]
        gene_ids=[]
        for i,gid in enumerate(all_genes):
            if gid in emb:
                emb_mat.append(emb[gid])
                gene_ids.append(gid)

        emb_mat=np.vstack(emb_mat)
        sim_pairwise=cosine_similarity(emb_mat, emb_mat)
        sim_mean=np.mean(sim_pairwise, axis=1)
        sim_std=np.std(sim_pairwise, axis=1)

        mean_std_dict={}
        for i,gid in enumerate(gene_ids):
            mean_std_dict[gid]=(sim_mean[i], sim_std[i])

        return mean_std_dict

    return sim_mean_std(go_emb), sim_mean_std(archs4_emb)

def compute_list_emb(glist, go_emb, archs4_emb, mean_std_dict_go, mean_std_dict_archs4):
    '''Compute the vector representation (embedding) for an input gene list
    '''
    def compute_gene_weight(mat, sim_mean, sim_std):
        sim_pairwise=cosine_similarity(mat, mat)
        sim_pairwise[np.isnan(sim_pairwise)]=0
        sim_sum=np.mean(sim_pairwise, axis=1)
        #Z-score nomarlize the average similarity with all the genes in the gene list
        gene_weight=(sim_sum - sim_mean) / sim_std
        return gene_weight

    go_mat=np.zeros([len(glist), len(list(go_emb.values())[0])])
    for i,gid in enumerate(glist):
        if gid in go_emb:
            go_mat[i,:]=go_emb[gid]
        else:
            go_mat[i,:]=np.zeros(len(list(go_emb.values())[0]))

    archs4_mat=np.zeros([len(glist), len(list(archs4_emb.values())[0])])
    for i,gid in enumerate(glist):
        if gid in archs4_emb:
            archs4_mat[i,:]=archs4_emb[gid]
        else:
            archs4_mat[i,:]=np.zeros(len(list(archs4_emb.values())[0]))

    #l2 normalize each embedding vector
    go_mat = normalize(go_mat)
    archs4_mat = normalize(archs4_mat)

    sim_mean_go=np.array([mean_std_dict_go[gid][0] if gid in mean_std_dict_go else 0 for gid in glist])
    sim_std_go=np.array([mean_std_dict_go[gid][1] if (gid in mean_std_dict_go and mean_std_dict_go[gid][1] > 0) else float('inf') for gid in glist])
    sim_mean_archs4=np.array([mean_std_dict_archs4[gid][0] if gid in mean_std_dict_archs4 else 0 for gid in glist])
    sim_std_archs4=np.array([mean_std_dict_archs4[gid][1] if (gid in mean_std_dict_archs4 and mean_std_dict_archs4[gid][1] > 0) else float('inf') for gid in glist])

    #weight each gene based on its average similarity with all the genes in the gene list
    #compute weights in terms of go and archs4 respectively
    gene_weight_go=compute_gene_weight(go_mat, sim_mean_go, sim_std_go)
    gene_weight_a4chs4=compute_gene_weight(archs4_mat, sim_mean_archs4, sim_std_archs4)
    #for each gene, take the largest weight from go and archs4
    gene_weight=np.maximum(gene_weight_go, gene_weight_a4chs4)
    gene_weight=np.clip(gene_weight, 0, 1)
    gene_weight=gene_weight.reshape((-1, 1))

    concatenated_mat=np.hstack((go_mat, archs4_mat))

    return np.sum(concatenated_mat * gene_weight, axis=0) / np.clip(np.sum(gene_weight), 1e-100, None)
