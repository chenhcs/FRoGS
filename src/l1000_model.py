import sys
import pandas as pd
import numpy as np
import os
import glob
import math
import csv
import tensorflow as tf
from utils import parallel
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers, losses
from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train l1000 model')
    parser.add_argument('--cpdlist_file', default='../data/compound_list_shRNA.txt',
                        help='Path to a file that lists the compounds for cross validation')
    parser.add_argument('--target_file', default='../data/cpd_gene_pairs.csv',
                        help='Path to a file containing target annotations of compounds')
    parser.add_argument('--sig_file', default='../data/L1000_PhaseI_and_II.csv',
                        help='Path to a file containing L1000 gene signatures')
    parser.add_argument('--perttype', default='shRNA',
                        help='Perturbagen type of gene signatures to use. cDNA or shRNA')
    parser.add_argument('--emb_go', default='../data/gene_vec_go_256.csv',
                        help='Gene embeddings learned from GO annotations')
    parser.add_argument('--emb_archs4', default='../data/gene_vec_archs4_256.csv',
                        help='Gene embeddings learned from archs4 gene expression experiments')
    parser.add_argument('--epochs', default=60,
                        help='Number of training epochs')
    parser.add_argument('--outdir', default='../results/',
                        help='Path to a directory to save prediction lists')
    parser.add_argument('--modeldir', default='../saved_model/',
                        help='Path to a directory to save trained models')
    return parser.parse_args()


def get_model(fp_dim, hid_dim = 2048):
    input_l = keras.Input(shape=(fp_dim,), name="cpd_fp_l")
    input_r = keras.Input(shape=(fp_dim,), name="target_fp_r")
    denselayers = keras.Sequential([
        layers.Dropout(0.25),
        layers.Dense(hid_dim),
        layers.BatchNormalization(),
        layers.ReLU()
    ])
    denseout_cpd = denselayers(input_l)
    denseout_target = denselayers(input_r)

    merge = layers.Multiply()([denseout_cpd, denseout_target])
    denseclassifier = keras.Sequential([
        layers.Dense(hid_dim/4),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(1)
    ])

    logit = denseclassifier(merge)

    classifier = keras.Model(inputs=[input_l, input_r], outputs=[logit], name="classifier")
    classifier.summary()

    return classifier

def gene_sim_mean_std():
    '''Compute the mean and standard diviation of each gene's cosine
       similarities with all the genes in the dataset
       return: a mapping from each gene ID to the mean of its
               similarities with all the genes in the dataset;
               a mapping from each gene ID to the standard diviation of its
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

def get_vec(X):
    sig2vec = {}
    l1k, S_hit = X
    vec = compute_list_emb(S_hit)
    sig2vec[l1k] = vec
    return sig2vec

def prepare_data(cpd_train, prt):
    positive_l = []
    positive_r = []
    negative_l = []
    negative_r = []
    pos_freq = {}
    neg_add = set()

    #Positive compound target pairs
    tgt_cl_idx = {cl:tgt_sig.index for cl, tgt_sig in pert_sig[pert_sig['Perturbagen']==perttype].groupby('CellLine')}
    sig_name_cpd = {cpd:cpd_sig for cpd, cpd_sig in pert_sig[pert_sig['Perturbagen']=='Cpd'].groupby('Name')}
    for k,cpd in enumerate(cpd_train):
        targets = set(cpd2target[cpd])
        if cpd not in sig_name_cpd:
            continue
        cpd_sig = sig_name_cpd[cpd]
        cpd_sig_cl_df = {cl:cpd_sig_cl for cl, cpd_sig_cl in cpd_sig.groupby('CellLine')}
        for cl in cpd_sig_cl_df:
            cpd_sig_cl = cpd_sig_cl_df[cl]
            if len(cpd_sig_cl) == 0:
                continue
            if cl not in tgt_cl_idx:
                continue

            for cidx in cpd_sig_cl.index:
                for tidx in tgt_cl_idx[cl]:
                    if pert_sig.loc[tidx, 'Name'] in targets:
                        positive_l.append(cidx)
                        positive_r.append(tidx)
                        if tidx not in pos_freq:
                            pos_freq[tidx] = 1
                        else:
                            pos_freq[tidx] += 1

    #Sample negative pairs and make sure the ratio of each target in positive pairs and negative pairs is 1:1
    cpd_cl_dic = {cl:cpd_sig for cl, cpd_sig in pert_sig[pert_sig['Perturbagen']=='Cpd'].groupby('CellLine')}
    tgt_sig = pert_sig[pert_sig['Perturbagen']==perttype].copy()
    tgt_sig_cl_df = {cl:tgt_sig_cl for cl, tgt_sig_cl in tgt_sig.groupby('CellLine')}
    cl_all = list(tgt_sig_cl_df.keys())
    for cl in cl_all:
        tgt_sig_cl = tgt_sig_cl_df[cl]
        if cl not in cpd_cl_dic:
            continue
        cpd_cl = cpd_cl_dic[cl]
        for tidx in tgt_sig_cl.index:
            if tidx in pos_freq and pos_freq[tidx] > 0:
                np.random.shuffle(cpd_train)
                for cpd in cpd_train:
                    targets = set(cpd2target[cpd])
                    if pert_sig.loc[tidx, 'Name'] not in targets:
                        for cidx in cpd_cl[cpd_cl['Name']==cpd].index:
                            pos_freq[tidx] -= 1
                            negative_l.append(cidx)
                            negative_r.append(tidx)
                            if pos_freq[tidx] <= 0:
                                break
                    if pos_freq[tidx] <= 0:
                        break

    print('#Positive training pairs:', len(positive_l), '#Negative training pairs:', len(negative_l))

    train_input_idx_l = positive_l.copy()
    train_input_idx_r = positive_r.copy()
    train_label = [1] * len(train_input_idx_l)
    train_input_idx_l.extend(negative_l)
    train_input_idx_r.extend(negative_r)
    train_label.extend([0] * len(negative_l))
    train_input_l = sigvec_all[train_input_idx_l]
    train_input_r = sigvec_all[train_input_idx_r]
    train_label = np.array(train_label)

    return train_input_l, train_input_r, train_label

def inference_testset(classifier, cpd_test):
    intop10 = 0
    intop100 = 0
    tgtcl_idx = {}
    for cl in all_cl:
        tgt_sig = pert_sig[pert_sig['Perturbagen']==perttype].copy()
        tgt_sig = tgt_sig[tgt_sig['CellLine']==cl]
        tgtcl_idx[cl] = tgt_sig.index
    for k,cpd in enumerate(cpd_test):
        print(k, len(cpd_test), cpd)
        cpd_sig = pert_sig[pert_sig['Name']==cpd].copy()
        cpd_sig_cl_df = {cl:cpd_sig_cl for cl, cpd_sig_cl in cpd_sig.groupby('CellLine')}
        for cl in cpd_sig_cl_df:
            cpd_sig_cl = cpd_sig_cl_df[cl]
            if len(cpd_sig_cl) == 0:
                continue

            if len(set(cpd2target[cpd]).intersection(pert_sig.loc[tgtcl_idx[cl], 'Name'].tolist())) > 0:
                test_input_idx_l = []
                test_input_idx_r = []
                tgt_idx = {}
                for cidx in cpd_sig_cl.index:
                    test_input_idx_l.extend([cidx] * len(tgtcl_idx[cl]))
                test_input_idx_r.extend(list(tgtcl_idx[cl]) * len(cpd_sig_cl.index))
                scnt = 0
                for tidx in tgtcl_idx[cl]:
                    if pert_sig.loc[tidx, 'Name'] not in tgt_idx:
                        tgt_idx[pert_sig.loc[tidx, 'Name']] = [scnt + i * len(tgtcl_idx[cl]) for i in range(len(cpd_sig_cl.index))]
                    else:
                        tgt_idx[pert_sig.loc[tidx, 'Name']].extend([scnt + i * len(tgtcl_idx[cl]) for i in range(len(cpd_sig_cl.index))])
                    scnt += 1


                test_input_l = sigvec_all[test_input_idx_l]
                test_input_r = sigvec_all[test_input_idx_r]
                scores = classifier.predict([test_input_l, test_input_r])
                scores = 1 / (1 + np.exp(-scores))
                genes = []
                max_scores = []
                for k in tgt_idx:
                    genes.append(k)
                    max_scores.append(np.max(scores[tgt_idx[k]]))
                max_scores = np.array(max_scores)
                genes = np.array(genes)
                sortidx = np.argsort(-max_scores)
                topgenes = genes[sortidx]
                rankedscores = max_scores[sortidx]
                top10 = topgenes[:10].copy()
                top100 = topgenes[:100].copy()
                fw = open(outdir + cpd + cl + '_' + perttype + '.txt', 'w')
                fw.write('gene\trank\tscore\n')
                for i in range(len(topgenes)):
                    fw.write(topgenes[i].split('@')[0] + '\t' + str(i+1) + '\t' + str(rankedscores[i]) + '\n')
                fw.close()
                for t in cpd2target[cpd]:
                    if t in top10:
                        intop10 += 1
                    if t in top100:
                        intop100 += 1
                print(cl, 'top10:', intop10, 'top100:', intop100)

def compute_list_emb(glist):
    '''Compute the vector representation (embedding) for an input gene list
       input: list of gene IDs
       return: a 512d vector reprsentation
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

args = parse_args()
cpdlist_file, target_file, sig_file, perttype, emb_go, emb_archs4, epochs, outdir, modeldir = args.cpdlist_file, args.target_file, args.sig_file, args.perttype, args.emb_go, args.emb_archs4, args.epochs, args.outdir, args.modeldir

with open(emb_archs4, mode='r') as infile:
    reader = csv.reader(infile)
    archs4_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

with open(emb_go, mode='r') as infile:
    reader = csv.reader(infile)
    go_emb = {rows[0]:np.array(rows[1:], dtype=np.float32) for rows in reader}

mean_std_dict_go, mean_std_dict_archs4 = gene_sim_mean_std()
t_target=pd.read_csv(target_file)
cpd2target={}
target_gene_id = t_target.Broad_target_gene_id.tolist()

for k,v in zip(t_target.term_name.tolist(), target_gene_id):
    if str(v) == 'nan':
        continue
    ck = k.split('Cpd:')[1].split(':')[0] + '@'
    for t in v.split(':'):
        t = t+'@'
        if ck not in cpd2target:
            cpd2target[ck] = [t]
        else:
            if t not in cpd2target[ck]:
                cpd2target[ck].append(t)
    cpd2target[ck] = list(set(cpd2target[ck]))

delk = []
for k in cpd2target:
    if len(cpd2target[k]) > 5:
        delk.append(k)
for k in delk:
    del cpd2target[k]

target_num = []
for c in cpd2target:
    target_num.append(len(cpd2target[c]))
print('Average tragets per compound:', np.mean(target_num))


#read gene signature file
id2sig = {}
t_sig=pd.read_csv(sig_file)
pert_sig = []
tasks = []
cnt = 0
targetsymbols = set()
all_cl = set()
#parse compound gene signature
for idx in t_sig.index:
    des = t_sig.loc[idx, 'term_name']
    ty = t_sig.loc[idx, 'term_name'].split(':')[0]
    if ty != 'Cpd':
        continue
    name = t_sig.loc[idx, 'term_name'].split(':')[1].split(':')[0] + '@'
    cl = t_sig.loc[idx, 'term_name'].split('@')[1].split('@')[0]
    dir = t_sig.loc[idx, 'term_name'].split('(')[1].split(')')[0]
    S_hit = set(t_sig.loc[idx, 'gids'].split(","))
    l1k = t_sig.loc[idx, 'term_id']

    if name in cpd2target:
        pert_sig.append([l1k, name, cl, dir, des, ty, S_hit])
        id2sig[l1k] = S_hit
        tasks.append((l1k, S_hit))
        all_cl.add(cl)

id_map=pd.read_csv('../data/term2gene_id.csv')
term2geneid = {}
for idx in id_map.index:
    term2geneid[id_map.loc[idx, 'term_name']] = str(id_map.loc[idx, 'gene_id'])

#parse target gene signature
for idx in t_sig.index:
    des = t_sig.loc[idx, 'term_name']
    ty = t_sig.loc[idx, 'term_name'].split(':')[0]
    if ty not in ['shRNA', 'cDNA']:
        continue
    if t_sig.loc[idx, 'term_name'] in term2geneid:
        name = term2geneid[t_sig.loc[idx, 'term_name']] + '@'
    else:
        continue
    cl = t_sig.loc[idx, 'term_name'].split('@')[1].split('@')[0]
    if cl not in all_cl:
        continue
    dir = t_sig.loc[idx, 'term_name'].split('(')[1].split(')')[0]
    S_hit = set(t_sig.loc[idx, 'gids'].split(","))
    l1k = t_sig.loc[idx, 'term_id']
    pert_sig.append([l1k, name, cl, dir, des, ty, S_hit])
    id2sig[l1k] = S_hit
    tasks.append((l1k, S_hit))
pert_sig=pd.DataFrame(pert_sig, columns=['l1k', 'Name', 'CellLine', 'Direction', 'Description', 'Perturbagen', 'Signature'])
print('Number of perturbagen signatures:', len(pert_sig))
print('Numer of cell lines:', len(all_cl))

print("Compute embeddings of gene signatures...")
sig2vec_dic = {}
rslt=parallel.map(get_vec, tasks, n_CPU=10, progress=False)
for dic in rslt:
    for k in dic:
        sig2vec_dic[k] = dic[k]
sigvec_all = []
for idx in pert_sig.index:
    sigvec_all.append(sig2vec_dic[pert_sig.loc[idx, 'l1k']])
sigvec_all = np.array(sigvec_all)

cpd_list = []
with open(cpdlist_file) as fr:
    for line in fr:
        cpd_list.append(line.split('\n')[0] + '@')
fr.close()
part = []
for testpar in range(0, 5):
    part.append([i for i in range(int(len(cpd_list) * 0.2 * testpar),int(len(cpd_list) * 0.2 * (testpar + 1)))])

#cross validation
for testpar in range(0, 5):
    trainidx = []

    fp_dim = 512

    for i in range(5):
        if i != testpar:
            trainidx += part[i]
    testidx = part[testpar]
    cpd_list = np.array(cpd_list)
    cpd_train = cpd_list[trainidx]
    cpd_test = cpd_list[testidx]
    print('#Train cpd:', len(cpd_train), '#Test cpd:', len(cpd_test))
    input_l, input_r, train_label = prepare_data(cpd_train, testpar)

    classifier= get_model(fp_dim)
    optimizer = keras.optimizers.Adam()
    classifier.compile(
        optimizer=optimizer,
        loss=[losses.BinaryCrossentropy(from_logits=True)]
    )

    classifier.fit([input_l, input_r], [train_label],
        shuffle=True,
        batch_size=1000,
        epochs=epochs,
        verbose=1)

    classifier.save_weights(modeldir + 'model_' + perttype + '_part_' + str(testpar) + '/ckpt')

    inference_testset(classifier, cpd_test)
