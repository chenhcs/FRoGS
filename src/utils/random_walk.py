import numpy as np

def run_rwr_diffusion(P, reset_prob, epsilon=1e-6, maxiter=100):
    reset = np.eye(np.shape(P)[0])
    Q = reset
    for i in range(maxiter):
        Q_new = reset_prob * reset + (1 - reset_prob) * np.dot(Q, P)
        delta = np.linalg.norm(Q - Q_new, ord='fro')
        print('Iter {} Frobenius norm: {}'.format(i, delta))
        Q = Q_new
        if delta < epsilon:
            print('Converged.')
            break
    print("Check Q:", np.sum(Q, axis=1))
    return Q

def compute_transition_p(association, edge_weight):
    node_degree = np.dot(association, edge_weight)
    edge_cap = np.sum(association, axis=0)
    P = np.dot(association * edge_weight / np.expand_dims(node_degree, axis=1), association.transpose() / np.expand_dims(edge_cap, axis=1))
    P[np.where(np.isnan(P))] = 0
    for i, s in enumerate(np.sum(P, axis=1)):
        if s == 0:
            P[i, i] = 1.
    print("Check P:", np.sum(P, axis=1), np.sum(P))
    return P

def go_ic(C_GO, association):
    goidx = {go:i for i,go in enumerate(C_GO)}

    from goatools.base import get_godag
    from goatools.gosubdag.gosubdag import GoSubDag
    godag = get_godag("../data/go.obo")

    freq = np.sum(association, axis=0)
    freq_copy = np.sum(association, axis=0)
    go_ = [go.split('19_')[1] for go in C_GO]
    gosubdag_r0 = GoSubDag(go_, godag, prt=None)
    for go in C_GO:
        if go.split('19_')[1] in gosubdag_r0.rcntobj.go2parents:
            for pgo in gosubdag_r0.rcntobj.go2parents[go.split('19_')[1]]:
                if '19_' + pgo in C_GO:
                    freq_copy[goidx['19_' + pgo]] += freq[goidx[go]]
    print(freq_copy / np.sum(freq))
    ic = -np.log2(freq_copy / np.sum(freq))
    ic = np.power(ic, 2)
    print(freq)
    print(freq_copy)
    ic[np.where(freq_copy == 0)] = 0
    return ic

def read_association_file(genesetsf):
    geneidx = {}
    genecnt = 0
    geneids = []
    cat2gene = {}
    with open(genesetsf) as fr:
        for line in fr:
            catid = line.split('\t')[0]
            cat2gene[catid] = []
            for gene in line.split('\n')[0].split('\t')[2].split(','):
                cat2gene[catid].append(gene)
                if gene not in geneidx:
                    geneidx[gene] = genecnt
                    geneids.append(gene)
                    genecnt += 1

    m = np.zeros([len(geneids), len(cat2gene)])
    catids = list(cat2gene.keys())
    for i,cat in enumerate(catids):
        for gene in cat2gene[cat]:
            m[geneidx[gene], i] = 1
    return m, catids, geneids

def random_walk_w_restart(datatype, association_file):
    association, catids, geneids = read_association_file(association_file)
    if datatype == 'go':
        edge_weight = go_ic(catids, association)
    elif datatype == 'archs4':
        edge_weight = 1. / np.sum(association, axis=0)
    print('Compute transition probability matrix...')
    P = compute_transition_p(association, edge_weight)
    print('Compute diffusion state matrix...')
    Q = run_rwr_diffusion(P, reset_prob=0.9)

    return geneids, association, Q
