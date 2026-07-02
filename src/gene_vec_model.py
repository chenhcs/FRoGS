import numpy as np
import tensorflow as tf
import gc
import os
import shutil
import pickle
import sys
from types import SimpleNamespace
from utils.random_walk import random_walk_w_restart as rwr
from utils.sampling_util import rw_sampling
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers, losses
from tensorflow import keras
from tensorflow.keras.models import Model

def parse_args(argv=None):
    '''Parse CLI arguments via a manual sys.argv scan.

    Mirrors the argument-parsing style used by graphify's `args-demo`
    command: each flag is checked explicitly with a cursor loop, and both
    "--flag value" and "--flag=value" forms are supported.
    '''
    argv = sys.argv[1:] if argv is None else argv

    if '-h' in argv or '--help' in argv:
        print('Usage: gene_vec_model.py [--datatype go|archs4] [--association_file PATH]')
        print('                          [--outfile PATH] [--LR_model DIR] [--demo DIR]')
        print()
        print('  --datatype TYPE          type of data the embeddings are learned from:')
        print('                           go or archs4 (default: go)')
        print('  --association_file PATH  GO/ARCHS4 gene association file')
        print('                           (default: ../data/go_gene_association.txt)')
        print('  --outfile PATH           path to save the learned embeddings')
        print('                           (default: ../data/gene_vec_go.csv)')
        print('  --LR_model DIR           directory to save or load the logistic regression model')
        print('  --demo DIR               demo directory to export the learned embeddings into')
        sys.exit(0)

    datatype = 'go'
    association_file = '../data/go_gene_association.txt'
    outfile = '../data/gene_vec_go.csv'
    LR_model = None
    demo = None

    i = 0
    while i < len(argv):
        if argv[i] == '--datatype' and i + 1 < len(argv):
            datatype = argv[i + 1]; i += 2
        elif argv[i].startswith('--datatype='):
            datatype = argv[i].split('=', 1)[1]; i += 1
        elif argv[i] == '--association_file' and i + 1 < len(argv):
            association_file = argv[i + 1]; i += 2
        elif argv[i].startswith('--association_file='):
            association_file = argv[i].split('=', 1)[1]; i += 1
        elif argv[i] == '--outfile' and i + 1 < len(argv):
            outfile = argv[i + 1]; i += 2
        elif argv[i].startswith('--outfile='):
            outfile = argv[i].split('=', 1)[1]; i += 1
        elif argv[i] == '--LR_model' and i + 1 < len(argv):
            LR_model = argv[i + 1]; i += 2
        elif argv[i].startswith('--LR_model='):
            LR_model = argv[i].split('=', 1)[1]; i += 1
        elif argv[i] == '--demo' and i + 1 < len(argv):
            demo = argv[i + 1]; i += 2
        elif argv[i].startswith('--demo='):
            demo = argv[i].split('=', 1)[1]; i += 1
        else:
            print(f'error: unrecognized argument: {argv[i]}', file=sys.stderr)
            sys.exit(1)

    return SimpleNamespace(
        datatype=datatype,
        association_file=association_file,
        outfile=outfile,
        LR_model=LR_model,
        demo=demo,
    )

def get_model(vocab_size, latent_dim, ori_dim):
    encoder_input_l = keras.Input(shape=(1,), name="geneidx_l")
    encoder_input_r = keras.Input(shape=(1,), name="geneidx_r")
    encoderlayers = layers.Embedding(vocab_size, latent_dim, input_length=1)
    encoder_output_l = keras.backend.squeeze(encoderlayers(encoder_input_l), axis=1)
    encoder_output_r = keras.backend.squeeze(encoderlayers(encoder_input_r), axis=1)

    encoder = keras.Model(inputs=encoder_input_l, outputs=encoder_output_l, name="encoder")
    encoder.summary()

    logit = layers.Dot(axes=1)([encoder_output_l, encoder_output_r])

    model = keras.Model(inputs=[encoder_input_l, encoder_input_r], outputs=[logit], name="autoencoder")
    model.summary()

    return encoder, model

def build_pair_features(encoded_genes, left_idx, right_idx):
    '''Build feature vectors for gene pairs from the learned embeddings.

    The feature for a pair is the element-wise product of the two genes'
    embedding vectors. ``left_idx`` and ``right_idx`` are arrays of gene
    indices into ``encoded_genes``.
    '''
    encoded_genes = np.asarray(encoded_genes)
    left_idx = np.asarray(left_idx, dtype=int)
    right_idx = np.asarray(right_idx, dtype=int)
    return encoded_genes[left_idx] * encoded_genes[right_idx]

def train_or_load_lr_model(X, y, lr_dir, filename='gene_pair_lr.pkl'):
    '''Train, or load if already present, a logistic regression model.

    The model predicts whether a pair of genes is associated based on the
    gene-pair features ``X`` and labels ``y``. It is persisted as a pickle
    file inside ``lr_dir``; if that file already exists it is loaded instead
    of being retrained. Returns a tuple ``(model, model_path)``.
    '''
    from sklearn.linear_model import LogisticRegression

    os.makedirs(lr_dir, exist_ok=True)
    lr_path = os.path.join(lr_dir, filename)

    if os.path.exists(lr_path):
        with open(lr_path, 'rb') as f:
            lr = pickle.load(f)
        print('Loaded LR model from', lr_path)
    else:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        with open(lr_path, 'wb') as f:
            pickle.dump(lr, f)
        print('Saved LR model to', lr_path)

    return lr, lr_path

def export_embeddings_for_demo(outfile, demo_dir):
    '''Copy the learned embeddings file into the demo directory.

    When ``demo_dir`` contains a ``data`` subdirectory the file is placed
    there (where the demo scripts look for it), otherwise it is copied into
    ``demo_dir`` directly. Returns the destination path.
    '''
    demo_data_dir = os.path.join(demo_dir, 'data')
    dest_dir = demo_data_dir if os.path.isdir(demo_data_dir) else demo_dir
    os.makedirs(dest_dir, exist_ok=True)
    demo_outfile = os.path.join(dest_dir, os.path.basename(outfile))
    shutil.copyfile(outfile, demo_outfile)
    print('Copied embeddings for demo to', demo_outfile)
    return demo_outfile

def main():
    args = parse_args()
    datatype, association_file, outfile = args.datatype, args.association_file, args.outfile
    LR_model, demo = args.LR_model, args.demo

    geneids, association_mat, diffusion_state = rwr(datatype, association_file)
    nonzeroidx = np.where(np.sum(association_mat, axis=1) > 0)[0]

    geneids = np.array(geneids)
    fp_array = association_mat[nonzeroidx]
    geneids = geneids[nonzeroidx]

    ori_dim = fp_array.shape[-1]
    latent_dim = 256

    encoder, autoencoder = get_model(len(geneids), latent_dim, ori_dim)
    optimizer = keras.optimizers.Adam()
    autoencoder.compile(
        optimizer=optimizer,
        loss=[losses.BinaryCrossentropy(from_logits=True)]
    )

    rs = rw_sampling(diffusion_state)
    epochs = 3000
    for e in range(epochs):
        print('epoch:', e)
        left_input = []
        right_input = []
        targets = []
        pos_pairs = 1
        neg_pairs = 5
        #sample training pairs
        left_input, right_input, targets = rs.sampling(np.arange(len(geneids)), pos_pairs, neg_pairs)

        left_input = np.squeeze(np.array(left_input))
        right_input = np.squeeze(np.array(right_input))
        targets = np.squeeze(np.array(targets))
        sample_size = len(left_input)
        smaple_idx = np.arange(sample_size)
        np.random.shuffle(smaple_idx)
        left_input = left_input[smaple_idx]
        right_input = right_input[smaple_idx]
        targets = targets[smaple_idx]

        autoencoder.fit([left_input[:int(0.8*sample_size)] * 1.0, right_input[:int(0.8*sample_size)] * 1.0], [targets[:int(0.8*sample_size)]],
                  batch_size=1000,
                  epochs=1,
                  verbose=1,
                  validation_data=([left_input[int(0.8*sample_size):] * 1.0, right_input[int(0.8*sample_size):] * 1.0], [targets[int(0.8*sample_size):]]))
        gc.collect()

    encoded_genes = encoder.predict(np.arange(len(geneids)) * 1.0)
    fw = open(outfile, 'w')
    for i in range(len(encoded_genes)):
        fw.write(geneids[i])
        for j in range(len(encoded_genes[0])):
            fw.write(',' + str(encoded_genes[i, j]))
        fw.write('\n')
    fw.close()

    # Train (or load) a logistic regression model that predicts whether a pair
    # of genes is associated, using the learned gene embeddings as features.
    # The model is persisted in the directory given by --LR_model.
    if LR_model is not None:
        X_lr = build_pair_features(encoded_genes, left_input, right_input)
        y_lr = np.asarray(targets, dtype=int)
        lr, _ = train_or_load_lr_model(X_lr, y_lr, LR_model)
        print('LR gene-pair prediction accuracy:', lr.score(X_lr, y_lr))

    # Export a copy of the learned embeddings into the demo directory so that
    # the demo scripts (e.g. classifier.py) can consume the trained embeddings.
    if demo is not None:
        export_embeddings_for_demo(outfile, demo)

if __name__ == '__main__':
    main()
