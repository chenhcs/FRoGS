import numpy as np
import tensorflow as tf
import gc
from utils.random_walk import random_walk_w_restart as rwr
from utils.sampling_util import rw_sampling
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers, losses
from tensorflow import keras
from tensorflow.keras.models import Model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train gene embeddings')
    parser.add_argument('--datatype', default='go',
                        help='Type of data from which the gene embeddings learned. go or archs4')
    parser.add_argument('--association_file', default='../data/go_gene_association.txt',
                        help='Path to a file containing GO term gene associations or ARCHS4 experiments gene associations')
    parser.add_argument('--outfile', default='../data/gene_vec_go.csv',
                        help='Path to save the learned embeddings')
    return parser.parse_args()

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

args = parse_args()
datatype, association_file, outfile = args.datatype, args.association_file, args.outfile

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
