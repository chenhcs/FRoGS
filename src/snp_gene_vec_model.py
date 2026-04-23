"""
SNP-based Gene Vector Model for FRoGS
Extends gene_vec_model.py to support SNP association data for training gene embeddings.
"""

import os
# Fix TensorFlow threading issues - must be set before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import numpy as np
import tensorflow as tf
import gc

# Additional TensorFlow thread configuration
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from utils.random_walk import random_walk_w_restart as rwr
from utils.sampling_util import rw_sampling
from tensorflow.keras import layers, losses
from tensorflow import keras
from tensorflow.keras.models import Model
import argparse
import pandas as pd
from utils.snp_utils import SNPProcessor, create_snp_associations
from utils.snp_validation import SNPValidator
import logging

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train gene embeddings with SNP data support')
    parser.add_argument('--datatype', default='go',
                        help='Type of data: go, archs4, or snp')
    parser.add_argument('--association_file', default='../data/go_gene_association.txt',
                        help='Path to gene associations file')
    parser.add_argument('--snp_file', default=None,
                        help='Path to SNP data file (required for SNP datatype)')
    parser.add_argument('--snp_format', default='auto',
                        help='SNP file format: vcf, bed, list, csv, or auto')
    parser.add_argument('--outfile', default='../data/gene_vec_go.csv',
                        help='Path to save learned embeddings')
    parser.add_argument('--validate_snps', action='store_true',
                        help='Validate SNP data before processing')
    parser.add_argument('--pvalue_threshold', type=float, default=5e-8,
                        help='P-value threshold for SNP filtering')
    parser.add_argument('--effect_threshold', type=float, default=0.01,
                        help='Effect size threshold for SNP filtering')
    return parser.parse_args()

def get_model(vocab_size, latent_dim, ori_dim):
    """Build the neural embedding model."""
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

def process_snp_data_for_training(snp_file: str, snp_format: str = 'auto',
                                 validate: bool = True,
                                 pvalue_threshold: float = 5e-8,
                                 effect_threshold: float = 0.01) -> str:
    """
    Process SNP data and create gene associations file for training.
    
    Args:
        snp_file: Path to SNP data file
        snp_format: SNP file format
        validate: Whether to validate SNP data
        pvalue_threshold: P-value threshold for filtering
        effect_threshold: Effect size threshold for filtering
    
    Returns:
        Path to created associations file
    """
    logger.info(f"Processing SNP data from {snp_file}")
    
    # Load SNP data
    processor = SNPProcessor()
    
    if snp_format == 'vcf':
        snp_data = processor.load_vcf_file(snp_file)
    elif snp_format == 'bed':
        snp_data = processor.load_bed_file(snp_file)
    elif snp_format in ['list', 'csv']:
        snp_data = processor.load_custom_snp_file(snp_file, snp_format)
    else:  # auto-detect
        from utils.snp_utils import load_snp_data
        snp_data = load_snp_data(snp_file, snp_format)
    
    # Validate SNP data if requested
    if validate:
        validator = SNPValidator()
        
        # Generate QC report
        qc_report = validator.generate_qc_report(snp_data, '../results/snp_qc_report.txt')
        logger.info("SNP QC report generated")
        
        # Filter by p-value and effect size if columns are available
        from utils.snp_validation import filter_snps_by_criteria
        
        pvalue_cols = [col for col in ['PVALUE', 'P', 'P_VALUE'] if col in snp_data.columns]
        effect_cols = [col for col in ['EFFECT', 'BETA', 'OR'] if col in snp_data.columns]
        
        if pvalue_cols or effect_cols:
            pvalue_col = pvalue_cols[0] if pvalue_cols else 'PVALUE'
            effect_col = effect_cols[0] if effect_cols else 'EFFECT'
            
            original_count = len(snp_data)
            snp_data = filter_snps_by_criteria(snp_data, 
                                             pvalue_threshold=pvalue_threshold,
                                             effect_threshold=effect_threshold,
                                             pvalue_column=pvalue_col,
                                             effect_column=effect_col)
            logger.info(f"Filtered SNPs: {len(snp_data)}/{original_count} retained")
    
    # Create gene associations file
    associations_file = '../data/snp_gene_association.txt'
    processor.create_snp_gene_associations(snp_data, associations_file)
    
    return associations_file

def train_snp_embeddings(args):
    """Train gene embeddings using SNP association data."""
    
    if args.datatype == 'snp':
        if not args.snp_file:
            raise ValueError("SNP file must be provided when datatype is 'snp'")
        
        # Process SNP data to create associations file
        association_file = process_snp_data_for_training(
            args.snp_file, 
            args.snp_format,
            args.validate_snps,
            args.pvalue_threshold,
            args.effect_threshold
        )
        
        # Update output filename to include SNP suffix
        if args.outfile == '../data/gene_vec_go.csv':
            args.outfile = '../data/gene_vec_snp_256.csv'
        
    else:
        association_file = args.association_file
    
    # Use original FRoGS training pipeline
    logger.info(f"Training embeddings with data type: {args.datatype}")
    
    geneids, association_mat, diffusion_state = rwr(args.datatype, association_file)
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
        print(f'epoch: {e}')
        left_input = []
        right_input = []
        targets = []
        pos_pairs = 1
        neg_pairs = 5
        
        # Sample training pairs
        left_input, right_input, targets = rs.sampling(np.arange(len(geneids)), pos_pairs, neg_pairs)

        left_input = np.squeeze(np.array(left_input))
        right_input = np.squeeze(np.array(right_input))
        targets = np.squeeze(np.array(targets))
        sample_size = len(left_input)
        sample_idx = np.arange(sample_size)
        np.random.shuffle(sample_idx)
        left_input = left_input[sample_idx]
        right_input = right_input[sample_idx]
        targets = targets[sample_idx]

        autoencoder.fit([left_input[:int(0.8*sample_size)] * 1.0, right_input[:int(0.8*sample_size)] * 1.0], 
                       [targets[:int(0.8*sample_size)]],
                       batch_size=1000,
                       epochs=1,
                       verbose=1,
                       validation_data=([left_input[int(0.8*sample_size):] * 1.0, 
                                       right_input[int(0.8*sample_size):] * 1.0], 
                                      [targets[int(0.8*sample_size):]]))
        gc.collect()

    # Save embeddings
    encoded_genes = encoder.predict(np.arange(len(geneids)) * 1.0)
    
    with open(args.outfile, 'w') as fw:
        for i in range(len(encoded_genes)):
            fw.write(geneids[i])
            for j in range(len(encoded_genes[0])):
                fw.write(',' + str(encoded_genes[i, j]))
            fw.write('\n')
    
    logger.info(f"SNP-based gene embeddings saved to {args.outfile}")
    
    return args.outfile

if __name__ == "__main__":
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    if args.datatype == 'snp':
        # Train SNP-based embeddings
        embedding_file = train_snp_embeddings(args)
        print(f"SNP-based gene embeddings trained and saved to: {embedding_file}")
        
        # Demonstrate usage with signature embedding
        print("\nTesting SNP signature embedding...")
        from snp_signature_embedding import SNPSignatureEmbedder
        
        # Load the newly trained embeddings
        embedder = SNPSignatureEmbedder(go_emb_file=embedding_file, 
                                      archs4_emb_file='../data/gene_vec_archs4_256.csv')
        
        # Test with example SNPs
        test_snps = ['rs2230317', 'rs10273927', 'rs7398691']
        test_effects = {'rs2230317': 0.05, 'rs10273927': -0.03, 'rs7398691': 0.08}
        
        embedding = embedder.compute_snp_signature_embedding(test_snps, test_effects)
        print(f"Test signature embedding shape: {embedding.shape}")
        print(f"Sample embedding values: {embedding[:5]}")
        
    else:
        # Use original training pipeline for GO/ARCHS4 data
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
            
            # Sample training pairs
            left_input, right_input, targets = rs.sampling(np.arange(len(geneids)), pos_pairs, neg_pairs)

            left_input = np.squeeze(np.array(left_input))
            right_input = np.squeeze(np.array(right_input))
            targets = np.squeeze(np.array(targets))
            sample_size = len(left_input)
            sample_idx = np.arange(sample_size)
            np.random.shuffle(sample_idx)
            left_input = left_input[sample_idx]
            right_input = right_input[sample_idx]
            targets = targets[sample_idx]

            autoencoder.fit([left_input[:int(0.8*sample_size)] * 1.0, right_input[:int(0.8*sample_size)] * 1.0], 
                           [targets[:int(0.8*sample_size)]],
                           batch_size=1000,
                           epochs=1,
                           verbose=1,
                           validation_data=([left_input[int(0.8*sample_size):] * 1.0, 
                                           right_input[int(0.8*sample_size):] * 1.0], 
                                          [targets[int(0.8*sample_size):]]))
            gc.collect()

        encoded_genes = encoder.predict(np.arange(len(geneids)) * 1.0)
        
        with open(outfile, 'w') as fw:
            for i in range(len(encoded_genes)):
                fw.write(geneids[i])
                for j in range(len(encoded_genes[0])):
                    fw.write(',' + str(encoded_genes[i, j]))
                fw.write('\n')
        
        print(f"Gene embeddings saved to: {outfile}")
