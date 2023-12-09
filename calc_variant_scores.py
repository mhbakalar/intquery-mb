import argparse
import numpy as np
import torch
import pysam
from lit_modules import data_modules, modules
import utils.fasta_data
import pandas as pd

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: A namespace containing the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Variant Sequence Scoring Script')
    parser.add_argument('--vcf_file', type=str, required=True, help='Path to VCF file')
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to FASTA file')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the trained model file')
    # parser.add_argument('--validated_sites', type=str, required=True, help='Validated sites')
    return parser.parse_args()

def load_model(model_path):
    """
    Load the trained model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = modules.Regression.load_from_checkpoint(model_path)
    print(model)
    return model

def read_vcf(vcf_file):
    """
    Reads variants from a VCF file.

    Args:
        vcf_file (str): Path to the VCF file.

    Returns:
        list: A list of variant tuples.
    """
    variants = []
    with open(vcf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            chrom, pos, ref, alts = parts[0], parts[1], parts[3], parts[4].split(',')
            if len(ref) == 1:
                for alt in alts:
                    if len(alt) == 1:
                        variants.append((chrom, pos, ref, alt))
    return variants

def process_variants(variants, fasta_file):
    """
    Processes variants to create a dictionary with sequences.

    Args:
        variants (list): List of variant tuples.
        fasta_file (str): Path to the FASTA file.

    Returns:
        dict: Dictionary with variant sequences.
    """
    fasta = pysam.FastaFile(fasta_file)
    alt_sequence_dict = {}
    for variant in variants:
        chrom, pos, ref, alt = variant
        pos = int(pos) - 1
        for i in range(-45, 1):
            start, end = pos + i, pos + i + 46
            sequence = fasta.fetch(chrom, start, end)
            ref_pos_in_seq = abs(i)
            original_sequence = (sequence[:ref_pos_in_seq] + ref + sequence[ref_pos_in_seq + 1:]).upper()
            modified_sequence = (sequence[:ref_pos_in_seq] + alt + sequence[ref_pos_in_seq + 1:]).upper()
            if i == -45:
                alt_sequence_dict[variant] = [(chrom, pos, original_sequence, modified_sequence)]
            else:
                alt_sequence_dict[variant].append((chrom, pos, original_sequence, modified_sequence))
    fasta.close()
    return alt_sequence_dict

def predict_validated_sites(model, validated_sites):
    model.eval()
    df = pd.read_csv(validated_sites)
    for index,row in df.iterrows():
        site = row['id']
        sequence = row['seq']
        one_hot_seq = utils.fasta_data.str_to_one_hot(sequence)
        one_hot_seq = one_hot_seq.unsqueeze(0)
        output = model(one_hot_seq).item()
        print(f'{site}:{output}')

def predict_variant_effects(model, alt_sequence_dict):
    """
    Predict the effects of variants using the model.

    Args:
        model (torch.nn.Module): The trained model.
        alt_sequence_dict (dict): Dictionary with variant sequences.
    """
    model.eval()
    cas031_sequence = 'AGTTGAGCCTTGAACAACAGGGNNTTCAACTGTGTGGATCCACTTA'
    test_sequence =   'AAAAAAAAAAAAAAAAAAAAAANNAAAAAAAAAAAAAAAAAAAAAA'
    one_hot_original_seq = utils.fasta_data.str_to_one_hot(cas031_sequence)
    one_hot_original_seq = one_hot_original_seq.unsqueeze(0)
    original_seq_output = model(one_hot_original_seq).item()
  
    for key, sequences in alt_sequence_dict.items():
        for seq in sequences:
            chrom, pos, original_sequence, variant_sequence = seq
            original_sequence_no_central_dinucleotide = original_sequence[:22] + 'NN' + original_sequence[24:]
            variant_sequence_no_central_dinucleotide = variant_sequence[:22] + 'NN' + variant_sequence[24:]

            one_hot_original_seq = utils.fasta_data.str_to_one_hot(original_sequence_no_central_dinucleotide).unsqueeze(0)
            
            one_hot_variant_seq = utils.fasta_data.str_to_one_hot(variant_sequence_no_central_dinucleotide).unsqueeze(0)
            
            original_seq_output = model(one_hot_original_seq).item()
            variant_seq_output = model(one_hot_variant_seq).item()

            # Output handling
            if original_seq_output > -7.353888034820557 or variant_seq_output > -7.353888034820557:
                print(f"Location: {chrom}:{pos}")
                print(f"Output for original sequence {original_sequence_no_central_dinucleotide}: {original_seq_output}")
                print(f"Output for variant sequence {variant_sequence_no_central_dinucleotide}: {variant_seq_output}")
                print(f"Difference: {str(np.abs(original_seq_output-variant_seq_output))}")
                print('----------')    


if __name__ == "__main__":

    args = parse_arguments()
    model = load_model(args.model_file)
    variants = read_vcf(args.vcf_file)
    alt_sequence_dict = process_variants(variants, args.fasta_file)
    predict_variant_effects(model, alt_sequence_dict)
    # predict_validated_sites(model, args.validated_sites)
