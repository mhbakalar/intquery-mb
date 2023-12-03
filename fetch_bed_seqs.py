import pandas as pd
import numpy as np
import random
import utils.fasta_data

fasta_file = '../data/reference/hg38.fa'
fasta = utils.fasta_data.FastaInterval(fasta_file=fasta_file, return_seq=True)
bed_file = pd.read_csv('output/chr3.bed', sep='\t')
print(bed_file.head())

seqs = []
for i, row in bed_file.iterrows():
    if row['pred'] > 1.0:
        seqs.append(fasta(row['chr'], int(row['start']), int(row['end'])).upper())

pd.DataFrame({'seq':seqs}).to_csv('output/sequences.csv', index=False)