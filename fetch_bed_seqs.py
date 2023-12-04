import pandas as pd
import numpy as np
import random
import utils.fasta_data

fasta_file = '../data/reference/hg38.fa'
fasta = utils.fasta_data.FastaInterval(fasta_file=fasta_file, return_seq=True)
fasta_rc = utils.fasta_data.FastaInterval(fasta_file=fasta_file, return_seq=True, rc_aug=True)

bed_file = pd.read_csv('output/chr7.bed', sep='\t')
print(bed_file.head())

seqs = []
for i, row in bed_file.iterrows():
    if row['pred'] > 2.0:
        if row['strand'] == '+':
            seqs.append(fasta(row['chr'], int(row['start']), int(row['end'])).upper())
        elif row['strand'] == '-':
            seqs.append(fasta_rc(row['chr'], int(row['start']), int(row['end'])).upper())

pd.DataFrame({'seq':seqs}).to_csv('output/sequences.csv', index=False)