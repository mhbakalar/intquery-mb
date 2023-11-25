import pandas as pd
import utils.fasta_data

fasta_file = '../data/reference/hg38.fa'
fasta = utils.fasta_data.FastaInterval(fasta_file=fasta_file, return_seq=True)
bed_file = pd.read_csv('output/chr1_positive.bed', sep='\t')
print(bed_file.head())

for i, row in bed_file.iterrows():
    if row['pred'] == 0:
        print(fasta(row['chr'], int(row['start']), int(row['end'])))