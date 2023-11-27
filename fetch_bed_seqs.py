import pandas as pd
import numpy as np
import random
import utils.fasta_data

fasta_file = '../data/reference/hg38.fa'
fasta = utils.fasta_data.FastaInterval(fasta_file=fasta_file, return_seq=True)
bed_file = pd.read_csv('output/chr1_positive.bed', sep='\t')
print(bed_file.head())

for i, row in bed_file.iterrows():
    print(fasta(row['chr'], int(row['start']), int(row['end'])).upper())


def write_decoys(n_decoys, length=46):
    # Draw samples from canonical chromosomes only
    chromosomes = np.append(np.arange(1,22).astype(str), ['X','Y'])
    chromosomes = np.char.add('chr', chromosomes)

    decoys = []
    for i in range(0, n_decoys):
        chr_name = random.choice(chromosomes)
        chromosome = fasta.seqs[chr_name]
        start = random.randint(0, len(chromosome)-length)
        end = start + length
        seq = str(chromosome[start:end]).upper()
        seq = seq[0:int(length/2)-1] + 'NN' + seq[int(length/2)+1:]
        decoys.append(seq)
    df = pd.DataFrame({'seq':decoys})
    return df

df = write_decoys(100000)
df.to_csv('data/decoys/decoys_100k.csv', index=False)