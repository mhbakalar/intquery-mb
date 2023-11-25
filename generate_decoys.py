from pyfaidx import Fasta
from utils import fasta_data
import random
import numpy as np
import pandas as pd

n_samples = 1000
length = 46

output_name = './data/decoys.bed'

def pick_genomic_decoys(fasta_file, n_samples):
    fasta = fasta_data.FastaInterval(fasta_file=fasta_file)

    # Draw samples from canonical chromosomes only
    chromosomes = np.arange(1,22).astype(str)
    chromosomes = np.append(chromosomes, ['X','Y'])
    chromosomes = np.char.add('chr', chromosomes)

    samples = []
    for i in range(0,n_samples):
        chr_name = random.choice(chromosomes)
        chromosome = fasta.seqs[chr_name]
        start = random.randint(0,len(chromosome)-length)
        end = start + length
        samples.append((chr_name, start, end))

    decoys = pd.DataFrame.from_records(samples, columns=['chr', 'start', 'end'])
    return decoys