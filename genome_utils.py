import genomepy.seq
import genomepy.genome
import pandas as pd

# Map cryptic site to genomepy genome to extract sequence
def map_site(site, genome):
    chr = site['reference_name']
    strand = site['positive_strand']
    center = site['position']
    start = center - 23 + strand*2
    end = center + 22 + strand*2
    try:
        sequence = genome.get_seq(chr, start, end)
        if strand == False:
            sequence = sequence.reverse.complement
        return sequence.seq
    except:
        return ''
    
def fetch_genomic_sites(sites, reference_file):
    genome = genomepy.Genome(reference_file)
    sites['sequence'] = sites.apply(lambda row: map_site(row, genome), axis=1).str.upper()
    sites['dinucleotide'] = sites['sequence'].str.slice(22, 24)
    return sites