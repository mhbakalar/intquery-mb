import pandas as pd
import os

def extract_excel_cs_data(input_file, sheet_names, output_dir, output_name, threshold, dn_exclusion=[]):
    # Load data from cryptic seq experiment
    sites_xl = pd.ExcelFile(input_file)
    sheets = []
    for name in sheet_names:
        sheets.append(sites_xl.parse(name))
    sites = pd.concat(sheets).reset_index()

    # Only take canonical dinucleotide insertions
    sites = sites[sites['genome_dinucleotide'] == sites['donor'].str.slice(5,7)]

    # Sum count at identical sites
    sites = sites.groupby(['seq'], as_index=False).sum(numeric_only=False)

    # Noramlize by on target count
    UMI_norm_factor = sites[sites['chrom'].str.contains('PL312')]['count'].mean()
    sites['count'] = sites['count']/UMI_norm_factor
    sites = sites[sites['count'] > threshold]

    # Exclude dinucleotides if necessary
    sites = sites[~sites['genome_dinucleotide'].isin(dn_exclusion)]

    # Split into left and right
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    reverse_complement = lambda x: "".join(complement.get(base, base) for base in reversed(x))

    left = sites['seq'].str.slice(0,22)
    left_rc = left.apply(reverse_complement)
    right = sites['seq'].str.slice(24,)
    right_rc = right.apply(reverse_complement)

    sites['left'] = left
    sites['right'] = right_rc
    sites['left_full'] = left + 'NN' + left_rc
    sites['right_full'] = right_rc + 'NN' + right
    sites['full_nn'] = left + 'NN' + right

    # Use both left and right half sites as data inputs
    #output_data = pd.concat([pd.DataFrame({"seq": sites[key], "norm_count": sites['count']}) for key in ['left_full', 'right_full']]).reset_index()
    
    output_data = pd.DataFrame({"seq": sites['full_nn'], "norm_count": sites['count']})  # Train with original data

    # Save to file
    output_data.to_csv(os.path.join(output_dir, output_name), index=False)

    return sites