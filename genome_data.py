import torch
import re

class DinucleotideDataset(torch.utils.data.IterableDataset):

    def __init__(self, data_file, dinucleotide):
        self.data_file = data_file

    def __iter__(self):
        with open(self.data_file, "r") as f:
            lines = []
            chr = ""
            for line in f:
                line = line.strip()

                # Check for fasta header line
                match = re.search(r">(.*)", line)
                if match:
                    chr = match.group(1)
                    idx = 0
                else:
                    lines.append(line)
                    idx += len(line)

                    if len(lines) > 3:
                        # Assemble 150 bp sequence
                        long_line = "".join(lines)
                        lines.pop(0)

                        # Search for GC dinucleotide in [50,100]
                        pattern = re.compile(r"(GC)")
                        for m in pattern.finditer(lines[1].upper()):
                            # Concatenate last three lines
                            long_line = "".join(lines)
                            span = m.span()

                            # Find the 50 bp region surrounding dinucleotide
                            start = span[0] - 22
                            end = span[1] + 22
                            sequence = long_line[start + len(lines[0]):end + len(lines[0])]
                            coords = (idx + start - 100, idx + end - 100)
                            yield (chr, coords[0], coords[1], sequence)

if __name__ == "__main__":
    dataset = DinucleotideDataset('hg38.fa')
    # Print first 10 sequences
    for i, item in enumerate(dataset):
        print(item)
        if i > 10:
            break