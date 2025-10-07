import argparse
from .core.pipeline import run_arclid
from .version import __version__

def main():
    parser = argparse.ArgumentParser(
        prog="arclid",
        description="ARCLID: Accurate and Robust Characterization of Long Insertions and Deletions",
    )
    parser.add_argument('-a', '--aln_path', type=str, required=True , help='path of the alignment file (*.bam, *.cram)')
    parser.add_argument('-r', '--ref_path', type=str, required=True, help='path of the reference genome (*.fa, *.fasta)')
    parser.add_argument('-o', '--out_path', type=str, required=True, help='output path (*.vcf)')
    parser.add_argument('-c', '--coverage', type=int, required=True, help='coverage of the alignment file')
    parser.add_argument('-t', '--threads', type=int, default=4, help='number of threads')
    parser.add_argument('-s', '--sample', type=str, default='SAMPLE', help='sample name')
    parser.add_argument('--contigs', type=str, default='all', help='contigs-default-> all contigs (for example: chr1,chr2)')
    parser.add_argument('--fast', type=int, default=1, help='fast mode is faster, but likely a bit drop in accuracy (default: 1, 1->fast, 0->slow)')
    
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()
    run_arclid(
        aln_path=args.aln_path,
        ref_path=args.ref_path,
        vcf_path=args.out_path,
        cov=args.coverage
        threads=args.threads,
        sample=args.sample
        contigs=args.contigs,
        fast=args.fast,
    )

if __name__ == "__main__":
    main()
