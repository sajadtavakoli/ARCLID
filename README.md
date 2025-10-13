# ARCLID
**ARCLID: Accurate and Robust Characterization of Long Insertions and Deletions**

ARCLID is a deep learning-based tool for detecting large genomic structural variants in PacBio HiFi data, excelling at low coverage (5×-10×). It reduces sequencing costs, aiding medical research, agriculture, conservation, and global scientific equity by enabling accessible variant discovery. 

![arclid logo](https://github.com/user-attachments/assets/d4df95d0-7226-483f-aa41-ea73e7bb40ad)



## 🧭 Features
- Detects insertions and deletions using YOLOv11x deep learning
- Maintains high accuracy even at low sequencing coverage  
- High accuracy for large SVs
- This version works only with PacBio HiFi 
- Supports GPU acceleration and multiprocessing


## 🚀 Installation
```bash
pip install "git+https://github.com/sajadtavakoli/ARCLID.git"

```

## 🧩 Usage
### Quick usage
```bash
arclid -a aln.bam -r ref.fa -o output.vcf -c 10
```

c is the coverage of the sample

### All options
```bash
arclid \
    -a path/to/aln.bam \ # alignment file path (*.bam or *.cram)
    -r path/to/reference.fa \ # reference file path
    -o path/to/output.vcf \ # output path (*.vcf)
    -c 10 \ # sample coverage 
    -s NA19240 \ # sample name (default=SAMPLE)
    -t 8 \ # number of threads (default=4)
    --contigs chr1,chr2 \ # contigs of interest (default=all)
    --fast 1 # fast mode is faster, but likely a bit drop in accuracy (default: 1, 1->fast, 0->slow)
``` 

| Flag                  | Name                | Type  | Default  | Description                                                                                    |
| --------------------- | ------------------- | ----- | -------- | ---------------------------------------------------------------------------------------------- |
| `-a`, `--aln_path`    | Alignment file path | `str` | —        | Path to input alignment file (`.bam` or `.cram`)                                               |
| `-r`, `--ref_path`    | Reference file path | `str` | —        | Path to reference genome file (`.fa`)                                                          |
| `-o`, `--out_path`    | Output file path    | `str` | —        | Path to output VCF file (`.vcf`)                                                               |
| `-c`, `--coverage`    | Sample coverage     | `int` | —        | Sequencing coverage (e.g., `10` for 10×)                                                       |
| `-s`, `--sample_name` | Sample name         | `str` | `SAMPLE` | Name of the sample used for labeling output                                                    |
| `-t`, `--threads`     | Threads             | `int` | `4`      | Number of CPU threads to use                                                                   |
| `--contigs`           | Contigs of interest | `str` | `all`    | Comma-separated list of contigs (e.g., `chr1,chr2`)                                            |
| `--fast`              | Fast mode           | `int` | `1`      | Enables fast mode (1 = fast, 0 = slow). Fast mode runs quicker with a slight drop in accuracy. |
