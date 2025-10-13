# ARCLID
**ARCLID: Accurate and Robust Characterization of Long Insertions and Deletions**
accurate detection of genomic structural variants (SVs) remains challenging, especially for large variants and at low sequencing coverage. Here, we introduce ARCLID, a novel deep learning-based SV caller for PacBio HiFi data that treats SVs as objects within pileup images. Evaluated across diverse real and synthetic datasets at varying coverage levels, ARCLID exhibits reliable and consistent accuracy, even for SVs larger than 1 kbp. Notably, it maintains high performance on samples with lower sequencing depths (e.g., 10× and 5×). This ability to preserve accuracy at reduced depths offers a substantial practical advantage for cost-constrained projects, facilitating robust SV discovery without requiring deep sequencing. Overall, ARCLID represents a promising step toward more accessible and efficient long-read SV analysis in diverse research applications. 

![arclid logo](https://github.com/user-attachments/assets/d4df95d0-7226-483f-aa41-ea73e7bb40ad)



## 🧭 Features
- Detects insertions and deletions using YOLOv11x deep learning
- Maintains high accuracy even at low sequencing coverage  
- High accuracy for large SVs
- This version works only with PacBio HiFi 
- Supports GPU acceleration and multiprocessing


## 🚀 Installation
```bash
git clone https://github.com/sajadtavakoli/ARCLID.git
cd ARCLID
pip install -r requirements.txt
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

