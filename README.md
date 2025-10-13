# ARCLID
**Accurate and Robust Characterization of Long Insertions and Deletions**

ARCLID is a deep learning–based structural variant caller that leverages long-read pileup images for accurate SV detection and genotyping.

![arclid logo](https://github.com/user-attachments/assets/d4df95d0-7226-483f-aa41-ea73e7bb40ad)


## 🧭 Features
- Detects insertions and deletions using YOLOv11x deep learning
- Works with PacBio HiFi 
- Supports GPU acceleration and multiprocessing
- Achieves high accuracy even at low sequencing coverage

## 🚀 Installation
```bash
git clone https://github.com/sajadtavakoli/ARCLID.git
cd ARCLID
pip install -r requirements.txt
```

## Usage
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

