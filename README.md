# ARCLID
**Accurate and Robust Characterization of Long Insertions and Deletions**

ARCLID is a deep learning–based structural variant caller that leverages long-read pileup images for accurate SV detection and genotyping.

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
