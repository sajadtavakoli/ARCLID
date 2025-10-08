# shm_per_worker_slots.py
import os, time
import numpy as np
import cv2
import torch
import torch.multiprocessing as mp
from multiprocessing import shared_memory
from ultralytics import YOLO
from .utils import *
from .postprocess import * 
from .workers import *
from .model_loader import load_model

import pysam 
import joblib




# ---------------- Config ----------------


DEVICE_ID = 0
MAX_COVERAGE = 1280
DTYPE = np.uint8
CUT_SIZE_SMALL = 5000
CUT_SZIE_LARGE = 50000
CUT_OVERLAP = 1000
IMG_MARGIN_RIGHT = 25000
TARGET_HW = (640, 640)     # (W, H)
W, H, C = 640, 640, 3 # width, height, channel
IMG_MAX = 255.0
GPU_BATCH = 16

# Status codes
FREE, READY, BUSY = 0, 1, 2




def run_arclid(aln_path, ref_path, out_path, cov, threads, sample, contigs, lengths, fast):
    user_input = (aln_path, ref_path, out_path, cov, threads, sample, contigs, lengths, fast)
     
    if threads > os.cpu_count():
        threads = os.cpu_count()
        
    if fast:
        n_imgs_5k, n_imgs_50k = 6, 1 # number of 5k images and number of 50k images
        n_imgs = n_imgs_5k + n_imgs_50k # total number of images
        slot_bytes = n_imgs * C * W * H # 6 images for 5k model and 1 image for 50k model
        
    else:
        pass

    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    ext_var_meta = mgr.list([mgr.dict({'contig': None, 'pos_start': None}) for _ in range(threads)]) # these meta data will be used for extraction_variants function in gpu_worker
    pred_meta = mgr.list([None for _ in range(threads)]) # None will be replaced by prediction list after extract_variants function in gpu_worker, the cpu_worker will uses them



    # Create one SHM slot per worker (fixed memory bound)
    shm_blocks = [shared_memory.SharedMemory(create=True, size=slot_bytes) for _ in range(threads)]
    shm_names = [shm.name for shm in shm_blocks]

    # Shared status array
    status = ctx.Array('b', [FREE] * threads, lock=False)  # int8 values

    # Per-slot signaling
    ready_events = [ctx.Event() for _ in range(threads)]
    done_events  = [ctx.Event() for _ in range(threads)]

    # Start GPU worker (single persistent)
    gpu_p = ctx.Process(target=gpu_worker,
                        args=(load_model, DEVICE_ID, (n_imgs_5k, n_imgs_50k), shm_names, status, 
                              ready_events, done_events, ext_var_meta, pred_meta))
    gpu_p.start()

    # Start CPU workers (each owns one slot/index)
    cpu_ps = []
    for wid in range(threads):
        if fast:
            p = ctx.Process(target=cpu_worker_fast,
                            args=(wid, user_input, n_imgs, shm_names[wid], status,
                                ready_events[wid], done_events[wid], ext_var_meta, pred_meta))
        else:
            pass

        p.start()
        cpu_ps.append(p)

    # Wait for CPU workers to finish
    for p in cpu_ps:
        p.join()

    # Stop GPU: simplest is terminate (since no model state to save)
    gpu_p.terminate()
    gpu_p.join()

    # Cleanup SHM
    for shm in shm_blocks:
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
    

