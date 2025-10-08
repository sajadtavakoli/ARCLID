import os, time
import numpy as np
import cv2
import torch
import torch.multiprocessing as mp
from multiprocessing import shared_memory
from ultralytics import YOLO
from .postprocess import * 
from .utils import *
from .model_loader import load_model

import pysam 
import joblib


DTYPE = np.uint8

# Global variables
MAX_COVERAGE = 1280
CUT_SIZE_SMALL = 5000
CUT_SIZE_LARGE = 50000
CUT_OVERLAP = 1000
IMG_MARGIN_RIGHT = 25000
TARGET_HW = (640, 640)     # (W, H)
W, H, C = 640, 640, 3 # width, height, channels
IMG_MAX = 255.0
CONF_THRESH = 0.5
GPU_BATCH = 16

# Status codes
FREE, READY, BUSY = 0, 1, 2



# ---------------- GPU Worker ----------------
def gpu_worker(device_id, n_imgs, shm_names, status_arr, ready_events, done_events, ext_var_meta, pred_meta):
    n_imgs_5k, n_imgs_50k = n_imgs
    n_imgs_total = n_imgs_5k + n_imgs_50k

    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    cv2.setNumThreads(1)

    # Attach to all shared buffers
    shms = [shared_memory.SharedMemory(name=n) for n in shm_names]
    bufs = [np.ndarray((n_imgs_total, C, H, W), dtype=DTYPE, buffer=shm.buf) for shm in shms]
    
    model_5k, model_50k = load_model()
    model_5k.to(device).eval()
    model_50k.to(device).eval()

    with torch.inference_mode():
        alive = True
        while alive:
            # Wait for any slot to become READY
            # (simple loop: wait on each event with short timeout)
            any_ready = False
            for i in range(len(shms)):
                if ready_events[i].wait(timeout=0.01):  # fast wakeup
                    any_ready = True
                    # double-check status
                    if status_arr[i] != READY:
                        ready_events[i].clear()
                        continue
                    # Mark BUSY
                    status_arr[i] = BUSY
                    ready_events[i].clear()
                    contig = ext_var_meta[i]['contig']
                    pos_starts = ext_var_meta[i]['pos_start']
                    pos_starts_5k = pos_starts[:n_imgs_5k]
                    pos_starts_50k = pos_starts[:n_imgs_50k]


                    # ----- process this slot -----
                    batch = bufs[i]
                    b = []
                    #t0 = time.time()
                    for idx in range(n_imgs_5k):
                        img_temp = torch.from_numpy(batch[idx, :, :, :]).float() 
                        b.append(img_temp)
                    b = torch.stack(b, 0).to(device, non_blocking=True)/IMG_MAX
                    #t1 = time.time() - t0
                    #print(f'time: {t1}')
                    preds_5k = model_5k.model(b)
                    SVs_5k = extract_variants(preds=preds_5k, chrom=contig, pos_starts=pos_starts_5k, org_size=CUT_SIZE_SMALL)

                    b = []
                    for idx in range(n_imgs_5k, n_imgs_total):
                        img_temp = torch.from_numpy(batch[idx, :, :, :]).float() 
                        b.append(img_temp)
                    b = torch.stack(b, 0).to(device, non_blocking=True)/IMG_MAX
                    preds_50k = model_50k.model(b)
                    SVs_50k = extract_variants(preds=preds_50k, chrom=contig, pos_starts=pos_starts_50k, org_size=CUT_SIZE_LARGE)

                    # Mark FREE and notify the CPU worker for this slot
                    pred_meta[i] = [SVs_5k, SVs_50k] # sending the prediction to cpu_worker_i 
                    
                    status_arr[i] = FREE
                    done_events[i].set()
                    
                    



# ---------------- CPU Worker ----------------
def cpu_worker_fast(worker_id, user_inputs, n_imgs, shm_name, status_arr, ready_event, done_event, ext_var_meta, pred_meta):
    aln_path, ref_path, out_path, cov, threads, sample, contigs, lengths, fast = user_inputs
    img_window=50000
    img_margin_left=0
    img_margin_right = 25000
    

    cv2.setNumThreads(1)
    rng = np.random.default_rng(1234 + worker_id)
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray((n_imgs, C, H, W), dtype=DTYPE, buffer=shm.buf)

    aln = pysam.AlignmentFile(aln_path)
    aln_window = pysam.AlignmentFile(aln_path)
    ref = pysam.FastaFile(ref_path)

    
    SVs_All = []
    for contig in contigs:
        print(f'worker_id {worker_id}, contig {contig}')
        pos_temp = 0

        # find the segment in contig for this thread
        length = lengths[lengths.index(contig)]
        portion = length//threads
        contig_start = worker_id*portion
        contig_end = (worker_id+1)*portion
        if contig_end>length:
            contig_end = length
        all_reads = aln.fetch(contig=contig, start=contig_start, end=contig_end)

        for r0 in all_reads:
            while status_arr[worker_id] != FREE:
                # If the GPU signaled done on a previous round, clear it
                done_event.wait(timeout=0.01)
                done_event.clear()
                time.sleep(0.001)
            
            if r0.is_secondary or r0.is_unmapped :
                continue
            #if not r0.mapping_quality:
            #    continue
            if int(r0.mapping_quality) < 20:
                continue
            if int(r0.pos) + int(r0.query_length) < pos_temp:
                continue 
            
            splt_flag = r0.has_tag('SA')
            ins_flg, del_flg = check_indel_flag(r0.cigartuples, size_thresh=40)
            if ins_flg or del_flg:
                
                r0_len = int(r0.query_length)
                
                time_flag0 = time.time()
                # create image and predict
                start = r0.pos - 1000
                if start<0:
                    start = 0 
                    img_margin_left=0
                end = start + img_window
                pos_temp = end - img_margin_right
                
                ref_seq = seq2num(str(ref.fetch(contig, start=start, end=end))) # load ref_seq and convert to digits

                reads_info = []
                reads_window = aln_window.fetch(contig, start, end)
                for read in reads_window:
                    read_pos = int(read.pos)
                    read_mq = int(read.mapping_quality)
                    
                    if read.is_unmapped or  read.is_secondary:
                        continue
                    if not read.query_qualities:
                        continue
                    #if not read.mapping_quality:
                    #    continue
                    read_mq = int(read.mapping_quality)
                    if read_mq <20: 
                        continue
                    #if read_pos < start: 
                    #    continue

                    read_qs = np.array(read.query_qualities)
                    
                    #read_seq = str(read.query_sequence)
                    #read_seq = np.array(list(read_seq.upper().encode('ascii')))

                    read_seq = np.ones_like(read_qs)*92
                    
                    
                    read_cigar = None
                    read_cig_tuple = read.cigartuples
                    read_rev = read.is_reverse
                    read_split = read.has_tag("SA")
                    read_sup = read.is_supplementary
                    reads_info.append((read_pos, read_seq, read_qs, read_mq, read_cigar, read_cig_tuple, 
                                    read_rev, read_split, read_sup, start, (MAX_COVERAGE, img_window, 6)))
                
                if len(reads_info) == 0:
                    continue
                
                img, img_seq = image_maker(reads=reads_info, max_coverage=MAX_COVERAGE, 
                                            img_window=img_window, ref_seq=ref_seq, overal_cov=cov, offset=start)
                
                batch = []
                pos_starts = []
                pos = img_margin_left
                while pos + CUT_SIZE_SMALL <= img.shape[1] - IMG_MARGIN_RIGHT + 1000:
                    cut = img[:, pos:pos + CUT_SIZE_SMALL, :].copy()
                    cut = cv2.resize(cut, (W, H))
                    cut = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
                    cut = np.ascontiguousarray(cut.transpose(2, 0, 1))
                    batch.append(cut)
                    pos_starts.append(start + pos) # the last line of code (06-10-2025)
                    pos = pos + CUT_SIZE_SMALL - CUT_OVERLAP
                
                img = cv2.resize(img, (W, H))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR image
                img = np.ascontiguousarray(img.transpose(2, 0, 1)) # WHC to CWH
                batch.append(img)
                pos_starts.append(start)
                
                batch = np.array(batch, dtype=np.uint8)
                ext_var_meta[worker_id]['contig'] = contig
                ext_var_meta[worker_id]['pos_start'] = pos_starts
                
                #print(f'worker_id {worker_id}, {contig}-{start}, contig {ext_var_meta[worker_id]}')
                buf[:] = batch
                # Mark READY and signal GPU

                status_arr[worker_id] = READY
                ready_event.set()

                # Wait until GPU marks it FREE again (processed)
                done_event.wait()
                done_event.clear()

                # collect SVs from images 5k
                SVs_5k = pred_meta[worker_id][0]
                SVs_5k = apply_length_thresh(SVs_5k, length_threshold=2100, cut_size=CUT_SIZE_SMALL)
                SVs_5k = postprocessing(img=img, img_seq=img_seq, variants=SVs_5k, start=start, 
                                               end=end-img_margin_right, thresh_flag_len=0.7, thresh_flag_cov=0)
                SVs_All.extend(SVs_5k)
                
                # collect SVs from images 50k 
                SVs_50k = pred_meta[worker_id][1]
                SVs_50k = apply_conf_thresh(SVs_50k, conf_threshold=CONF_THRESH)
                SVs_50k = apply_length_thresh(SVs_50k, length_threshold=2000, cut_size=CUT_SIZE_LARGE)
                SVs_All.extend(SVs_50k)

                
    aln.close()
    aln_window.close()
    joblib.dump(SVs_All, f'SVs_{worker_id}.joblib')
    print(f'worker_id {worker_id} is done')
    