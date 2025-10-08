import torch 
import numpy as np
import pysam
import joblib 
import cv2, torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes, clip_boxes  
import time


def unormalize_bbox(nbbox, ratio=5000):
    return np.array(nbbox) * ratio

def xywh2sel(bbox):
    """
    return Unormalized start, end, and length of the SV
    """
    bbox = unormalize_bbox(bbox, ratio=5000)
    xc, w = bbox[0], bbox[2]
    start = round(xc-w/2) 
    end = round(xc+w/2)
    length = end - start
    return [start, end, length] 


def cls_2_svgt(cls):
    cls = int(cls)
    if cls==0:
        sv = 'DEL'
        gt = "0/1"
    elif cls==1:
        sv = 'DEL'
        gt = "1/1"
    elif cls==2:
        sv = 'INS'
        gt = "0/1"
    elif cls==3:
        sv = 'INS'
        gt = "1/1"
    return sv, gt


def extract_pred(pred): 
    """
    Extract predicted bboxes in only one image
    Args:
        input: 
            pred -> from on image that might contains 0 object to multiple objects

        output: 
            list -> list of predicted variants in one image. This list contains (start, end, length, conf, sv, gt)
    """
    bboxes = np.array(pred.boxes.xywhn).tolist()
    clses = np.array(pred.boxes.cls).tolist()
    confs = np.array(pred.boxes.conf).tolist()
    pred_SVs = []
    for i in range(len(bboxes)):
        bbox, cls, conf = bboxes[i], clses[i], confs[i]
        start, end, length = xywh2sel(bbox)
        sv, gt = cls_2_svgt(cls)
        pred_SVs.append((start, end, length, conf, sv, gt))
    return pred_SVs



def filter_var(region, sv_type, sv_gt, sv_length, conf, cov, thresh_flag_len=0.7, thresh_flag_cov=0.2):
    # region_org = region.copy()
    region = region[:, :, 0]
    flag_value = 250 if sv_type=='INS' else 200
    kernel = np.array([flag_value] * int(thresh_flag_len*sv_length))
    time_np_lib = time.time()
    windows = np.lib.stride_tricks.sliding_window_view(region, window_shape=len(kernel), axis=1)
    print('time_np_lib: ', time.time() - time_np_lib)
    
    time_matches = time.time()
    matches = np.all(windows == kernel, axis=-1)
    print(f'time_matches: {time.time() - time_matches}')

    time_row_matches = time.time()
    row_matches = np.any(matches, axis=1)
    print(f'time_row_mathces: {time.time() - time_row_matches}')

    time_flag_count = time.time()
    flag_count = int(np.sum(row_matches))
    print(f'time_flag_count: {time.time() - time_flag_count}')
    print('v'*20)
    if flag_count >= int(thresh_flag_cov*cov):
        #cv2.imwrite(f'./vars_post_kept/True_type{sv_type}_gt{sv_gt}_len{sv_length}_conf{conf}_cov{cov}_fcount{flag_count}.png', region_org[:, :, :3])
        return True, flag_count
    else:
        #cv2.imwrite(f'./vars_post_filtered/False_type{sv_type}_gt{sv_gt}_len{sv_length}_conf{conf}_cov{cov}_fcount{flag_count}.png', region_org[:, :, :3])
        return False, flag_count
        



def filter_var2(region, sv_type, sv_gt, sv_length, conf, cov,
               thresh_flag_len=0.7, thresh_flag_cov=0.2):
    """
    proposed by ChatGPT
    Faster version: detects runs of `flag_value` of length run_len in each row
    using a moving-sum (via cumulative sums), instead of sliding_window_view + np.all.
    Returns (keep_flag: bool, flag_count: int) exactly like your original.
    """
    # Work on the first channel only (as in your original code)
    region2d = region[:, :, 0]

    # Choose the flag we're searching for
    flag_value = 250 if sv_type == 'INS' else 200

    # Required run length (kernel length)
    run_len = max(1, int(thresh_flag_len * sv_length))

    H, W = region2d.shape
    if run_len > W:
        # No window can fit; immediately fail the length criterion
        flag_count = 0
        return (flag_count >= int(thresh_flag_cov * cov)), flag_count

    # Boolean mask where entries equal the flag value
    mask = (region2d == flag_value).astype(np.uint8)  # (H, W)

    # Moving window sum of length `run_len` along axis=1, via cumsum (O(H*W), no 3D views)
    # s[:, j] = sum(mask[:, :j+1])
    s = np.cumsum(mask, axis=1, dtype=np.int32)

    # Sum over each window [j, j+run_len)
    # window_sum[:, j] = s[:, j+run_len-1] - s[:, j-1]
    left = s[:, run_len - 1:] # (H, W - run_len + 1)
    right = np.concatenate(
        (np.zeros((H, 1), dtype=np.int32), s[:, :-run_len]), # pad one zero column
        axis=1
    )
    window_sum = left - right # (H, W - run_len + 1)

    # A match exists where the window sum equals run_len
    matches = (window_sum == run_len) # boolean (H, W - run_len + 1)

    # Row has at least one matching window?
    row_matches = np.any(matches, axis=1) # (H,)

    # Count rows with a match
    flag_count = int(np.sum(row_matches))

    # Coverage threshold check (same logic as before)
    keep = flag_count >= int(thresh_flag_cov * cov)
    return keep, flag_count



def postprocessing(img, img_seq, variants, start, end, thresh_flag_len=0.7, thresh_flag_cov=0.2):
    #time_post_0 = time.time()
    real_vars = []
    #print('Number of variants:', len(variants))
    # print(variants)
    for var in variants:
        pos = var[1]
        lng = var[3]
        conf = var[4]
        if pos < start or pos+lng > end:
            continue
        sv_type = var[5]
        sv_gt = var[6][0]

        reg_start = (pos - start) - int(0.2*lng)
        reg_end = (pos-start + lng) + int(0.2*lng)
        if reg_start<0:
            reg_start = 0
        if reg_end > img.shape[1]:
            reg_end = img.shape[1]
        
        
        if sv_type=='DEL':
            sv_region = img[2:150, reg_start: reg_end, :]
            
        elif sv_type=='INS':
            sv_region = img[150:, reg_start: reg_end, :]

        cov = np.mean(img_seq[0, reg_start: reg_end])
        #time_filter_var0 = time.time()
        is_real_var, flag_count = filter_var2(sv_region, sv_type, sv_gt, lng, conf, cov, thresh_flag_len=thresh_flag_len, thresh_flag_cov=thresh_flag_cov)
        #time_filter_var = time.time() - time_filter_var0
        #print(f'time_filter_var: {time_filter_var}')
        var.extend([cov, flag_count])
        if is_real_var:
            real_vars.append(var)
    #print('Number of REAL variants:', len(real_vars))
    # print(real_vars)
    #time_post = time.time() - time_post_0
    #print(f'All time_post: {time_post}')
    #print('-'*50)
    return real_vars
    


# def extract_variant(pred, chrom, start_pos):
#     SVs = []        
#     pred_SVs = extract_pred(pred)
#     for pred in pred_SVs:
#         start, end, length, conf, sv, gt = pred
#         start = start_pos + start
#         end = start + length
#         SVs.append((chrom, start, end, length, conf, sv, gt))
#     return SVs

def scale_pred(pred, org_size):
    start, end, conf, cls = pred[0], pred[2], pred[4], pred[5]
    length = round(((end-start)/640)*org_size)
    start = round((start/640)*org_size)
    return start, length, conf, int(cls)


def extract_variants(preds, chrom, pos_starts, org_size):
    sv_types = ['DEL', 'DEL', 'INS', 'INS']
    sv_gts = ['0/1', '1/1', '0/1', '1/1']
    preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.5)
    SVs = []
    for i, pred in enumerate(preds): 
        pos_start = pos_starts[i]
        pred = np.array(pred.cpu())
        #neg_count = np.sum(pred < 0, axis=1)
        #pred = pred[neg_count % 2 != 0]
        for p in pred:
            start, length, conf, cls= scale_pred(p, org_size=org_size)
            start = pos_start + start 
            sv_type, sv_gt = sv_types[cls], sv_gts[cls]
            SVs.append([chrom, start, start+length, length, conf, sv_type, sv_gt])
            
    return SVs
        

def parse_chromes(SVs_list):
    chromes = [str(chrom) for chrom in np.arange(1, 23)]
    chromes.extend(['X', 'Y'])
    chrom_SVs = []
    for chrom in chromes:
        temp_SVs = []
        for sv in SVs_list:
            if sv[0] == chrom:
                temp_SVs.append(sv)
        chrom_SVs.append(temp_SVs)
    return chrom_SVs


def sort_SVs(SVs_list):
    chrom_SVs = parse_chromes(SVs_list)
    sorted_SVs = []
    for chrom_sv in chrom_SVs:
        chrom_sv.sort(key=lambda v: v[1])
        sorted_SVs.extend(chrom_sv)
    return sorted_SVs


def apply_conf_thresh(SVs, conf_threshold):
    return [sv for sv in SVs if sv[4]>=conf_threshold]

def apply_length_thresh(SVs, length_threshold, cut_size):
    if cut_size==5000:
        return [sv for sv in SVs if sv[3]<length_threshold]
    elif cut_size==50000:
        return [sv for sv in SVs if sv[3]>length_threshold]
    

def stitch_vars(sorted_SVs, distance=100):
    chrom_sorted_SVs = parse_chromes(sorted_SVs)
    stitched_SVs = []
    for chrom_SVs in chrom_sorted_SVs:
        stitched_flags = [False] * len(chrom_SVs)
        for i, sv1 in enumerate(chrom_SVs):
            if stitched_flags[i]:
                continue
            for j in range(i+1, len(chrom_SVs)):
                sv2 = chrom_SVs[j]

                sv1_s, sv1_e = sv1[1], sv1[2]
                sv2_s, sv2_e = sv2[1], sv2[2]

                if sv2_s > sv1_e + distance: 
                    break 

                if sv1[5]!=sv2[5] or sv1[6]!=sv2[6]:
                    # break
                    continue

                if sv2_s < sv1_e and sv2_e <= sv1_e:
                    new_sv = sv1
                #elif sv2_s <= sv1_e + distnace:
                else:
                    chrom, sv_type, sv_gt = sv1[0], sv1[5], sv1[6]
                    start, end = sv1[1], max(sv2[2], sv1[2])
                    length = end - start
                    conf = (sv1[4]+sv2[4])/2
                    new_sv = (chrom, start, end, length, conf, sv_type, sv_gt)
                
                sv1 = new_sv
                stitched_flags[j] = True
            stitched_SVs.append(sv1)
    return stitched_SVs



def create_vcf(variants, dest_path, gtruth_path, sample):
    sample = sample
    predicted_vcf_path = dest_path
    gtruth_vcf_path = gtruth_path

    gtruth_vcf = pysam.VariantFile(gtruth_vcf_path)
    vcf_out = pysam.VariantFile(predicted_vcf_path, 'w', header=gtruth_vcf.header)

    predicted_variants = [{'chrom': var[0], "start": var[1], "end": var[2], "svtype": var[5], 
                  "length": -var[3] if var[5]=='DEL' else var[3], "gt": var[-1], "conf": var[4]}
                  for var in variants
                  ]
    
    for variant in predicted_variants:
        record = vcf_out.new_record(
            contig= variant["chrom"],
            start= variant["start"] - 1,  # VCF uses 0-based indexing
            stop= variant["end"],
            id = variant['svtype'], 
            alleles=("N", "<{}>".format(variant["svtype"])),
            filter='PASS',
            qual=variant["conf"],
            info={"SVTYPE": variant["svtype"], "SVLEN": variant["length"]},
            #samples={"HG002": {"GT": tuple(map(int, variant["gt"].split('/')))}}
        )
        record.samples[sample]['GT'] = tuple(map(int, variant['gt'].split("/")))

        vcf_out.write(record)
    vcf_out.close()


def run_truvari():
    pass

