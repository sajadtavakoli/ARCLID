


"""
This Version (21)
-- previous updates: 
    1. overlap problem solved (overlap between two images in a row) 
    2. overwrite of reads was solved 
    3. the size of image is (512, 1000000), and each image is split to sub images with the length of 10000
    4. sub images have overlap with 2000 bp with each other
    5. sub images are resized to (max_coverage, 1280, 5) (new)
    6. convert 'image saving' part to image_saver function (new)
    7. change the images from 5 channels to 3 channels in order to make Yolo use pre-trained weights 
    8. coverage channel is removed
    9. channels 3 and 4 (quality score and mapping quality) are combined
        This is done by function combine_qs_mq()
    10. an unexpected character 'M' was found in chromosome 3, so, ref = seq2num(...) raised an error, 
        To handle it, we just added 'M' character to trans_table inside seq2num function
    11. a bug about resizing image was fixed 
    12. a new update for combining mq and qs, making it faster

New updates: 
General view: 
    1. Shift problem by insertions is fixed
    2. channel 0 -> sequences, 
    3. channel 1 -> insertions and deletion flags
    4, channel 2 -> combination of qs and mq
    5. row 0 -> coverage
    6. row 1 -> ref sequence
    5. rows 2-150 for deletions
    6. rows 150-300 for insetions 
    7. 'ATCGDI' values were scaled to [40, 80, 120, 160, 200, 250]

Functions: 
    1. new function: extract_indel_flags(): 
        This function extract features from reads and solve the shift problem. The function 
        returns seq_num_no_ins, qs_no_ins, insetions 
            . seq_num_no_ins is the read sequence without insertion(note that it contains deletions)
            . qs_no_ins is the quality scores of aligned reads without insertions (note that it consider 0 for deletions)
            . insetions is a tuple which includes all the insertions happend during the aligned read. 
              each element of insertions tuple has been made of pos, length, insertion sequence, quality score, mapping quality, and insertion flags
    2. new function: find_insertions():
        This function is to extract the insertions inside an alinged reads. 
        This function returns a tuple of insertions. each element in the tuple is made of 
        position inside the read, length, inserted sequence, quality scores of the sequence, 
        mapping quality, and flags (250 as flag for insertions)
    3. new function: rescale_255():
        This function rescale ATCGDI values to [40, 80, 120, 160, 200, 250]
    4. modified function: image_maker():
        a) This function uses extract_indel_flags instead of seq2num_indel()
        b) This function write insertions in rows 150-300

"""




"""
This Version (24)
-- previous updates: 
    1. overlap problem solved (overlap between two images in a row) 
    2. overwrite of reads was solved 
    3. the size of image is (512, 1000000), and each image is split to sub images with the length of 10000
    4. sub images have overlap with 2000 bp with each other
    5. sub images are resized to (max_coverage, 1280, 5) (new)
    6. convert 'image saving' part to image_saver function (new)
    7. change the images from 5 channels to 3 channels in order to make Yolo use pre-trained weights 
    8. coverage channel is removed
    9. channels 3 and 4 (quality score and mapping quality) are combined
        This is done by function combine_qs_mq()
    10. an unexpected character 'M' was found in chromosome 3, so, ref = seq2num(...) raised an error, 
        To handle it, we just added 'M' character to trans_table inside seq2num function
    11. a bug about resizing image was fixed 
    12. a new update for combining mq and qs, making it faster

New updates:  
General view: 
    1. Shift problem by insertions is fixed
    2. channel 0 -> sequences, 
    3. channel 1 -> insertions and deletion flags
    4, channel 2 -> combination of qs and mq
    5. row 0 -> coverage
    6. row 1 -> ref sequence
    5. rows 2-150 for deletions
    6. rows 150-300 for insetions 
    7. 'ATCGDI' values were scaled to [40, 80, 120, 160, 200, 250]

Functions: 
    1. new function: extract_indel_flags(): 
        This function extract features from reads and solve the shift problem. The function 
        returns seq_num_no_ins, qs_no_ins, insetions 
            . seq_num_no_ins is the read sequence without insertion(note that it contains deletions)
            . qs_no_ins is the quality scores of aligned reads without insertions (note that it consider 0 for deletions)
            . insetions is a tuple which includes all the insertions happend during the aligned read. 
              each element of insertions tuple has been made of pos, length, insertion sequence, quality score, mapping quality, and insertion flags
    2. new function: find_insertions():
        This function is to extract the insertions inside an alinged reads. 
        This function returns a tuple of insertions. each element in the tuple is made of 
        position inside the read, length, inserted sequence, quality scores of the sequence, 
        mapping quality, and flags (250 as flag for insertions)
    3. new function: rescale_255():
        This function rescale ATCGDI values to [40, 80, 120, 160, 200, 250]
    4. modified function: image_maker():
        a) This function uses extract_indel_flags instead of seq2num_indel()
        b) This function write insertions in rows 150-300

New updates (23-02-225):
General View:
    1. The model supports only ONT data and HiFi. so CLR are not supported anymore
    2. The reads that don't have quality score are removed
    3. Cut size changed to 5000 and the overlap between cut images is 1000 bp. 
    4. A Coverage line is drawn 
    5. only reads with mapping quality > 20 is kept. So, the reads with lower mapping quality are removed 
    5. combination of quality score and mapping quality are not used anymore, instead we just consider the quality score of the reads


New updates version24 (18-03-2025):
General View: 
    1. Two seperate models are used to predict PacBio HiFi and ONT. One model for hifi and one model for ONT
    2. The previous structure of channels are changed. The previous version had sequence channel(the first channel)/
        , insertion, deletion flags (the second channel), quality score (the third channel). Now, The channels are as follow:
        - First channel) combination of quality scores and mapping quality
        - Second channel) insertion, deletion, and soft-cliped flags from Cigar String
        - Third channel) split-read, suplemantary read, reverse or forward read codes
    3. leading soft-cliped are coded as 150 and trailing soft-cliped as 100
    4. IN addition of the three mentioned channels, the base pairs sequence channel is kept, but not used in trainig or prediction by 
        the deep learning model, this channel is used for postprocessing step, creating variant sequence. IN this channel, the sequences (A, T, C, G) are 
        converted to their ascii code. 

Functions:
    1. 
    2. 
    3. 


"""



import cv2
cv2.setNumThreads(0)

import os
import pysam 
import pyfaidx
import glob as gb  
import numpy as np

import time
import re
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
#from numba import njit



            
#@njit        
def extract_indel_flags(cigar_tuple, seq_ascii, qs, mq, split_sup_strand):
    #print(f'****** seq.shape: {seq_ascii.shape}')

    #seq_no_ins, seq_with_ins = [], []
    #qs_no_ins, qs_with_ins = [], []


    all_flags = np.zeros(shape=(200000,), dtype=np.uint8)
    #all_flags[:] = 200
    
    seq_no_ins_idx, seq_with_ins_idx = 0, 50000
    qs_no_ins_idx, qs_with_ins_idx = 100000, 150000
    
    read_index = 0

    for op, length in cigar_tuple:
            
        if op in [0, 3, 4, 6, 7, 8]: # cigar tuple: 0->M, 3->N, 4->S, 6->P, 7->=, 8->X
            
            all_flags[seq_no_ins_idx: seq_no_ins_idx+length] = seq_ascii[read_index:read_index + length].copy()
            all_flags[seq_with_ins_idx: seq_with_ins_idx + length] = seq_ascii[read_index:read_index + length].copy()

            all_flags[qs_no_ins_idx: qs_no_ins_idx + length] = qs[read_index:read_index+length].copy()
            all_flags[qs_with_ins_idx: qs_with_ins_idx + length] = qs[read_index:read_index+length].copy()

            read_index += length
            seq_no_ins_idx += length
            seq_with_ins_idx += length
            qs_no_ins_idx += length
            qs_with_ins_idx += length

        elif op == 1: # 1 in cigar tuple is equal to INS
            
            all_flags[seq_with_ins_idx: seq_with_ins_idx + length] = seq_ascii[read_index:read_index + length].copy()
            all_flags[qs_with_ins_idx: qs_with_ins_idx + length] = qs[read_index:read_index+length].copy()

            read_index += length
            seq_with_ins_idx += length
            qs_with_ins_idx += length

        elif op==2: # 2 in cigar tuple is equal to DEL    

            all_flags[seq_no_ins_idx: seq_no_ins_idx+length] = 200
            all_flags[seq_with_ins_idx: seq_with_ins_idx + length] = 200

            #all_flags[qs_no_ins_idx: qs_no_ins_idx + length] = 0
            #all_flags[qs_with_ins_idx: qs_with_ins_idx + length] = 0

            seq_no_ins_idx += length
            seq_with_ins_idx += length
            qs_no_ins_idx += length
            qs_with_ins_idx += length


    seq_no_ins = all_flags[0:seq_no_ins_idx]
    seq_with_ins = all_flags[50000: seq_with_ins_idx]
    
    qs_no_ins = all_flags[100000: qs_no_ins_idx]
    qs_with_ins = all_flags[150000:qs_with_ins_idx]
    return seq_no_ins, qs_no_ins, seq_with_ins, qs_with_ins




def extract_indel_flags_original(cigar_tuple, seq_ascii, qs, mq, split_sup_strand):

    seq_no_ins, seq_with_ins = [], []
    qs_no_ins, qs_with_ins = [], []

    # seq_no_ins = np.zeros(shape=(500000,), dtype=np.uint8)
    # seq_with_ins = np.zeros(shape=(500000,), dtype=np.uint8)
    # qs_no_ins = np.zeros(shape=(500000,), dtype=np.uint8)
    # qs_with_ins = np.zeros(shape=(500000,), dtype=np.uint8)
    
    read_index = 0

    for op, length in cigar_tuple:
            
        if op in [0, 3, 4, 6, 7, 8]: # cigar tuple: 0->M, 3->N, 4->S, 6->P, 7->=, 8->X
            seq_temp = seq_ascii[read_index:read_index + length]
            
            seq_no_ins.extend(seq_temp)
            seq_with_ins.extend(seq_temp)

            qs_temp = qs[read_index:read_index+length]
            qs_no_ins.extend(qs_temp)
            qs_with_ins.extend(qs_temp)

            read_index += length

        elif op == 1: # 1 in cigar tuple is equal to INS
            seq_temp = seq_ascii[read_index:read_index + length]
            qs_temp = qs[read_index:read_index+length]

            seq_with_ins.extend(seq_temp)
            qs_with_ins.extend(qs_temp)

            read_index += length

        elif op==2: # 2 in cigar tuple is equal to DEL    
            seq_temp = [200]*length
            seq_no_ins.extend(seq_temp)
            seq_with_ins.extend(seq_temp)

            qs_temp = [0]*length
            qs_no_ins.extend(qs_temp)
            qs_with_ins.extend(qs_temp)

    return seq_no_ins, qs_no_ins, seq_with_ins, qs_with_ins


def rescale_255(arr):
    arr[arr == 2] = 40
    arr[arr == 3] = 80
    arr[arr == 4] = 120
    arr[arr == 5] = 160
    arr[arr == 9] = 200
    return arr


#@njit
def find_insertions(cigar_tuple, seq_with_ins, qs_with_ins, mq, split_sup_strand):
    insertions =  []
    index_no_ins, index_with_ins = 0, 0
    for op, length in cigar_tuple:
        if op == 1:
            start_no_ins, end_no_ins = index_no_ins, index_no_ins+length # that's correct don't change it to index_with_ins
            if length >20:
                seq = seq_with_ins[index_with_ins: index_with_ins+length]
                qs = qs_with_ins[index_with_ins: index_with_ins+length]
                ins_flag = np.array([250]*length, dtype=np.uint8)
                mq_new = np.array([mq]*length, dtype=np.float64)
                split_sup_strand_flag = np.array([split_sup_strand]*length, dtype=np.uint8)
                insertions.append((start_no_ins, length, seq, qs, mq_new, ins_flag, split_sup_strand_flag))
            index_with_ins += length       
        elif op in [0, 2, 3, 4, 6, 7, 8]: 
            index_with_ins += length
            index_no_ins += length

    return insertions


#@njit('uint8[:](unicode_type)')
# @jit
def seq2num(seq):
    #trans_table = str.maketrans('NnAaTtCcGg', '0022334455')
    trans_table = str.maketrans('NnMRAaTtCcGg', '000022334455') # 'M' was added to handle 'M' in chromosome 3
    seq_num = np.array(list(seq.translate(trans_table)), dtype=np.uint8)
    seq_num = rescale_255(seq_num)
    # np.int8 can store number less than <=127, otherwise, it will overwrite
    return seq_num

#@njit
def combine_qs_mq(qs, mq):
    pqs = np.power(10, (-1*qs)/10, dtype=np.float32)
    pmq = np.power(10, (-1*mq)/10)
    qs_mq = np.asanyarray(-10*np.log10(pqs+pmq-pqs*pmq), dtype=np.uint8)    
    return qs_mq


def resize_img(img, max_coverage, width=1280, height=1280):
    img_new = np.zeros(shape=(height, width, img.shape[2]), dtype=np.uint8)
    time_resize_5k_1280 = time.time()
    #img = cv2.resize(img, dsize=(width, max_coverage), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.resize(img, dsize=(width, max_coverage))
    #print(f'time_resize_5k_1280: {time.time()-time_resize_5k_1280}')
    img_new[:max_coverage, :, :] = img
    
    return img_new


def extract_features(reads):
    features_all = []
    t0 = time.time()
    for read in reads:
        t0_call_reads = time.time()
        read_pos = read[0]
        read_seq = read[1]
        qs = read[2]
        mq = read[3]
        cigar = read[4]
        cigar_tuple = read[5]
        split, sup, strand = read[6:9]
        border_start, img_shape = read[9], read[10]

        split_sup_strand = ((split << 2) | (sup << 1) | strand ) * 20 # flag for split, suplementary, and reverese reads 
        strand = 1 if read[6] else 2
        
        skips = cigar_tuple[0][1] if cigar_tuple[0][0]==4 else 0 # how many skips exist in the begining of the cigar string
        skips_tail = cigar_tuple[-1][1] if cigar_tuple[-1][0]==4 else 0 # how many skips exist in the begining of the cigar string
        t1_call_reads = time.time() - t0_call_reads
        #print(f'time_call_reads: {t1_call_reads}')


        t0_extract_indel_flags = time.time()
        seq_num_no_ins, qs_no_ins, seq_with_ins, qs_with_ins = extract_indel_flags(cigar_tuple, read_seq, qs, mq, split_sup_strand)
        t1_extract_indel_flags = time.time() - t0_extract_indel_flags
        #print(np.mean(seq_num_no_ins), np.mean(qs_no_ins), np.mean(seq_with_ins), np.mean(qs_with_ins))
        #print(f'time extract_indel_flags: {t1_extract_indel_flags}')


        #t0_convert2np = time.time()
        #seq_num_no_ins = np.asanyarray(seq_num_no_ins, dtype=np.uint8)
        #qs_no_ins = np.asanyarray(qs_no_ins, dtype=np.float64)
        #qs_with_ins = np.asanyarray(qs_with_ins, dtype=np.float64)
        #t1_convert2np = time.time() - t0_convert2np
        #print(f'time convert2np: {t1_convert2np}')

        t0_find_insertions = time.time()
        insertions = find_insertions(cigar_tuple, seq_with_ins, qs_with_ins, mq, split_sup_strand)
        t1_find_insertions = time.time() - t0_find_insertions
        #print(f'time_find_insertions: {t1_find_insertions}')

        #mq_no_ins = np.array([mq]*len(seq_num_no_ins), dtype=np.float64)
        #t0_mq_no_ins = time.time()
        #mq_no_ins = np.zeros_like(seq_num_no_ins, dtype=np.uint8)
        #mq_no_ins[:] = mq
        #t1_mq_no_ins = time.time() - t0_mq_no_ins
        #print(f'time mq_no_ins: {t1_mq_no_ins}')
        
        # split_sup_strand_no_ins = [split_sup_strand]*len(seq_num_no_ins)
        t0_splt_sup_strand_no_ins = time.time()
        split_sup_strand_no_ins = np.zeros_like(seq_num_no_ins, dtype=np.uint8)
        split_sup_strand_no_ins[:] = split_sup_strand
        t1_splt_sup_strand_no_ins = time.time() - t0_splt_sup_strand_no_ins
        
        #print(f'time splt_sup_strand_no_ins: {t1_splt_sup_strand_no_ins}')

        t0_poses = time.time()
        # start and end columns, along with start and end of the read
        read_start, read_end = 0, len(seq_num_no_ins)
        col_start = read_pos - border_start - skips
        col_end = col_start + read_end
        if col_start < 0 :
            read_start = -1 * col_start
            col_start = 0
        
        if col_end > img_shape[1]: # bug got fixed 
            read_end = read_end - (col_end-img_shape[1]) # bug got fixed
            col_end = img_shape[1]
        t1_poses = time.time() - t0_poses
        #print(f'time start poses: {t1_poses}')

        t0_skips = time.time()
        del_flags = seq_num_no_ins.copy()
        del_flags[del_flags!=200] = 0
        del_flags[:skips] = 150 # leading skipps are considered as 150 value
        if skips_tail:
            del_flags[-skips_tail:] = 100 # trailing skips are considered as 100 value
        t1_skips = time.time() - t0_skips
        #print(f'time skips: {t1_skips}')

        #qs_mq_no_ins = qs_no_ins.copy()
        t0_combine = time.time()
        qs_mq_no_ins = combine_qs_mq(qs_no_ins, mq)
        t1_combine = time.time() - t0_combine
        #print(f'time combine: {t1_combine}')
        time_all = t1_call_reads + t1_extract_indel_flags + t1_find_insertions + t1_splt_sup_strand_no_ins + t1_poses + t1_skips + t1_combine
        #print(f'time all: {time_all}')
        #print('-'*100)
        #print(np.mean(split_sup_strand_no_ins), np.mean(qs_mq_no_ins), np.mean(mq_no_ins), np.max(del_flags))

        features_all.append((read_start, read_end, col_start, col_end, seq_num_no_ins, del_flags, qs_mq_no_ins, split_sup_strand_no_ins, insertions))
    t_finall = time.time()
    msg = f'\nlen(reads): {len(reads)}\nwhole time in extract_feature function: {t_finall - t0}\n' + '-'*50
    f = open('time_extract_func_whole.txt', 'a+')
    f.write(msg)
    f.close()

    return features_all



def preprocess(imgs, device, max_coverage, width, height):
    
    batch = []
    time_all_preprocess = time.time()
    for img_new in imgs:
        
        #img_new = resize_img(img_new, max_coverage, width=width, height=height) # RGB image
        img_new = cv2.resize(img_new, (640, 640))
    
        img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB) # BGR image
    
        img_new = np.ascontiguousarray(img_new.transpose(2, 0, 1)) # WHC to CWH
        
        time_to_torch = time.time()
        img_new = torch.from_numpy(img_new).float()
        
        batch.append(img_new)
    batch = torch.stack(batch, 0).to(device)  # (B,3,H,W)
    #batch = F.interpolate(batch, size=(640, 640), mode='bilinear')
    batch = batch / 255.0
    print(f'time_all_preprocess: {time.time() - time_all_preprocess}, device: {device}')
    
    return batch



def check_indel_flag(cigar_tuple, size_thresh=20):
    count_del, count_ins = 0, 0
    for op, length in cigar_tuple:
        if op==1 and length >=size_thresh:
            count_ins+=1
        elif op==2 and length>=size_thresh:
            count_del += 1
    return count_ins, count_del




def collect_reads(reads, start, end, img_shape):
    reads_info = []
    for read in reads:
        read_pos = int(read.pos)
        read_mq = int(read.mapping_quality)
        
        if read.is_unmapped or  read.is_secondary:
            continue
        if not read.query_qualities:
            continue

        read_mq = int(read.mapping_quality)
        if read_mq <20: 
            continue
        #if read_pos < start: 
        #    continue

        read_seq = str(read.query_sequence)
        read_seq = list(read_seq.upper().encode('ascii'))
        read_qs = list(read.query_qualities)
        
        read_cigar = None
        read_cig_tuple = read.cigartuples
        read_rev = read.is_reverse
        read_split = read.has_tag("SA")
        read_sup = read.is_supplementary
        reads_info.append((read_pos, read_seq, read_qs, read_mq, read_cigar, read_cig_tuple, 
                        read_rev, read_split, read_sup, start, img_shape))
    return reads_info





def image_maker(reads, max_coverage, img_window, ref_seq, overal_cov,offset):
    """
    A new version of image_maker function
    Updates:
        create the main image as three channels image
        create one-channel image for sequence
        create one-channel image for as indicator of the filled rows  
    
    """

    img_seq = np.zeros(shape=(max_coverage, img_window), dtype=np.uint8)
    img_fill_indicator = np.zeros(shape=(max_coverage, img_window), dtype=np.uint8)
    img = np.zeros(shape=(max_coverage, img_window, 3), dtype=np.uint8)
    
    img_seq[1, :len(ref_seq)] = ref_seq # put ref_seq in the first row of img
    img_fill_indicator[1, :len(ref_seq)] = 1 # write '1' in the last channel to indicate that those rows have been written 
    
    img[overal_cov+10: overal_cov+10+10, :, :] = 255 # coverage line for deletions with 10 pixels thickness
    img[150+overal_cov+10: 150+overal_cov+10+10, :, :] = 255 # coverage line for insertions with 10 pixels thickness
    img_fill_indicator[overal_cov+10: overal_cov+10+10, :] = 1 # # write '1' in the last channel to indicate that those rows have been written 
    img_fill_indicator[150+overal_cov+10: 150+overal_cov+10+10, :] = 1 # write '1' in the last channel to indicate that those rows have been written 

    border_start = offset
    # This "for loop" is to write alignment information in each image

    counter = 0
    num_thread = 4
    idx_end = 0
    slice_size = len(reads)//num_thread
    
    #if slice_size >= 5*num_thread:
    if False:
        batch_reads = []
        flag = True
        while flag: 
            idx_start = counter*slice_size
            idx_end = (counter+1)*slice_size 
            if idx_end + slice_size > len(reads):
                idx_end = len(reads)
                flag = False
            batch_reads.append(reads[idx_start: idx_end])
            counter+=1

        t0 = time.time()
        features_all = []
        with ProcessPoolExecutor(max_workers=num_thread) as executor:
            results = [executor.submit(extract_features, br) for br in batch_reads]
            for result in as_completed(results):
                features_all.extend(result.result())
    
    else:
        features_all = extract_features(reads)

    features_all.sort(key=lambda tup:tup[2])
    t0_feature = time.time()
    #f = open('len_features.txt', 'a+')
    # msg = f'\nborder_start: {reads[0][9]}, len results: {len(results)}, len all_features {len(features_all)}, len features: {len(features_all[0])}'
    # f.write(msg)
    # f.close()
    t1 = time.time()
    for features in features_all:
        #t1 = time.time()
        read_start, read_end, col_start, col_end, seq_num_no_ins, del_flags, qs_mq_no_ins, split_sup_strand_no_ins, insertions = features
        try: 
            row_no_ins = list(img_fill_indicator[2:150, col_start]).index(0) + 2 # bug got fixed
        except:
            continue

        img_seq[row_no_ins, col_start: col_end] = seq_num_no_ins[read_start: read_end] # channel 0 of the image
        img_fill_indicator[row_no_ins, col_start: col_end] = 1 # row-indicator channel 
        
        img[row_no_ins, col_start: col_end, 0] = del_flags[read_start: read_end] # channel 1 of the image
        img[row_no_ins, col_start: col_end, 1] = qs_mq_no_ins[read_start: read_end] # channel 2 of the image
        img[row_no_ins, col_start: col_end, 2] = split_sup_strand_no_ins[read_start: read_end] # channel 3 of the image
        #img[row, col_start: col_end, 4] = strand[read_start: read_end] # channel 4 of the image
        
        for ins in insertions:
            start_ins, length = ins[0] - read_start, ins[1]
            if start_ins < 0: 
                break
            seq_ins, qs_ins = ins[2], ins[3]
            mq_ins, flag_ins = ins[4], ins[5]
            split_sup_strand_ins = ins[6]

            qs_mq_ins = combine_qs_mq(qs_ins, mq_ins)

            col_start_ins = col_start + start_ins
            #print(f'col_start_ins {col_start_ins}, len {length}')
            if col_start_ins >= img.shape[1]:
                break 
            if col_start_ins + length > img.shape[1]:
                length = img.shape[1] - col_start_ins
            col_end_ins = col_start_ins + length
            # print(f'{col_start_ins} - {col_end_ins}')

            try: 
                row_ins = list(img_fill_indicator[150:, col_start_ins]).index(0) + 150
            except: 
                row_ins = False
                for shift in range(1, 11):
                    if not img_fill_indicator[150:, col_start_ins+shift].min():
                        row_ins = list(img_fill_indicator[150:, col_start_ins+shift]).index(0) + 150
                        break 
                if not row_ins:
                    continue

            img_seq[row_ins, col_start_ins: col_end_ins] = seq_ins[:length]
            img_fill_indicator[row_ins, col_start_ins: col_end_ins] = 1
            
            img[row_ins, col_start_ins: col_end_ins, 0] = flag_ins[:length]
            img[row_ins, col_start_ins: col_end_ins, 1] = qs_mq_ins[:length]
            img[row_ins, col_start_ins: col_end_ins, 2] = split_sup_strand_ins[:length]
            
    
    img_ch0 = img_seq[2:150, :]
    img_ch0[img[2:150, :, 0]==100] = 0
    img_ch0[img[2:150, :, 0]==150] = 0
    img_ch0[img[2:150, :, 0]==255] = 0
    cov = np.sum(img_ch0>0, axis=0)
    del img_ch0
    img_seq[0, :] = cov
    
    # t2 = time.time()
    # msg = f'\nfeature extraction time: {t0_feature - t0}\n'
    # msg = msg + f'time elapsed for writing reads in image: {t2-t1}' + '\n'
    # msg = msg + f'whole time for img_maker: {t2-t0}\n' + '-'*50

    # f = open('time_write_reads.txt', '+a')
    # f.write(msg)
    # f.close()

    return img, img_seq


                
