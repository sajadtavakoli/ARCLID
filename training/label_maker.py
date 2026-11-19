import numpy as np
import pysam
import os
import glob as gb
import multiprocessing


def label_maker_new(img_paths, vcf_path, sample, output_dir):
    vcf = pysam.VariantFile(vcf_path)
    count_img_with_label = 0
    count_num_variants = 0
    for img_path in img_paths:
        img_name = img_path.split('/')[-1][:-4]
        temp = img_name.split('_')
        chr = temp[-2]
        start, end = int(temp[-1].split('-')[0]), int(temp[-1].split('-')[1])
        img_size_x = end - start          # 5,000 bp window
        img_size_y = 1280                 # rows of the pileup image
        vcf_start = 0 if start - img_size_x < 0 else start - img_size_x
        try:
            variants = vcf.fetch(chr, vcf_start, end)
        except:
            print('ERROR: ', chr, start, end, img_name)
            continue
        labels = []
        counter = 0
        for i, variant in enumerate(variants):
            if variant.filter.keys()[0] != 'PASS':
                continue
            sv_len = np.abs(variant.info['SVLEN'][0])

            # Keep the variants of at least 50 bp
            if sv_len < 50:
                continue

            sv_type = variant.info['SVTYPE']
            sv_gt = variant.samples[sample]["GT"]
            sv_pos = int(variant.pos)
            if sv_gt == (None, None):
                continue
            if sv_pos + sv_len <= start + 50:
                continue

            counter += 1
            count_num_variants += 1

            if sv_pos < start:
                x_min = 0
                x_max = sv_pos + sv_len - start
            elif sv_pos >= start:
                x_min = sv_pos - start
                x_max = x_min + sv_len
            if x_max >= img_size_x:
                x_max = img_size_x
            y_min = 0
            y_max = 300
            w = (x_max - x_min) / img_size_x
            h = (y_max - y_min) / img_size_y
            cx = (x_max + x_min) / (2 * img_size_x)
            cy = (y_max + y_min) / (2 * img_size_y)

            if sv_type == 'DEL':
                if sv_gt in [(0, 1), (1, 0)]:
                    cls = 0
                elif sv_gt == (1, 1):
                    cls = 1

            elif sv_type == 'INS':
                if sv_gt in [(0, 1), (1, 0)]:
                    cls = 2
                elif sv_gt == (1, 1):
                    cls = 3

            else:
                continue
            label = f'{cls} {cx} {cy} {w} {h}'
            labels.append(label)

        if counter == 0:
            continue
        count_img_with_label += 1
        file_name = ''
        for i in range(len(temp) - 2):
            file_name += temp[i]
            file_name += '_'
        file_name = file_name + chr + '_' + str(start) + '-' + str(end) + '.txt'
        path = output_dir + file_name

        f = open(path, 'w')
        for label in labels:
            f.write(label + '\n')
        f.close()
    print(f'number of images that has variant: {count_img_with_label}')
    print(f'number of all variants passed {count_num_variants}')


# ---------------------------------------------------------------------------
dir_run = '/path/to/HG002/'      # 
imgs_dir_all_cov = [
                    'HG002_CCS_5x_imgs/',
                    'HG002_CCS_10x_imgs/',
                    'HG002_CCS_20x_imgs/',
                    'HG002_CCS_28x_imgs/',
                    ]

labels_dir_all_cov = [
                    'HG002_CCS_5x_labels/',
                    'HG002_CCS_10x_labels/',
                    'HG002_CCS_20x_labels/',
                    'HG002_CCS_28x_labels/',
                    ]

vcf_path = '/path/to/groundtruth.vcf.gz'   # <-- GIAB Tier1 v0.6 truth set
sample = 'HG002'
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    processes = []
    for i in range(len(imgs_dir_all_cov)):
        print('-' * 50, '\n', imgs_dir_all_cov[i])
        imgs_dir = dir_run + imgs_dir_all_cov[i]
        imgs_paths = gb.glob(imgs_dir + '*.png')

        output_dir = dir_run + labels_dir_all_cov[i]
        os.makedirs(output_dir, exist_ok=True)

        process = multiprocessing.Process(target=label_maker_new, args=(imgs_paths, vcf_path, sample, output_dir))
        processes.append(process)
        process.start()
        print(f'process {i+1} started')

    for j, process in enumerate(processes):
        process.join()
        print(f'process {j+1} joined')
