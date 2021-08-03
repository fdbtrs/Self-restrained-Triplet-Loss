import argparse
import os
import sys

import numpy as np
from  sklearn.metrics.pairwise import cosine_similarity



def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def main(args):
    pairs=read_pairs(args.pairs)

    #UMR-UMP-SRT
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    umr_ump_dir=os.path.join(args.output,"umr_ump_srt")
    os.makedirs(umr_ump_dir)
    file_ext="npy"
    genuine=open(os.path.join(umr_ump_dir,"genuine.txt"),'w')
    imposter=open(os.path.join(umr_ump_dir,"imposter.txt"),'w')

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(args.embedding_srt, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.embedding_srt, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)

            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(args.embedding_srt, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.embedding_srt, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        r=np.load(path0).reshape(1,-1)
        p=np.load(path1).reshape(1,-1)
        sim = cosine_similarity(r,p)
        if(issame):
            genuine.write(str(sim[0][0]) + '\n')
        else:
            imposter.write(str(sim[0][0]) + '\n')
    genuine.close()
    imposter.close()




    #UMR-UMP
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    umr_ump_dir=os.path.join(args.output,"umr_ump")
    os.makedirs(umr_ump_dir)
    file_ext="npy"
    genuine=open(os.path.join(umr_ump_dir,"genuine.txt"),'w')
    imposter=open(os.path.join(umr_ump_dir,"imposter.txt"),'w')

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(args.embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)

            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(args.embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.embedding_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        r=np.load(path0).reshape(1,-1)
        p=np.load(path1).reshape(1,-1)
        sim = cosine_similarity(r,p)
        if(issame):
            genuine.write(str(sim[0][0]) + '\n')
        else:
            imposter.write(str(sim[0][0]) + '\n')
    genuine.close()
    imposter.close()


    #umr-mp
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    outdir=os.path.join(args.output,"umr_mp")
    os.makedirs(outdir)
    file_ext="npy"
    genuine=open(os.path.join(outdir,"genuine.txt"),'w')
    imposter=open(os.path.join(outdir,"imposter.txt"),'w')

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(args.embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)

            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(args.embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        r=np.load(path0).reshape(1,-1)
        p=np.load(path1).reshape(1,-1)
        sim = cosine_similarity(r,p)
        if(issame):
            genuine.write(str(sim[0][0]) + '\n')
        else:
            imposter.write(str(sim[0][0]) + '\n')
    genuine.close()
    imposter.close()


    #umr-mp-triplet

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    outdir = os.path.join(args.output, "umr_mp_triplet")
    os.makedirs(outdir)
    file_ext = "npy"
    genuine = open(os.path.join(outdir, "genuine.txt"), 'w')
    imposter = open(os.path.join(outdir, "imposter.txt"), 'w')

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(args.embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_triplet, pair[0],
                                 pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)

            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(args.embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_triplet, pair[2],
                                 pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        r = np.load(path0).reshape(1, -1)
        p = np.load(path1).reshape(1, -1)
        sim = cosine_similarity(r, p)
        if (issame):
            genuine.write(str(sim[0][0]) + '\n')
        else:
            imposter.write(str(sim[0][0]) + '\n')
    genuine.close()
    imposter.close()



    #umr-mp-srt

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    outdir = os.path.join(args.output, "umr_mp_srt")
    os.makedirs(outdir)
    file_ext = "npy"
    genuine = open(os.path.join(outdir, "genuine.txt"), 'w')
    imposter = open(os.path.join(outdir, "imposter.txt"), 'w')

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(args.embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_srt, pair[0],
                                 pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)

            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(args.embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_srt, pair[2],
                                 pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        r = np.load(path0).reshape(1, -1)
        p = np.load(path1).reshape(1, -1)
        sim = cosine_similarity(r, p)
        if (issame):
            genuine.write(str(sim[0][0]) + '\n')
        else:
            imposter.write(str(sim[0][0]) + '\n')
    genuine.close()
    imposter.close()



    #mr-mp

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    outdir = os.path.join(args.output, "mr_mp")
    os.makedirs(outdir)
    file_ext = "npy"
    genuine = open(os.path.join(outdir, "genuine.txt"), 'w')
    imposter = open(os.path.join(outdir, "imposter.txt"), 'w')

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(args.mask_embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_dir, pair[0],
                                 pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)

            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(args.mask_embedding_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_dir, pair[2],
                                 pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        r = np.load(path0).reshape(1, -1)
        p = np.load(path1).reshape(1, -1)
        sim = cosine_similarity(r, p)
        if (issame):
            genuine.write(str(sim[0][0]) + '\n')
        else:
            imposter.write(str(sim[0][0]) + '\n')
    genuine.close()
    imposter.close()




    #mr-mp

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    outdir = os.path.join(args.output, "mr_mp_triplet")
    os.makedirs(outdir)
    file_ext = "npy"
    genuine = open(os.path.join(outdir, "genuine.txt"), 'w')
    imposter = open(os.path.join(outdir, "imposter.txt"), 'w')

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(args.mask_embedding_triplet, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_triplet, pair[0],
                                 pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)

            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(args.mask_embedding_triplet, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_triplet, pair[2],
                                 pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        r = np.load(path0).reshape(1, -1)
        p = np.load(path1).reshape(1, -1)
        sim = cosine_similarity(r, p)
        if (issame):
            genuine.write(str(sim[0][0]) + '\n')
        else:
            imposter.write(str(sim[0][0]) + '\n')
    genuine.close()
    imposter.close()



    #mr-mp

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    outdir = os.path.join(args.output, "mr_mp_srt")
    os.makedirs(outdir)
    file_ext = "npy"
    genuine = open(os.path.join(outdir, "genuine.txt"), 'w')
    imposter = open(os.path.join(outdir, "imposter.txt"), 'w')

    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(args.mask_embedding_srt, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_srt, pair[0],
                                 pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)

            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(args.mask_embedding_srt, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(args.mask_embedding_srt, pair[2],
                                 pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        r = np.load(path0).reshape(1, -1)
        p = np.load(path1).reshape(1, -1)
        sim = cosine_similarity(r, p)
        if (issame):
            genuine.write(str(sim[0][0]) + '\n')
        else:
            imposter.write(str(sim[0][0]) + '\n')
    genuine.close()
    imposter.close()



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding-dir', type=str, default='lfw/face_embedding/outputlwf-Mobilefacenet',
                        help='Directory with extracted features.')
    parser.add_argument('--mask-embedding-dir', type=str, default='lfw/face_embedding_mask/outputlwf-Mobilefacenet',
                        help='Directory with masked extracted features.')
    parser.add_argument('--embedding-srt', type=str,
                        default='SRT2/face_embedding/MobilefacenetSRT2',
                        help='Directory with masked extracted features from eum.')
    parser.add_argument('--mask-embedding-srt', type=str, default='SRT2/face_embedding_mask/outputlwf-MobilefacenetSRT2',
                        help='Directory with masked extracted features from eum.')
    parser.add_argument('--mask-embedding-triplet', type=str,
                        default='Triplet/face_embedding_mask/outputlwf-MobilefacenetTriplet',
                        help='Directory with masked extracted features from eum.')
    parser.add_argument('--pairs', type=str, default='pairs.txt',
                        help='lfw pairs.')
    parser.add_argument('--output', type=str, default='lfw_comparsion_Mobilefacenet',
                        help='output path.')


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
