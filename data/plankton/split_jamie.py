#!/usr/bin/env python
import random
# import h5py
import numpy as np

ROOT="/home/jamie/data/Plankton/train/"
RES_SUFFIX="_256"
DO_FULL=True
USE_SEED_SUFFIX=True
SEED=2
NUM_SPLITS=0

def image_id_from_list_line(list_line):
    return list_line.split(" ")[0].split("/")[1].split("_")[0]

def write_output(root,seed, infile, trainfile, valfile, \
                 # train_h5, val_h5, train_h5_names, val_h5_names, list_h5, \
                 fullfile = ""):

    random.seed(seed)

    full_list = []
    perm = []
    len_list = 0

    fin=open(infile,'r')
    for i,line in enumerate(fin):
        line_without_dir=line.replace(root,'')
        full_list.append(line_without_dir)
        perm.append(i)

        len_list += 1
    fin.close()

    random.shuffle(perm)

    # with h5py.File(list_h5, 'r') as h5dat:
    #     X=h5dat['data'].value
    #     Y=h5dat['label'].value
    #     X_permuted = X[perm,:]
    #     Y_permuted = Y[perm]

    full_list = [full_list[i] for i in perm]
    frac=0.8
    n=int(len(full_list)*frac)
    print "Train set size = %d, Validation set size = %d, Seed = %d" % (frac*len_list,(1.0-frac)*len_list,seed)

    n1=(n//10)*10
    n2=(len(full_list)/10)*10

    train = full_list[0:n1]
    val = full_list[n1:n2]
    # Xtrain = X_permuted[0:n1,:]
    # Xval = X_permuted[n1:n2,:]
    # Ytrain = Y_permuted[0:n1]
    # Yval = Y_permuted[n1:n2]

    ftrain=open(trainfile,'w')
    for line in train:
        info=' '.join(line.split()[0:3])+'\n'
        ftrain.write(info)
    ftrain.close()
    fval=open(valfile,'w')
    for line in val:
        info=' '.join(line.split()[0:3])+'\n'
        fval.write(info)
    fval.close()
    if fullfile:
        ffull=open(fullfile,'w')
        for line in full_list:
            info=' '.join(line.split()[0:3])+'\n'
            ffull.write(info)
        ffull.close()

    # with h5py.File(train_h5, 'w') as f:
    #     f['data'] = Xtrain
    #     f['label'] = Ytrain

    # with h5py.File(val_h5, 'w') as f:
    #     f['data'] = Xval
    #     f['label'] = Yval

    # with open(train_h5_names, 'w') as f:
    #     f.write(train_h5 + '\n')
    # with open(val_h5_names, 'w') as f:
    #     f.write(val_h5 + '\n')




def setup_filenames_and_write(root,suffix,infile,seed=1):
    trainfile = "train"+suffix+".txt"
    valfile = "val"+suffix+".txt"
    if DO_FULL:
        fullfile = "full"+suffix+".txt"
    else:
        fullfile = ""

    # train_h5 = "train" + suffix + ".h5"
    # val_h5 = "val" + suffix + ".h5"
    # train_h5_names = "train" + suffix + ".h5.txt"
    # val_h5_names = "val" + suffix + ".h5.txt"
    # list_h5 = "list" + suffix + ".h5"

    write_output(root,seed,infile,trainfile,valfile, \
                 #train_h5, val_h5, train_h5_names, val_h5_names, list_h5, \
                 fullfile)

def main():

    suffix = RES_SUFFIX
    infile = "list"+suffix+".txt"
    if USE_SEED_SUFFIX:
        suffix += "_seed" + str(SEED)

    if NUM_SPLITS > 0:

        for split_num in range(NUM_SPLITS):
            suffix=RES_SUFFIX+"_split%d" % (split_num)
            seed = 999+SEED+split_num
            setup_filenames_and_write(ROOT,suffix,infile,seed)
    
    else:
        setup_filenames_and_write(ROOT,suffix,infile,SEED)

if __name__ == "__main__":
    main()
