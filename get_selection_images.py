#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:35:10 2023

@author: yzahran
"""
import numpy as np
import os
import shutil
filename = './Sony_test_list.txt'
def input_gt_ids(filename,outputdir,num_images = 100):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    input_ids = []
    gt_ids = []
    
    for line in lines:
        elements = line.split()
        input_ids.append(elements[0])
        gt_ids.append(elements[1])
        
    
    indx = np.random.choice(len(input_ids),num_images,replace=False)
    print(indx)
    inputids = [input_ids[i] for i in indx]
    gtids = [gt_ids[i] for i in indx]
    
    
    if os.path.isdir(outputdir):
        if os.listdir(outputdir):
            # If not empty, clear the directory
            print(f"Output directory '{outputdir}' is not empty. Clearing directory...")
            shutil.rmtree(outputdir)
            os.makedirs(outputdir)
    else:
        os.makedirs(outputdir)
    
    # Copy the files to outputdir
    for inputid, gtid in zip(inputids, gtids):
        shutil.copy(inputid, os.path.join(outputdir, os.path.basename(inputid)))
        shutil.copy(gtid, os.path.join(outputdir, os.path.basename(gtid)))
    
    return inputids,gtids

inputids, gtids = input_gt_ids(filename, './images_test/')