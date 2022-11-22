import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
# from extractor import FeatureExtractor
from aistplusplus_api.aist_plusplus.loader import AISTDataset
from smplx import SMPL
import torch
import pickle


def args_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_dir', type=str, default='./aist_plusplus_final/all_musics')
    parser.add_argument('--input_annotation_dir', type=str, default='./aist_plusplus_final')
    parser.add_argument('--smpl_dir', type=str, default='smpl')

    parser.add_argument('--train_dir', type=str, default='./data/aistpp_train_wav')
    parser.add_argument('--test_dir', type=str, default='./data/aistpp_test_full_wav')

    parser.add_argument('--split_train_file', type=str, default='./aist_plusplus_final/splits/crossmodal_train.txt')
    parser.add_argument('--split_test_file', type=str, default='./aist_plusplus_final/splits/crossmodal_test.txt')
    parser.add_argument('--split_val_file', type=str, default='./aist_plusplus_final/splits/crossmodal_val.txt')

    parser.add_argument('--sampling_rate', type=int, default=15360 * 2)
    args = parser.parse_args()
    return args


def main():
    args = args_set()

    file_path = os.path.join('./aist_plusplus_final/motions', 'gBR_sBM_cAll_d04_mBR0_ch01.pkl')
    assert os.path.exists(file_path), f'File {file_path} does not exist!'

    with open(file_path, 'rb') as f:
      data = pickle.load(f)
    smpl_poses = data['smpl_poses']  # (N, 24, 3)
    smpl_scaling = data['smpl_scaling']  # (1,)
    smpl_trans = data['smpl_trans']  # (N, 3)
    print(smpl_poses.shape)

    exit()

    assert os.path.exists(args.train_dir), f'Data does not exist at {args.train_dir}!'

    print(args.input_annotation_dir)
    aist_dataset = AISTDataset(args.input_annotation_dir)
    print(aist_dataset.motion_dir)


if __name__ == "__main__":
    main()
