
import argparse
import os
import sys
from pathlib import Path

from utils.EvalMRI import EvalMRI
from utils.common.utils import seed_fix

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate MRI models on FastMRI challenge',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')
    parser.add_argument("--target_key", type=str, default='image_label', help='Name of target key')
    parser.add_argument("--max_key", type=str, default='max', help='Name of max key')
    parser.add_argument('-key', '--output_key', type=str, default='reconstruction')

    # Data paths
    parser.add_argument('--leaderboard_path', type=Path, default='/root/Data/leaderboard/', help='Path to leaderboard data')
    parser.add_argument('--forward_dir', type=Path, default='/root/FastMRI_challenge/reconstructions_leaderboard', help='Directory for saving reconstructions')
    
    # Model checkpoints
    parser.add_argument('--result_path', type=Path, default='/root/FastMRI_challenge/results',
                        help='result path')
    parser.add_argument('--brain_special_name', type=str, default='base',
                        help='Brain model specification')
    parser.add_argument('--knee_special_name', type=str, default='base',
                        help='Knee model specification')
    parser.add_argument('--brain_slice', type=int, default=1,
                        help='Brain slice moe num')
    parser.add_argument('--knee_slice', type=int, default=1,
                        help='Knee slice moe num')
    parser.add_argument('--brain_ckpt', type=Path, 
                        default='epoch5_fold5_slice1_retrain2', help='Checkpoint for brain model')
    parser.add_argument('--knee_ckpt', type=Path, 
                        default='epoch5_fold5_slice1_retrain2', help='Checkpoint for knee model')
  
    # model hyperparameter
    parser.add_argument('--model', type=str, default='fivarnet', choices=['varnet', 'fivarnet'], help='Model type to evaluate')
    parser.add_argument('-f', '--feature_cascades', type=int, default=8, help='Number of cascades | Should be less than 12')
    parser.add_argument('-i', '--image_cascades', type=int, default=2, help='Number of cascades | Should be less than 12')
    parser.add_argument('-a', '--attn_stride', type=int, default=0, help='Applying block-wise attention for feature processor')
    parser.add_argument('--chans', type=int, default=32, help='Number of channels for cascade U-Net | 18 in original varnet')
    parser.add_argument('--sens-chans', type=int, default=8, help='Number of channels for sensitivity map U-Net | 8 in original varnet')


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()


    task = EvalMRI(args)
    task.run_reconstruction()