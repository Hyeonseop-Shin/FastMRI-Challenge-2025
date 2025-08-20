
import numpy as np
import time
from collections import defaultdict
import os
import cv2 
import h5py
import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils.data.eval_loader import create_eval_loaders
from utils.model.VarNet import VarNet
from utils.model.FIVarNet import FIVarNet
from utils.learning.mask_classifier import MRIClassifier
from utils.common.utils import save_reconstructions
from utils.common.loss_function import SSIMLoss


class SSIM(SSIMLoss):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__(win_size, k1, k2)
            
    def forward(self, X, Y, data_range):
        if len(X.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(X.shape)))
        if len(Y.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(Y.shape)))
            
        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        return S.mean()
    

def forward(args, leaderboard_data_path=None, your_data_path=None, anatomy='all'):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    file_num = 58 if anatomy == 'all' else 29

    leaderboard_data = glob.glob(os.path.join(leaderboard_data_path, '*.h5'))
    your_data = glob.glob(os.path.join(your_data_path, '*.h5'))
    if anatomy != 'all':
        leaderboard_data = [f for f in leaderboard_data if anatomy in os.path.basename(f)]
        your_data = [f for f in your_data if anatomy in os.path.basename(f)]

    if len(leaderboard_data) != file_num:
        raise  NotImplementedError(f'Leaderboard Data Size Should Be {file_num}')
    if len(your_data) != file_num:
        raise  NotImplementedError(f'Your Data Size Should Be {file_num}')           
    
    ssim_total = 0
    idx = 0
    ssim_calculator = SSIM().to(device=device)

    test_list = ['brain_test', 'knee_test'] if anatomy == 'all' else [f"{anatomy}_test"]
    with torch.no_grad():
        for part in test_list:
            for i_subject in range(29):
                l_fname = os.path.join(leaderboard_data_path, part + str(i_subject+1) + '.h5')
                y_fname = os.path.join(your_data_path, part + str(i_subject+1) + '.h5')
                with h5py.File(l_fname, "r") as hf:
                    num_slices = hf['image_label'].shape[0]
                for i_slice in range(num_slices):
                    with h5py.File(l_fname, "r") as hf:
                        target = hf['image_label'][i_slice]
                        mask = np.zeros(target.shape)
                        if part == 'knee_test':
                            mask[target>2e-5] = 1
                        elif part == 'brain_test':
                            mask[target>5e-5] = 1
                        kernel = np.ones((3, 3), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=1)
                        mask = cv2.dilate(mask, kernel, iterations=15)
                        mask = cv2.erode(mask, kernel, iterations=14)
                        
                        target = torch.from_numpy(target).to(device=device)
                        mask = (torch.from_numpy(mask).to(device=device)).type(torch.float)

                        maximum = hf.attrs['max']
                        
                    with h5py.File(y_fname, "r") as hf:
                        recon = hf[args.output_key][i_slice]
                        recon = torch.from_numpy(recon).to(device=device)
                        
                    ssim_total += ssim_calculator(recon*mask, target*mask, maximum).cpu().numpy()
                    idx += 1
            
    return ssim_total/idx

class EvalMRI:
    """Class for evaluating MRI models"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._get_device(args)
        self.model = self._build_model(args)
        self.model.eval()
        self.model2 = self._build_model(args)
        self.model2.eval()
        self.model3 = self._build_model(args)
        self.model3.eval()
        self.mri_classifier = self._select_MRI_classifier()

        self.current_model = None
        self.brain_slice = args.brain_slice
        self.knee_slice = args.knee_slice

        brain_arch = knee_arch = f"{args.model}_f{args.feature_cascades}_i{args.image_cascades}_attn{args.attn_stride}_c{args.chans}_s{args.sens_chans}"
        
        if len(args.brain_special_name.strip()) > 0:
            brain_arch += f"_{args.brain_special_name.strip()}"
        if len(args.knee_special_name.strip()) > 0:
            knee_arch += f"_{args.knee_special_name.strip()}"

        ckpt_path = dict()
        ckpt_path2 = dict()
        ckpt_path3 = dict()
        for slice_num in range(args.brain_slice):
            for acc in [4,8]:
                brain_key = f"acc{acc}-brain-slice{slice_num}"
                ckpt_path[brain_key] = os.path.join(args.result_path,
                                                    brain_arch,
                                                    args.brain_ckpt,
                                                    brain_key,
                                                    "checkpoints/last_model.pt")
                ckpt_path2[brain_key] = os.path.join(args.result_path,
                                                     brain_arch,
                                                     args.brain_ckpt,
                                                     brain_key,
                                                     "checkpoints/last_model.pt")
                ckpt_path3[brain_key] = os.path.join(args.result_path,
                                                     brain_arch,
                                                     args.brain_ckpt,
                                                     brain_key,
                                                     "checkpoints/last_model.pt")
        for slice_num in range(args.knee_slice):
            for acc in [4,8]:
                knee_key = f"acc{acc}-knee-slice{slice_num}"
                ckpt_path[knee_key] = os.path.join(args.result_path,
                                                   knee_arch,
                                                   args.knee_ckpt,
                                                   knee_key,
                                                   "checkpoints/last_model.pt")
                ckpt_path2[knee_key] = os.path.join(args.result_path,
                                                    knee_arch,
                                                    args.knee_ckpt,
                                                    knee_key,
                                                    "checkpoints/last_model.pt")
                ckpt_path3[knee_key] = os.path.join(args.result_path,
                                                    knee_arch,
                                                    args.knee_ckpt,
                                                    knee_key,
                                                    "checkpoints/last_model.pt")
        self.ckpt_path = ckpt_path
        self.ckpt_path2 = ckpt_path2
        self.ckpt_path3 = ckpt_path3


    def _get_device(self, args):
        device = torch.device(f'cuda:{args.GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')    
        torch.cuda.set_device(device)
        print(f'Current device: {device}')
        return device
      

    def _build_model(self, args):

        model_name = args.model.lower()
        assert model_name in ['varnet', 'fivarnet'], f"Unknown model name: {args.model}"


        if model_name == 'varnet':
            model = VarNet(num_cascades=args.feature_cascades, 
                           chans=args.chans, 
                           sens_chans=args.sens_chans)
        elif model_name == 'fivarnet':
            model = FIVarNet(num_feature_cascades=args.feature_cascades,
                             num_image_cascades=args.image_cascades,
                             attn_stride=args.attn_stride,
                             chans=args.chans,
                             sens_chans=args.sens_chans)
        else:
            raise NotImplementedError(f"Model {args.model} is not implemented.")
        
        return model.to(self.device)
        

    def _select_MRI_classifier(self):
        return MRIClassifier()
    

    def load_model(self, model_type):
        assert model_type in self.ckpt_path.keys(), f"Invalid model type: {model_type}"

        # model1 loading
        checkpoint_path = self.ckpt_path[model_type]
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        
        # model2 loading
        checkpoint_path = self.ckpt_path2[model_type]
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model2.load_state_dict(checkpoint['model'])

        # model3 loading
        checkpoint_path = self.ckpt_path3[model_type]
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model3.load_state_dict(checkpoint['model'])
    
        self.current_model = model_type
    

    def reconstruct(self, data_path=None, forward_dir=None, recon_anatomy='all'):
        reconstructions = defaultdict(dict)
        data_loader = create_eval_loaders(data_path=data_path,
                                          args=self.args,
                                          isforward=True,
                                          anatomy=recon_anatomy)

        with torch.no_grad():
            for (mask, kspace, _, _, fnames, slices, num_slices) in tqdm(data_loader):
                kspace = kspace.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                classification = self.mri_classifier.classify(kspace, mask)
                anatomy = classification['anatomy']
                acceleration = classification['acceleration']
                
                model_slice = int((self.brain_slice if anatomy == 'brain' else self.knee_slice) * slices / num_slices)
                model_type = f"{acceleration}-{anatomy}-slice{model_slice}"
                if self.current_model != model_type:
                    self.load_model(model_type)
                    print(f"Loaded model: {model_type}")

                output1 = self.model(kspace, mask, use_grad_ckpt=False)
                output2 = self.model2(kspace, mask, use_grad_ckpt=False)
                output3 = self.model3(kspace, mask, use_grad_ckpt=False)
                output = (output1 + output2 + output3) / 3

                for i in range(output.shape[0]):
                    reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

        for fname in reconstructions:
            reconstructions[fname] = np.stack(
                [out for _, out in sorted(reconstructions[fname].items())]
            )
        
        save_reconstructions(reconstructions, forward_dir, targets=None, inputs=None)
        print(f"Reconstructions saved to {forward_dir}")


    def run_reconstruction(self):
        """Run the reconstruction process for leaderboard evaluation"""
        if not self.args.forward_dir.exists():
            self.args.forward_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # acc4
        acc4_data_path = self.args.leaderboard_path / "acc4"
        acc4_save_path = self.args.forward_dir / "acc4"
        self.reconstruct(data_path=acc4_data_path,
                         forward_dir=acc4_save_path,
                         recon_anatomy='all')

        # acc8
        acc8_data_path = self.args.leaderboard_path / "acc8"
        acc8_save_path = self.args.forward_dir / "acc8"
        self.reconstruct(data_path=acc8_data_path,
                         forward_dir=acc8_save_path,
                         recon_anatomy='all')

        reconstructions_time = time.time() - start_time
        
        print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

        reconstruction_success = reconstructions_time < 3600
        print('Reconstruction Success!') if reconstruction_success else print('Reconstruction Fail!')


    def leaderboard_eval(self):
        """Evaluate the model on the leaderboard dataset"""
        
        # acc4
        acc4_data_path = self.args.leaderboard_path / "acc4/image"
        acc4_save_path = self.args.forward_dir / "acc4"
        SSIM_acc4 = forward(self.args,
                            leaderboard_data_path=acc4_data_path,
                            your_data_path=acc4_save_path)
        # acc8
        acc8_data_path = self.args.leaderboard_path / "acc8/image"
        acc8_save_path = self.args.forward_dir / "acc8"
        SSIM_acc8 = forward(self.args,
                            leaderboard_data_path=acc8_data_path,
                            your_data_path=acc8_save_path)

        print("Leaderboard SSIM : {:.4f}".format((SSIM_acc4 + SSIM_acc8) / 2))
        print("="*10 + " Details " + "="*10)
        print("Leaderboard SSIM (acc4): {:.4f}".format(SSIM_acc4))
        print("Leaderboard SSIM (acc8): {:.4f}".format(SSIM_acc8))
    

    def partial_reconstruction(self, acc, anatomy):
        if not self.args.forward_dir.exists():
            self.args.forward_dir.mkdir(parents=True, exist_ok=True)
        
        assert acc in [4,8]
        assert anatomy in ['brain', 'knee']

        data_path = self.args.leaderboard_path / f"acc{acc}"
        save_path = self.args.forward_dir / f"acc{acc}"
        self.reconstruct(data_path=data_path,
                         forward_dir=save_path,
                         recon_anatomy=anatomy)
        
        print(f"Reconstruciton for {anatomy}-acc{acc} completed!")

    def partial_lb_eval(self, acc, anatomy):

        assert acc in [4,8]
        assert anatomy in ['brain', 'knee']

        data_path = self.args.leaderboard_path / f"acc{acc}/image"
        save_path = self.args.forward_dir / f"acc{acc}/"
        SSIM = forward(self.args,
                       leaderboard_data_path=data_path,
                       your_data_path=save_path,
                       anatomy=anatomy)
        
        print(f"{anatomy}-acc{acc} SSIM = {SSIM}")


    def evaluate(self):
        self.run_reconstruction()
        self.leaderboard_eval()


    def partial_eval(self, acc, anatomy):
        self.partial_reconstruction(acc=acc, anatomy=anatomy)
        self.partial_lb_eval(acc=acc, anatomy=anatomy)