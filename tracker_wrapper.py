import torch
import numpy as np
import sys
import os

# Add the directory containing this file to sys.path to allow imports
# This works both when run directly and when imported as a module
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import saverloader
from nets.pips import Pips
import torch.nn.functional as F
from utils.basic import print_stats

class PipsTracker:
    def __init__(self, configs, track_length=8):
        self.configs = configs
        self.score_threshold = -1
        current_dir = os.path.dirname(os.path.abspath(__file__))
        init_dir = os.path.join(current_dir, 'reference_model')
        self.model = Pips(stride=4, S=track_length).cuda()
        if init_dir:
            print('loading model from', init_dir)
            _ = saverloader.load(init_dir, self.model)
        self.model.eval()


    def track(self, prevImg, nextImg, prevPts):
        if isinstance(nextImg, list):
            images = [prevImg] + nextImg
        else:
            images = [prevImg, nextImg]

        imgs_np = np.stack(images, axis=0)  # (T, H, W, C)
        imgs_np = imgs_np.astype(np.float32)
        imgs_np = np.transpose(imgs_np, (0, 3, 1, 2))
        rgbs = torch.from_numpy(imgs_np).unsqueeze(0)  # (1, T, C, H, W)
        rgbs = rgbs.cuda().float() # B, S, C, H, W

        B, S, C, H, W = rgbs.shape
        # Store original dimensions for coordinate scaling
        H_orig, W_orig = H, W
        
        rgbs_ = rgbs.reshape(B*S, C, H, W)
        H_, W_ = 360, 640
        rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
        H, W = H_, W_
        rgbs = rgbs_.reshape(B, S, C, H, W)
        
        # Scale points from original image space to resized space
        xy = torch.from_numpy(prevPts).float().unsqueeze(0).cuda()  # (1, N, 2)
        scale_x = W / W_orig  # e.g., 640 / 640 = 1.0 or 640 / 1280 = 0.5
        scale_y = H / H_orig  # e.g., 360 / 480 = 0.75
        xy[:, :, 0] = xy[:, :, 0] * scale_x  # Scale x coordinates
        xy[:, :, 1] = xy[:, :, 1] * scale_y  # Scale y coordinates
        
        with torch.no_grad():
            preds, _, vis_e, _ = self.model(xy, rgbs, iters=6)
        status = (vis_e > self.score_threshold).detach().cpu().numpy().astype(bool)
        trajs_e = preds[-1]
        pred_pts = trajs_e.squeeze(0)[1:].detach().cpu().numpy()
        # Resize pred_pts back to original image space
        pred_pts[..., 0] = pred_pts[..., 0] / scale_x
        pred_pts[..., 1] = pred_pts[..., 1] / scale_y

        return pred_pts, status

if __name__ == '__main__':
    
    configs = {}
    tracker = PipsTracker(configs)
    prevImg = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    nextImg = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    N = 1000  # number of keypoints
    height, width = prevImg.shape[:2]
    prevPts = np.stack([
        np.random.randint(0, width, size=N),
        np.random.randint(0, height, size=N)
    ], axis=-1)
    pred_pts, status = tracker.track(prevImg, nextImg, prevPts)
    print('pred_pts', pred_pts.shape)
    print('status', status.shape)