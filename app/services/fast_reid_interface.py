import cv2
import numpy as np
import logging

import torch
import torch.nn.functional as F
from torch.backends import cudnn
import sys
sys.path.append('./reid')
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from typing import Optional

logger = logging.getLogger(__name__)

# GPU Optimization: Enable cudnn benchmark for auto-tuned convolution kernels
cudnn.benchmark = True


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg


def postprocess(features, non_blocking: bool = True):
    """Normalize features and transfer to CPU.
    
    Args:
        features: GPU tensor of features
        non_blocking: Use non-blocking transfer for better pipelining
    """
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    # GPU Optimization: Non-blocking transfer to CPU
    features = features.to('cpu', non_blocking=non_blocking).data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


class FastReIDInterface:
    def __init__(self, config_file, weights_path, device, batch_size=8):
        super(FastReIDInterface, self).__init__()
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.batch_size = batch_size

        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)

        if self.device != 'cpu':
            self.model = self.model.eval().to(device=self.device).half()
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST
        
        # GPU Optimization: CUDA stream for overlapped execution
        self._cuda_stream: Optional[torch.cuda.Stream] = None
        if self.device == 'cuda':
            self._cuda_stream = torch.cuda.Stream()
            
        logger.info(f"FastReIDInterface configured: device={self.device}, batch_size={batch_size}, cuda_stream={self._cuda_stream is not None}")

    def inference(self, image, detections):

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])
            tlbr[1] = max(0, tlbr[1])
            tlbr[2] = min(W - 1, tlbr[2])
            tlbr[3] = min(H - 1, tlbr[3])
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]

            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image.
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)

            # Make shape with a new batch dimension which is adapted for network input
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            # GPU Optimization: Non-blocking transfer to GPU
            patch = patch.to(device=self.device, non_blocking=True).half()

            patches.append(patch)

            if (d + 1) % self.batch_size == 0:
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 2048))

        # GPU Optimization: Use inference_mode for better performance
        with torch.inference_mode():
            for patches in batch_patches:
                # GPU Optimization: Use CUDA stream for overlapped execution
                if self._cuda_stream is not None:
                    with torch.cuda.stream(self._cuda_stream):
                        patches_ = torch.clone(patches)
                        pred = self.model(patches)
                        pred[torch.isinf(pred)] = 1.0
                    self._cuda_stream.synchronize()
                else:
                    patches_ = torch.clone(patches)
                    pred = self.model(patches)
                    pred[torch.isinf(pred)] = 1.0

                # GPU Optimization: Non-blocking postprocess
                feat = postprocess(pred, non_blocking=True)

                nans = np.isnan(np.sum(feat, axis=1))
                if np.isnan(feat).any():
                    for n in range(np.size(nans)):
                        if nans[n]:
                            patch_np = patches_[n, ...]
                            patch_np_ = torch.unsqueeze(patch_np, 0)
                            pred_ = self.model(patch_np_)

                            patch_np = torch.squeeze(patch_np).to('cpu', non_blocking=True)
                            patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                            patch_np = patch_np.numpy()

                features = np.vstack((features, feat))

        return features