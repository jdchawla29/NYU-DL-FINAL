
from utils import get_lightning_module_dataloader

import torch
from torchvision import transforms

from hydra import compose, initialize
from omegaconf import DictConfig

from pathlib import Path
import argparse
from models import STDiffPipeline, STDiffDiffusers
from diffusers import DDPMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    return args.config

def main(cfg : DictConfig) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    ckpt_path = cfg.TestCfg.ckpt_path
    r_save_path = cfg.TestCfg.test_results_path
    if not Path(r_save_path).exists():
        Path(r_save_path).mkdir(parents=True, exist_ok=True) 

    # Load stdiff model
    stdiff = STDiffDiffusers.from_pretrained(ckpt_path, subfolder='stdiff').eval()

    # Print the number of parameters
    num_params = sum(p.numel() for p in stdiff.parameters() if p.requires_grad)
    print('Number of parameters is: ', num_params)

    # Init scheduler
    scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder = 'scheduler')

    stdiff_pipeline = STDiffPipeline(stdiff, scheduler).to(device)
    
    if not accelerator.is_main_process:
        stdiff_pipeline.disable_pgbar()

    _, test_loader = get_lightning_module_dataloader(cfg)
    stdiff_pipeline, test_loader = accelerator.prepare(stdiff_pipeline, test_loader)

    To = cfg.Dataset.num_observed_frames
    Tp = cfg.Dataset.num_predict_frames

    idx_o = torch.linspace(0, To-1, To).to(device)
    idx_p = torch.linspace(To, Tp+To-1, Tp).to(device)
    
    def get_resume_batch_idx(r_save_path):
        save_path = Path(r_save_path)
        saved_preds = sorted(list(save_path.glob('Pred*')))
        saved_batches = [int(str(p.name).split('_')[1]) for p in saved_preds]
        try:
            return saved_batches[-1]
        except IndexError:
            return -1
    resume_batch_idx = get_resume_batch_idx(r_save_path)

    print('number of test batches: ', len(test_loader))
    print('resume batch index: ', resume_batch_idx)

    def visualize_batch_clips(past_batch, future_batch, pred_batch, file_dir):
        if not Path(file_dir).exists():
            Path(file_dir).mkdir(parents=True, exist_ok=True) 
        def save_clip(clip, file_name):
            imgs = []
            for i in range(clip.shape[0]):
                img = transforms.ToPILImage()(clip[i, ...])
                imgs.append(img)

            imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:], loop = 0)

        original_clip = torch.cat([past_batch, future_batch], dim = 1) # shape (N, 2*clip_length, C, H, W)
        pred_clip = torch.cat([past_batch, pred_batch], dim = 1) # shape (N, 2*clip_length, C, H, W)
        batch = torch.cat([original_clip, pred_clip], dim = -1) # shape (N, 2*clip_length, C, H, 2W)
        batch = batch.cpu()
        N = batch.shape[0]
        for n in range(N):
            clip = batch[n, ...]
            file_name = file_dir.joinpath(f'clip_{n}.gif')
            save_clip(clip, file_name)

    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader))
        progress_bar.set_description(f"Testing...") 

        for idx, batch in enumerate(test_loader):
            if idx > resume_batch_idx: # Resume test
                Vo, Vp, Vo_last_frame, _, _ = batch

                pred = stdiff_pipeline(
                    Vo,
                    Vo_last_frame,
                    idx_o,
                    idx_p,
                    num_inference_steps=100,
                    to_cpu=False
                )  # Torch Tensor (N, Tp, C, H, W), range (0, 1)

                Vo = (Vo / 2 + 0.5).clamp(0, 1)
                Vp = (Vp / 2 + 0.5).clamp(0, 1)

                g_pred = accelerator.gather(pred)
                g_Vo = accelerator.gather(Vo)
                g_Vp = accelerator.gather(Vp)

                if accelerator.is_main_process:
                    progress_bar.update(1)
                    # only for testing, when you have the future frames
                    visualize_batch_clips(g_Vo, g_Vp, g_pred, file_dir=Path(r_save_path).joinpath(f'Pred_{idx}'))

                    # we want the last frame predicted by the model for the segementation task
                    for i in range(g_pred.shape[0]):
                        img = transforms.ToPILImage()(g_pred[i,-1,...])
                        img.save(Path(r_save_path).joinpath(f'Pred_{idx}/img_{i}.png'))

                    del g_Vo
                    del g_Vp
                    del g_pred
        
        progress_bar.close()
                     
    print("Inference finished")

if __name__ == '__main__':
    config_path = Path(parse_args())
    initialize(version_base=None, config_path=str(config_path.parent))
    cfg = compose(config_name=str(config_path.name))

    main(cfg)