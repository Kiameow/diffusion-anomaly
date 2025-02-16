"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.dataset=='brats':
        ds = BRATSDataset(args.data_dir, test_flag=True)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False)
    
    elif args.dataset=='chexpert':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
        datal = iter(data)
        
    elif args.dataset=='opmed':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            dataset_type="opmed"
        )
        datal = iter(data)
    
    elif args.dataset=='ideas':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            dataset_type="ideas"
        )
        datal = iter(data)
   
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path)
    )

    print('loaded classifier')
    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    print('pmodel', p1, 'pclass', p2)


    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()


    def cond_fn(x, t,  y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            a=th.autograd.grad(selected.sum(), x_in)[0]
            return  a, a * args.classifier_scale



    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []

    for img in datal:

        model_kwargs = {}
     #   img = next(data)  # should return an image from the dataloader "data"
        print('img', img[0].shape, img[1])
        if args.dataset=='brats':
          Labelmask = th.where(img[3] > 0, 1, 0)
          number=img[4][0]
          if img[2]==0:
              continue    #take only diseased images as input
        else:
          print('img1', img[1])
          number=img[1]["path"]
          print('number', number)

        if args.class_cond:
            classes = th.randint(
                low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
            print('y', model_kwargs["y"])
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        print('samplefn', sample_fn)
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()
        sample, x_noisy, org = sample_fn(
            model_fn,
            (args.batch_size, 4, args.image_size, args.image_size), img, org=img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise_level=args.noise_level
        )
        end.record()
        th.cuda.synchronize()
        th.cuda.current_stream().synchronize()


        print('time for 1000', start.elapsed_time(end))

        if args.dataset=='brats':
            difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
          
        elif args.dataset=='chexpert':
            diff=abs(visualize(org[0, 0,...])-visualize(sample[0,0, ...]))
            diff=np.array(diff.cpu())
          
        elif args.dataset=='opmed':
            diff=abs(visualize(org[0, 0,...])-visualize(sample[0,0, ...]))
            diff=np.array(diff.cpu())

        elif args.dataset=='ideas':
            diff=abs(visualize(org[0, 0,...])-visualize(sample[0,0, ...]))
            diff=np.array(diff.cpu())
            
        # 确保使用正确的设备
        org = org.to(dist_util.dev())
        sample = sample.to(dist_util.dev())

        # 遍历批次中的每个样本
        for i in range(args.batch_size):
            # 获取样本唯一标识
            if args.dataset == 'brats':
                number = f"{img[4][i].item()}"  # BraTS病例编号
            else:
                path = img[1]['path'][i]
                number = os.path.splitext(os.path.basename(path))[0]

            # 创建样本子目录
            sample_dir = os.path.join(args.output_dir, f"sample_{number}_i{i}")
            os.makedirs(sample_dir, exist_ok=True)

            # 提取单个样本数据
            original = org[i].detach().cpu()
            generated = sample[i].detach().cpu()
            difference = (original - generated).abs()
            difference = difference / difference.max()

            # 根据数据集类型保存
            if args.dataset == 'brats':
                # 保存每个通道
                for c in range(4):
                    save_image(original[c], os.path.join(sample_dir, f'original_ch{c}.png'), normalize=False)
                    save_image(generated[c], os.path.join(sample_dir, f'generated_ch{c}.png'), normalize=False)
                    save_image(difference[c], os.path.join(sample_dir, f'diff_ch{c}.png'), normalize=False)
                # 保存通道总差异
                save_image(difference.sum(dim=0), os.path.join(sample_dir, 'diff_sum.png'), normalize=False)
            else:
                # 单通道数据集
                save_image(original, os.path.join(sample_dir, 'original.png'), normalize=False)
                save_image(generated, os.path.join(sample_dir, 'reconstructed.png'), normalize=False)
                save_image(difference, os.path.join(sample_dir, 'diff.png'), normalize=False)

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats'
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--output_dir', type=str, default='sample_results', help='Output directory for saved images')
    return parser

if __name__ == "__main__":
    main()

