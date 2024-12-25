"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
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
            shuffle=False
        )
    
    elif args.dataset=='chexpert':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
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
    all_metrics = []

    for idx, img in enumerate(datal):
        model_kwargs = {}
     #   img = next(data)  # should return an image from the dataloader "data"
        print('img', img[0].shape, img[1])
        if args.dataset=='brats':
          Labelmask = th.where(img[3] > 0, 1, 0)
          number = f"{str(idx).zfill(4)}"
          if img[2]==0:
              continue    #take only diseased images as input
              
        #   viz.image(visualize(img[0][0, 0, ...]), opts=dict(caption="img input 0"))
        #   viz.image(visualize(img[0][0, 1, ...]), opts=dict(caption="img input 1"))
        #   viz.image(visualize(img[0][0, 2, ...]), opts=dict(caption="img input 2"))
        #   viz.image(visualize(img[0][0, 3, ...]), opts=dict(caption="img input 3"))
        #   viz.image(visualize(img[3][0, ...]), opts=dict(caption="ground truth"))
        else:
        #   viz.image(visualize(img[0][0, ...]), opts=dict(caption="img input"))
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
        #   viz.image(visualize(sample[0,0, ...]), opts=dict(caption="sampled output0"))
        #   viz.image(visualize(sample[0,1, ...]), opts=dict(caption="sampled output1"))
        #   viz.image(visualize(sample[0,2, ...]), opts=dict(caption="sampled output2"))
        #   viz.image(visualize(sample[0,3, ...]), opts=dict(caption="sampled output3"))
          difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
        #   viz.heatmap(visualize(difftot), opts=dict(caption="difftot"))
          
        elif args.dataset=='chexpert':
        #   viz.image(visualize(sample[0, ...]), opts=dict(caption="sampled output"+str(name)))
          diff=abs(visualize(org[0, 0,...])-visualize(sample[0,0, ...]))
          diff=np.array(diff.cpu())
        #   viz.heatmap(np.flipud(diff), opts=dict(caption="diff"))


        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        seg_path = img[1]["seg_path"][0]
        metrics = save_validation_results(
            sample, 
            org, 
            seg_path,
            number,  # Already defined in your code
            output_dir="../validation_results",
            dataset_type=args.dataset
        )
        all_metrics.append(metrics)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    
    total_dice_score = sum(metric['dice_score'] for metric in all_metrics)
    num_metrics = len(all_metrics)
    average_dice_score = total_dice_score / num_metrics if num_metrics > 0 else 0

    logger.log("Average Dice Score:", average_dice_score)
    np.save(os.path.join(args.output_dir, "all_metrics.npy"), all_metrics)
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
        dataset='brats',
        output_dir='./validation_results'
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def calculate_dice(pred, target):
    """Calculate Dice coefficient between prediction and target masks"""
    smooth = 1e-5
    pred = pred > 0.5  # Convert difference map to binary mask with threshold
    target = target > 0  # Binary segmentation mask
    intersection = (pred & target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def save_validation_results(sample, org, seg_path, number, output_dir, dataset_type='brats'):
    """Save validation results with anomaly detection dice score"""
    import os
    import numpy as np
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    
    sample_np = sample.cpu().numpy()
    org_np = org.cpu().numpy()
    
    if dataset_type == 'brats':
        os.makedirs(output_dir + f"/{number}", exist_ok=True)
        # Calculate difference map
        diff = np.abs(org_np[0, :5, ...] - sample_np[0, ...]).sum(axis=0)
        # Get segmentation mask (assuming it's in org_np[0, -1, ...])
        seg_mask = np.load(seg_path)
        max = np.max(seg_mask)
        min = np.min(seg_mask)
        if max - min < 1e-6:
            seg_mask = np.zeros_like(seg_mask)
        else:
            seg_mask = (seg_mask - min)  / (max - min)
        
        seg_image = np.zeros((1,256,256))
        seg_image[:,8:-8,8:-8]=seg_mask
        # Calculate dice between difference map and segmentation
        dice_score = calculate_dice(diff, seg_image)
        
        # Save visualizations
        diff_img = visualize(diff)
        seg_img = visualize(seg_image)
        Image.fromarray((diff_img * 255).astype(np.uint8)).save(
            os.path.join(output_dir, number, f"{number}_difference.png"))
        Image.fromarray((seg_img * 255).astype(np.uint8)).save(
            os.path.join(output_dir, number, f"{number}_segmentation.png"))
        
        # Save metrics
        metrics = {
            'dice_score': dice_score,
            'number': number
        }
        np.save(os.path.join(output_dir, number, f"{number}_metrics.npy"), metrics)
    
    return metrics

if __name__ == "__main__":
    main()

