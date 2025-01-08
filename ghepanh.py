import torch
import os
import torch.nn.functional as F
from .TPrmbg import RMBG
from torchvision.transforms.functional import normalize
import os, yaml
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
from einops import rearrange, repeat
from PIL import ImageColor
from comfy.model_management import soft_empty_cache
import nodes
import folder_paths
import numpy as np
from PIL import Image
#import nodes
#----------------------------------variable-------------------------------------------> 
REMBG_MODELS = {"RMBG-2.0": {"model_url": "briaai/RMBG-2.0"}}
REMBG_DIR = os.path.join(folder_paths.models_dir, "rembg")
config_path = os.path.join(os.path.dirname(__file__), "./config.yaml")
if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
else:
    raise Exception("config.yaml file is neccessary, plz recreate the config file by downloading it from https://github.com/Fannovel16/ComfyUI-Video-Matting")
current_directory = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
download_url = "briaai/RMBG-2.0"
#----------------------------------function -------------------------------------------> 
def prepare_frames_color(video_frames, bg_color, batch_size):
    orig_num_frames = video_frames.shape[0]
    video_frames = rearrange(video_frames, "n h w c -> n c h w")
    pad_frames = repeat(video_frames[-1:], "1 c h w -> n c h w", n=batch_size - (orig_num_frames % batch_size))
    video_frames = torch.cat([video_frames, pad_frames], dim=0)  
    bg_color = torch.Tensor(ImageColor.getrgb(bg_color)[:3]).float() / 255.
    bg_color = repeat(bg_color, "c -> n c 1 1", n=batch_size) 
    return video_frames, orig_num_frames, bg_color

def resize_tensor(image_tensor, target_height, target_width):
    return F.interpolate(image_tensor, size=[target_height, target_width], mode='bilinear', align_corners=False)


def easySave(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):
    """Save or Preview Image"""
    from nodes import PreviewImage, SaveImage
    if output_type in ["Hide", "None"]:
        return list()
    elif output_type in ["Preview", "Preview&Choose"]:
        filename_prefix = 'easyPreview'
        results = PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
    else:
        results = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
#----------------------------------node ghép ảnh-------------------------------------------> 
class ghepanh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sub_image": ("IMAGE",),
                "bg_image": ("IMAGE",),
                "mirror": ("BOOLEAN", {"default": False}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 2048, "step": 8}),
                "position": ("INT", {"default": 50, "min": 0, "max": 100, "step": 5}),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": { "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ("IMAGE","MASK")
    FUNCTION = "generate_ghepanh"
    OUTPUT_NODE = True
    CATEGORY = "Tupham"

    def generate_ghepanh(self, sub_image, bg_image,mirror,height,position,save_prefix,image_output = "Preview",prompt=None, extra_pnginfo=None):  
        
        repo_id = REMBG_MODELS['RMBG-2.0']['model_url']
        model_path = os.path.join(REMBG_DIR, 'RMBG-2.0')
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt"])
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(model_path, trust_remote_code=True)
        torch.set_float32_matmul_precision('high')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()        
        video_frames, orig_num_frames, bg_color = prepare_frames_color(sub_image, "green", 4)
        bg_frames, _, _ = prepare_frames_color(bg_image, "green", 4)
        bg_color = bg_color.to(device)
        h, w = video_frames.shape[2:4]
        w = int(w * height / h)
        h = height
        h1, w1 = bg_frames.shape[2:4] 
        w1 = int(w1 * height / h1)
        h1 = height
        print(h1,w1)
        x = int(w1 * (100 - position) / 100)
        bg=resize_tensor(bg_frames,h1,w1) 
        new_images = list()
        masks = list()
        for i in range(video_frames.shape[0] // 4):
            batch_imgs = video_frames[i*4:(i+1)*4].to(device)
            batch_imgs = resize_tensor(batch_imgs,h,w)
            if mirror:
                batch_imgs = torch.flip(batch_imgs, dims=[3])
            batch_imgs = F.pad(batch_imgs, (int(w1-w/2), int(w1-w/2), 0, 0), mode='constant', value=0)
            batch_imgs = batch_imgs[:, :, 0:h1, x:w1+x]
            resized_input = batch_imgs
            resized_input = F.interpolate(resized_input, size=[h1,w1], mode='bilinear')
            resized_input = normalize(resized_input,[0.5,0.5,0.5],[1.0,1.0,1.0])
            
            #orig_im = tensor2pil(batch_imgs)
            input_tensor = resize_tensor(batch_imgs,1024,1024)
            with torch.no_grad():
                preds = model(input_tensor)[-1].sigmoid().cpu()
                mask = preds[0].squeeze()
                mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-9)  # Chuẩn hóa giá trị mask

                # Thêm 2 chiều batch và channel để có dạng (N, C, H, W)
                mask = mask.unsqueeze(0).unsqueeze(0).float()
                print(mask)
                # Thực hiện nội suy
                mask_ten = F.interpolate(mask, size=[h1, w1], mode='bilinear', align_corners=False)
                mask_ten = mask_ten.squeeze(0).squeeze(0)
                mask_ten = mask_ten.to(device)  # Đưa mask về thiết bị phù hợp
                new_im = batch_imgs * mask_ten + bg.to(device) * (1 - mask_ten)
                print(new_im)
                new_images.append(new_im)
                masks.append(mask_ten)

        torch.cuda.empty_cache()
        new_images = rearrange(torch.cat(new_images), "n c h w -> n h w c")[:orig_num_frames].float().detach()
        masks = torch.cat(masks)[:orig_num_frames].squeeze(1).float().detach()
        results = easySave(new_images, save_prefix, image_output, prompt, extra_pnginfo)
        return {"ui": {"images": results},
            "result": (new_images, masks)}

#----------------------------------node run node selected-------------------------------------------> 
class Runnodeselected:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "string": ("STRING", ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ()
    FUNCTION = "process"
    CATEGORY = 'Tupham'

    def process(self, input_data):
        # Xử lý đầu vào và thực hiện hành động, ví dụ: ghi log
        print(f"Received input: {input_data}")
        return ()


#----------------------------------node multiPrompt-------------------------------------------> 
class AreaCondition_v2:
    def __init__(self) -> None:
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", ),
                "resolutionX":("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64}),
                "resolutionY":("INT", {"default": 1024, "min": 512, "max": 4096, "step": 64}),
                "prompt":("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "index":("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "x":("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                "y":("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                "width":("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                "height":("INT", {"default": 0, "min": 0, "max": 4096, "step": 64}),
                "strength":("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }
    
    RETURN_TYPES = ("CONDITIONING", "INT", "INT")
    RETURN_NAMES = (None, "resolutionX", "resolutionY")
    FUNCTION = "doStuff_v2"
    CATEGORY = "Tupham"

    def doStuff_v2(self,clip,resolutionX,resolutionY,prompt,index,x,y,width,height,strength,extra_pnginfo, unique_id):
        c = []
        values = []
        resolutionX = 512
        resolutionY = 512
        for node in extra_pnginfo["workflow"]["nodes"]:
            if node["id"] == int(unique_id):
                values = node["properties"]["values"]
                resolutionX = node["properties"]["width"]
                resolutionY = node["properties"]["height"]
                break
        for arg in values:        
            x, y = arg[1], arg[2]
            w, h = arg[3], arg[4]
            text = arg[0]
            tokens = clip.tokenize(text)
            output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = output.pop("cond") 
            condition = [cond, output],
            # If fullscreen   
            # if (x == 0 and y == 0 and w == resolutionX and h == resolutionY):
            #     c.append([cond, output.copy()])
            #     continue            
            if x+w > resolutionX:
                w = max(0, resolutionX-x)          
            if y+h > resolutionY:
                h = max(0, resolutionY-y)
            if w == 0 or h == 0: continue
            n = [cond, output.copy()]
            n[1]['area'] = (h // 8, w // 8, y // 8, x // 8)
            n[1]['strength'] = arg[5]
            n[1]['min_sigma'] = 0.0
            n[1]['max_sigma'] = 99.0               
            c.append(n) 
            soft_empty_cache()       
        return (c, resolutionX, resolutionY)

#----------------------------------node multiPrompt-------------------------------------------> 

class ConditionUpscale():
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "scalar": ("INT", {"default": 2, "min": 1, "max": 100, "step": 0.5}),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = 'upscale'
    CATEGORY = "Tupham"
    
    def upscale(self, conditioning, scalar):
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            if 'area' in n[1]:               
                n[1]['area'] = tuple(map(lambda x: ((x*scalar + 7) >> 3) << 3, n[1]['area']))
            c.append(n)
        return (c, )
    
class MultiLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples_to": ("LATENT",),
                "samples_from0": ("LATENT",),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "composite"
    CATEGORY = "Tupham"

    def composite(self, samples_to, extra_pnginfo, unique_id, **kwargs):
        values = []
        for node in extra_pnginfo["workflow"]["nodes"]:
            if node["id"] == int(unique_id):
                values = node["properties"]["values"]
                break   
        samples_out = samples_to.copy()
        s = samples_to["samples"].clone()
        samples_to = samples_to["samples"]
        k = 0
        for arg in kwargs:
            if k > len(values): break
            x =  values[k][0] // 8
            y = values[k][1] // 8
            feather = values[k][2] // 8
            samples_from = kwargs[arg]["samples"]
            if feather == 0:
                s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
            else:
                samples_from = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x]
                mask = torch.ones_like(samples_from)
                for t in range(feather):
                    if y != 0:
                        mask[:,:,t:1+t,:] *= ((1.0/feather) * (t + 1))

                    if y + samples_from.shape[2] < samples_to.shape[2]:
                        mask[:,:,mask.shape[2] -1 -t: mask.shape[2]-t,:] *= ((1.0/feather) * (t + 1))
                    if x != 0:
                        mask[:,:,:,t:1+t] *= ((1.0/feather) * (t + 1))
                    if x + samples_from.shape[3] < samples_to.shape[3]:
                        mask[:,:,:,mask.shape[3]- 1 - t: mask.shape[3]- t] *= ((1.0/feather) * (t + 1))
                rev_mask = torch.ones_like(mask) - mask
                s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] = samples_from[:,:,:samples_to.shape[2] - y, :samples_to.shape[3] - x] * mask + s[:,:,y:y+samples_from.shape[2],x:x+samples_from.shape[3]] * rev_mask
            k += 1
        samples_out["samples"] = s
        return (samples_out,)


NODE_CLASS_MAPPINGS = {
    "ghepanh":ghepanh,
    "AreaCondition_v2":AreaCondition_v2,
    "ConditionUpscale":ConditionUpscale,
    "MultiLatent":MultiLatent,
    "Runnodeselected":Runnodeselected,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ghepanh":"Ghép Ảnh",
    "AreaCondition_v2":"Multi Prompt v2.0",
    "ConditionUpscale":"Condition Upscale",
    "MultiLatent":"Multi sampler",
    "Runnodeselected":"Run node selected",
}
