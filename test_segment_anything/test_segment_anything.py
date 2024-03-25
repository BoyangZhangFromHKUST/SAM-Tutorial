#%%
import torch
from segment_anything import sam_model_registry
import cv2
from segment_anything import SamPredictor
import numpy as np
import time
#%%
import os
# os.chdir('/')
cwd = os.getcwd()
#%%
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

MODEL_TYPE = "vit_l"
CHECKPOINT_PATH = '/home/lifan/Documents/SAM/test_segment_anything/sam_vit_l.pth'
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_predictor = SamPredictor(sam)
#%%
start_time = time.time()
IMAGE_PATH = '3_c.png'
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
mask_predictor.set_image(image_rgb)


box = np.array([250, 100, 1700, 1000])
masks, scores, logits = mask_predictor.predict(
    box=box,
    multimask_output=True
)
end_time = time.time()
print("used time is {}".format(end_time-start_time))
#%%
import supervision as sv
mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS)


detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks
)
detections = detections[detections.area == np.max(detections.area)]

source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[source_image, segmented_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

#%%