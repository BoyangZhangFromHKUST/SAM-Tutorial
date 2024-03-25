#%%
import torch
import cv2
import supervision as sv

from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

#%%
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"    
CHECKPOINT_PATH = "sam_vit_b.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
#%%
IMAGE_PATH = "3_c.png"
mask_generator = SamAutomaticMaskGenerator(sam)
img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
result = mask_generator.generate(img_rgb)
print('result \n',result)
#%%
mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(result)
annotated_image = mask_annotator.annotate(img_bgr, detections)
cv2.imshow('ai',annotated_image)
cv2.waitKey(0)
# %%
