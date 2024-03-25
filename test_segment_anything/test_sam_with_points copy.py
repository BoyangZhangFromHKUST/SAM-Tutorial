#%%
import torch
from segment_anything import sam_model_registry
import cv2
from segment_anything import SamPredictor
from segment_anything import SamAutomaticMaskGenerator
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt

#%%
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

#%%
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "/home/lifan/Documents/SAM/test_segment_anything/sam_vit_h.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
#%%
IMAGE_PATH = '/home/lifan/Documents/SAM/test_segment_anything/3_c.png'
result_masks  = []

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

mask_predictor = SamPredictor(sam)
mask_predictor.set_image(image_rgb)

# box = np.array([0, 0, 600, 600])
masks, scores, logits = mask_predictor.predict(
    # box=box,
    point_coords=np.array([[300, 400]]),
    point_labels=np.array([1]),
    multimask_output=True
)
print(scores)
index = np.argmax(scores)
result_masks.append(masks[index])
result_masks = np.array(result_masks)
#%%
# IMAGE_PATH = "3_c.png"
# mask_generator = SamAutomaticMaskGenerator(sam)
# img_bgr = cv2.imread(IMAGE_PATH)
# img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
# result = mask_generator.generate(img_rgb)
# print('result \n',result)
#%%
from inference.models.utils import get_roboflow_model

# image = cv2.imread(<SOURCE_IMAGE_PATH>)
model = get_roboflow_model(model_id="yolov8s-640")

result = model.infer(image_bgr)[0]
#%%
mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
box_annotator = sv.BoxAnnotator()
detections = sv.Detections.from_roboflow(result)
detections.mask = result_masks
print(detections.class_id)
annotated_image = mask_annotator.annotate(image_bgr, detections)
annotated_image = box_annotator.annotate(annotated_image, detections)
cv2.imshow('ai',annotated_image)
cv2.waitKey(0)