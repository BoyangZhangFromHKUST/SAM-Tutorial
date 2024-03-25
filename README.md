# Tutorial1

### Get Started:

Install Segment Anything:

`pip install git+https://github.com/facebookresearch/segment-anything.git`

Install some necessary python packages:

`pip install opencv-python pycocotools matplotlib onnxruntime onnx`

Install supervision (if you are following the tutorial of roboflow, please install the *`supervison==0.5.0`*):

`pip install supervision`

Download the checkpoint:

vit_h:

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

vit_l:

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

vit_b:

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

h,l,b means huge, large, and base. If you want to test, the base one is enough. But I would like to use the huge one to avoid any bad consequences. 
