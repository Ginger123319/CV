import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                 facecolor=(0, 0, 0, 0), lw=2))


def show_res(masks, scores, input_point, input_label, input_box, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis('off')
        # plt.show()
        plt.savefig('example_plot{}.png'.format(i))


def show_res_multi(masks, scores, input_point, input_label, input_box, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.show()

from segment_anything import sam_model_registry, SamPredictor
sam_checkpoint = "pretrained_checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# test1
# image = cv2.imread('demo/input_imgs/example0.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# input_box = np.array([[4,13,1007,1023]])
# input_point, input_label = None, None
# predictor.set_image(image)
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     box = input_box,
#     multimask_output=False,
#     hq_token_only= False,
# )
# show_res(masks,scores,input_point, input_label, input_box, image)

# test2
# image = cv2.imread('demo/input_imgs/example1.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# input_box = np.array([[306, 132, 925, 893]])
# input_point, input_label = None, None
# predictor.set_image(image)
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     box = input_box,
#     multimask_output=False,
#     hq_token_only= True,
# )
# show_res(masks,scores,input_point, input_label, input_box, image)

# test3
# image = cv2.imread('demo/input_imgs/example2.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# input_point = np.array([[495,518],[217,140]])
# input_label = np.ones(input_point.shape[0])
# input_box = None
# predictor.set_image(image)
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     box = input_box,
#     multimask_output=True
# )
# show_res(masks,scores,input_point, input_label, input_box, image)

# test4
image = cv2.imread('demo/input_imgs/example5.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_point = np.array([[373,363], [452, 575]])
input_label = np.ones(input_point.shape[0])
input_box = None
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box = input_box,
    multimask_output=True
)
show_res(masks,scores,input_point, input_label, input_box, image)


