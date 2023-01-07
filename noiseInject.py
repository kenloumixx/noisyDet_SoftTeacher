import copy
import json
from collections import defaultdict
import random
from random import uniform
coco_class = (1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,\
              21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,\
              41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,\
              61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,\
              81,82,84,85,86,87,88,89,90)

# path = "./data/coco/annotations/instances_train2017.json"
path = "/home/cvlab11/project/noisyDet/SoftTeacher/data/coco/annotations/instances_train2017.json"

with open(path,"r") as json_file:
    original_anns = json.load(json_file)

anns = copy.deepcopy(original_anns) # 그냥 대입하면 얕은 복사
ann_len = len(anns['annotations'])

noise_ratio = 0.1

"""### missing anno ###"""
random.seed(0)
num_noise = int(ann_len * noise_ratio) # 개 만큼 버림

idx = list(range(ann_len))
random.shuffle(idx)
noise_idx = idx[:num_noise]
noise_idx = sorted(noise_idx,reverse=True)

for index in noise_idx:
    del anns['annotations'][index]
    
"""### bbox ###"""
random.seed(1)
image_ids_annotations = defaultdict(list)

for ann in anns['annotations']:
    image_id = ann['image_id']
    image_ids_annotations[image_id].append(ann)
    
# 이미지 아이디 별로 이미지 크기 모으기
image_ids_to_wh = defaultdict(list)

for ann in anns['images']:
    image_ids_to_wh[ann['id']].append(ann['width'])
    image_ids_to_wh[ann['id']].append(ann['height'])
    
data_len = len(anns["annotations"])
# data_len = ann_len
num_noise = int(noise_ratio*ann_len)

idx = list(range(data_len))
random.shuffle(idx)
noise_idx = idx[:num_noise]

for i in noise_idx:
    x,y,w,h = anns['annotations'][i]['bbox']
    img_w,img_h = image_ids_to_wh[anns["annotations"][i]['image_id']]
    
    # 랜덤한 두 점 뽑기
    xlist = sorted([uniform(x-w/2,x+(3/2)*w),uniform(x-w/2,x+(3/2)*w)])
    ylist = sorted([uniform(y-h/2,y+(3/2)*h),uniform(y-h/2,y+(3/2)*h)])
    # 클램핑 및 반올림
    xlist = list(map(lambda x: min(max(0,round(x,2)),img_w),xlist))
    ylist = list(map(lambda y: min(max(0,round(y,2)),img_h),ylist))
    
    new_x = xlist[0]
    new_y = ylist[0]
    new_w = round(xlist[1]-xlist[0],2)
    new_h = round(ylist[1]-ylist[0],2)
    
    anns["annotations"][i]['bbox'] = [new_x, new_y, new_w, new_h]
    
"""### class ###"""
random.seed(2)
data_len = len(anns["annotations"])
# data_len = ann_len
num_noise = int(noise_ratio*ann_len)
num_category = max([anns["annotations"][i]["category_id"] for i in range(data_len)])

idx = list(range(data_len))
random.shuffle(idx)
noise_idx = idx[:num_noise]

for i in noise_idx:
    anns["annotations"][i]["category_id"] = random.choice(coco_class)
    
"""### bg fp ###"""
random.seed(3)
num_of_images = len(anns['images']) # 이미지 개수
max_id_plus_one = max([ann['id'] for ann in anns["annotations"]]) + 1 # max id 바로 쓰면 아래 i = 0 일 때 한번 겹침

image_idx = 0
for i in range(round(ann_len * noise_ratio / (1-noise_ratio))): # 개 만큼 만들면 됨.
    image_idx = i % num_of_images
    image_h = anns['images'][image_idx]['height']
    image_w = anns['images'][image_idx]['width']
    
    # 이미지 크기 내에서 랜덤한 점 두개 뽑아서 sort 하고 left top 이 x y 이고 차이가 w h 임
    xlist = sorted([uniform(0,image_w),uniform(0,image_w)])
    ylist = sorted([uniform(0,image_h),uniform(0,image_h)])
    xlist = list(map(lambda x: round(x,2),xlist))
    ylist = list(map(lambda y: round(y,2),ylist))
    
    new_x = xlist[0]
    new_y = ylist[0]
    new_w = round(xlist[1]-xlist[0],2)
    new_h = round(ylist[1]-ylist[0],2)
    
    area = new_w * new_h
    iscrowd = 0
    image_id = anns['images'][image_idx]['id']
    bbox = [new_x, new_y, new_w, new_h]
    category_id = random.choice(coco_class)
    id = max_id_plus_one + i # 안겹치게 최대 값 위에서 하나씩 더해줌
    
    anns['annotations'].append({'area': area, 'iscrowd': iscrowd, 'image_id': image_id, 'bbox': bbox, 'category_id': category_id, 'id': id})
    
"""### 저장 ###"""
# path = "~/project/noisyDet/SoftTeacher/data/coco/annotations/instances_train2017.json"
# target_path = "./data/coco/annotations/newmixnoisy40_instances_train2017.json"
target_path = "/home/cvlab11/project/noisyDet/SoftTeacher/data/coco/annotations/newmixnoisy10_instances_train2017.json"
with open(target_path,"w") as outfile:
    json.dump(anns, outfile, indent=4)