import torch, os
from torchvision import transforms
import pandas as pd
from PIL import Image
import json
import random


def parse_jsonl_file(jsonl_file_path, read_limit=None):
    '''
    return:
    {
        "caption": "A ",
        "image": "pathxxx/image_id.jpg",
        "entities": [
            {
                "entity": "xxx",
                "bbox": [x1, y1, x2, y2]
            },
            ...
        ],
        "image_id": "xxxx",
        "__dj__stats__": {...},
        "text": "",
    }
    '''
    with open(jsonl_file_path, 'r') as file:
        all_infos = []
        for line in file:
            # 解析每一行数据
            try:
                sample = json.loads(line)
                # 提取并打印实体信息
                all_entities = []
                entities = sample.get("entities", [])
                for entity in entities:
                    entity_description = entity.get("entity", "Unknown category")
                    bboxes = entity.get("bboxes", [])
                    for bbox in bboxes:
                        all_entities.append({'entity': entity_description, 'bbox': bbox})
                sample['entities'] = all_entities
                sample['image'] = sample['image'][0]
                all_infos.append(sample)
            except Exception as e:
                print(f"Error: {e}")
                continue
            if read_limit and len(all_infos) >= read_limit:
                break
        return all_infos

# 可视化函数
def visualize_sample(image, mask):
    import matplotlib.pyplot as plt
    import numpy as np
    # 反归一化图像
    image = image * 0.5 + 0.5
    image = image.permute(1, 2, 0).numpy()  # 转换为HxWxC格式

    # 将mask转换为HxW格式
    mask = mask.squeeze().numpy()

    # 创建一个与图像大小相同的RGB图像
    overlay = np.zeros_like(image)
    overlay[:, :, 0] = mask  # 使用红色通道显示mask

    # 将mask叠加在图像上
    combined = image.copy()
    combined[mask > 0.5] = overlay[mask > 0.5]

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # 显示叠加了mask的图像
    axes[1].imshow(combined)
    axes[1].set_title("Image with Mask")
    axes[1].axis("off")
    plt.savefig('sample.png')

def visualize_bbox(image, bbox, color="red", width=3):
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    width, height = image.size
    x1_pixel = int(x1 * width)
    y1_pixel = int(y1 * height)
    x2_pixel = int(x2 * width)
    y2_pixel = int(y2 * height)
    draw.rectangle([x1_pixel, y1_pixel, x2_pixel, y2_pixel], outline="red", width=3)
    
    image.save('bbox.png')

class ControlledTextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False, single_mask=True):
        self.steps_per_epoch = steps_per_epoch

        metadata = parse_jsonl_file(dataset_path)
        print(f"Loaded {len(metadata)} samples in the dataset.")
        
        self.path = []
        self.text = []
        self.entities = []
        for data in metadata:
            entities = data.get("entities", [])
            if len(entities) == 0:
                continue
            self.path.append(data['image'])
            self.text.append(data['caption'])
            self.entities.append(entities)
        self.image_processor = transforms.Compose(
            [
                transforms.Resize(max(height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_processor = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(max(height, width), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.single_mask = single_mask


    def __getitem__(self, index):
        while True:
            try:
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path) # For fixed seed.
                text = self.text[data_id]
                image = Image.open(self.path[data_id]).convert("RGB")
                width, height = image.size

                # process entities
                entities = self.entities[data_id]
                selected_entity = [] if self.single_mask else list(range(len(entities)))
                metas = []
                for i in selected_entity:
                    entity = entities[i]
                    bbox = entity["bbox"]
                    x_min, y_min, x_max, y_max = bbox
                    if x_max - x_min < 0.05 or y_max - y_min < 0.05:
                        raise Exception("bbox too small")
                    x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)
                    mask = torch.zeros((3, height, width))
                    mask[:, y_min:y_max, x_min:x_max] = 1.0
                    mask = self.mask_processor(mask)
                    metas.append({"text": entity["entity"], "mask": mask})

                image = self.image_processor(image)
            except Exception as e:
                print(f"Error: {e}")
                continue
            return {"text": text, "image": image, 'mask': metas[0]['mask'], "control_prompt": metas[0]['text']}

        # visualize_sample(image, mask)
        # image = Image.open(self.path[data_id]).convert("RGB")
        # visualize_bbox(image, entity["bbox"])
        # print(metas[0]['text'])



    def __len__(self):
        return self.steps_per_epoch
