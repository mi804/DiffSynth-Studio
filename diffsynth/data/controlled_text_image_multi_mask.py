import torch, os
from torchvision import transforms
from PIL import Image
import json
import random
import torch.nn.functional as F


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

class ControlledTextImageDatasetMultiMask(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False, max_mask=5, drop_prob=0.1):
        self.steps_per_epoch = steps_per_epoch

        metadata = parse_jsonl_file(dataset_path)

        self.path = []
        self.text = []
        self.entities = []
        for data in metadata:
            entities = data.get("entities", [])
            if len(entities) == 0 or len(entities) > max_mask:
                continue
            self.path.append(data['image'])
            self.text.append(data['caption'])
            self.entities.append(entities)
        print(f"Loaded {len(self.path)} samples in the dataset.")
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
                # transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.max_mask = max_mask
        self.height = height
        self.width = width
        self.drop_prob = drop_prob
        print('image size:', self.height, self.width)
        print('drop prob:', self.drop_prob)
        print('max mask:', self.max_mask)
        print(f'image size: {self.height} x {self.width}')

    def __getitem__(self, index):
        while True:
            try:
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path) # For fixed seed.
                text = self.text[data_id]
                entities = self.entities[data_id]
                selected_entities = random.sample(range(len(entities)), k=self.max_mask) if len(entities) > self.max_mask else range(len(entities))
                # if len(entities) > self.max_mask:
                #     continue
                image = Image.open(self.path[data_id]).convert("RGB")
                width, height = image.size

                # process entities
                masks = []
                control_prompts = []
                for i in selected_entities:
                    entity = entities[i]
                    bbox = entity["bbox"]
                    x_min, y_min, x_max, y_max = bbox
                    if x_max - x_min < 0.05 or y_max - y_min < 0.05:
                        # print("bbox too small")
                        continue
                    x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)
                    mask = torch.zeros((1, height, width))
                    mask[:, y_min:y_max, x_min:x_max] = 1.0
                    mask = self.mask_processor(mask)
                    masks.append(mask)
                    control_prompts.append(entity["entity"])
                if len(masks) == 0:
                    continue
                image = self.image_processor(image)
            except Exception as e:
                print(f"Error: {e}")
                continue
            num_masks = len(masks)
            # pad masks
            # if num_masks < self.max_mask:
            #     masks += [torch.zeros((1, self.height, self.width))] * (self.max_mask - num_masks)
            #     control_prompts += [""] * (self.max_mask - len(control_prompts))
            masks = torch.stack(masks)
            text = text if random.random() >= self.drop_prob else ""
            # for i in range(len(control_prompts)):
            #     control_prompts[i] = control_prompts[i] if random.random() > self.drop_prob else ""

            return {"text": text, "image": image, "masks": masks, "control_prompts": control_prompts, "num_masks": num_masks+1}

        # visualize_sample(image, mask)
        # image = Image.open(self.path[data_id]).convert("RGB")
        # visualize_bbox(image, entity["bbox"])
        # print(metas[0]['text'])



    def __len__(self):
        return self.steps_per_epoch
