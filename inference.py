import os
import json
import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff
import zipfile
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from pycocotools import mask as maskUtils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import batched_nms

# =====================================================
# 1. 模型架構定義 (保持原樣)
# =====================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_planes // ratio, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class CBAMResNetWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.body = backbone
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

    def forward(self, x):
        features = self.body(x)
        out = {}
        for k, v in features.items():
            if k == '0': out[k] = self.cbam1(v)
            elif k == '1': out[k] = self.cbam2(v)
            elif k == '2': out[k] = self.cbam3(v)
            elif k == '3': out[k] = self.cbam4(v)
            else: out[k] = v
        return out

def get_model(num_classes=5):
    backbone_with_fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=None,
        trainable_layers=4,
        extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelMaxPool(),
    )
    backbone_with_fpn.body = CBAMResNetWrapper(backbone_with_fpn.body)

    model = torchvision.models.detection.MaskRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        ),
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_test=1500,
        rpn_nms_thresh=0.7,
        box_detections_per_img=1000,
        box_nms_thresh=0.5,
        box_score_thresh=0.003, 
    )

    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)

    return model

def binary_mask_to_rle(binary_mask):
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

# =====================================================
# 2. 終極融合推論流程 (Ensemble Inference)
# =====================================================
if __name__ == '__main__':
    TEST_DIR = "/disk2/ccchen/DL_CV_class/HW/HW3/test_release"
    JSON_PATH = "/disk2/ccchen/DL_CV_class/HW/HW3/test_image_name_to_ids.json"
    OUTPUT_JSON = "test-results.json"
    OUTPUT_ZIP = "submission.zip"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 🎯 定義要融合的模型列表
    MODEL_PATHS = [
        "hw3_epoch_030.pth",
        "hw3_best_model.pth",
        "hw3_epoch_035.pth"
    ]
    
    models = []
    print("[*] 正在載入模型進行預測融合 (Ensemble)...")
    for path in MODEL_PATHS:
        print(f"    - 載入權重: {path}")
        m = get_model(num_classes=5)
        # 廢除 PyTorch 內建的縮放干預
        m.transform.resize = lambda images, targets: (images, targets)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        m.to(DEVICE)
        m.eval()
        models.append(m)

    with open(JSON_PATH, 'r') as f:
        test_images_info = json.load(f)

    results = []
    
    TARGET_SIZE = 800
    MASK_THRESH = {1: 0.3, 2: 0.5, 3: 0.2, 4: 0.2}

    print(f"\n[*] 開始終極推論 (3 個模型 x TTA 翻轉)，共 {len(test_images_info)} 張照片...")
    
    for idx, img_info in enumerate(test_images_info):
        file_name = img_info['file_name']
        img_id = img_info['id']
        
        img = tiff.imread(os.path.join(TEST_DIR, file_name))[:, :, :3]
        orig_h, orig_w = img.shape[:2]
        img_t = torch.as_tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # 尺寸同步邏輯
        max_dim = max(orig_h, orig_w)
        scale = 1.0
        
        if max_dim > TARGET_SIZE:
            scale = TARGET_SIZE / float(max_dim)
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            img_t = TF.resize(img_t, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        else:
            new_h, new_w = orig_h, orig_w

        input_tensor = img_t.unsqueeze(0).to(DEVICE)
        
        # TTA 操作：原圖、水平翻轉
        patches = [
            ("orig", input_tensor),
            ("hflip", TF.hflip(input_tensor))
        ]
        
        all_boxes, all_scores, all_labels, all_masks = [], [], [], []
        
        # 🎯 雙重迴圈：讓每個模型都對每種 TTA patch 進行預測
        for m in models:
            for aug_type, p_img in patches:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    preds = m(p_img)[0]
                    
                if len(preds['scores']) == 0: continue
                    
                boxes = preds['boxes'].cpu()
                scores = preds['scores'].cpu()
                pred_labels = preds['labels'].cpu()
                raw_masks = preds['masks'].cpu() # shape: [N, 1, new_h, new_w]
                
                # TTA 還原：翻轉預測的座標和 Mask
                if aug_type == "hflip":
                    boxes_flipped = boxes.clone()
                    boxes_flipped[:, 0] = new_w - boxes[:, 2] # x1 = W - x2
                    boxes_flipped[:, 2] = new_w - boxes[:, 0] # x2 = W - x1
                    boxes = boxes_flipped
                    raw_masks = torch.flip(raw_masks, dims=[3]) 
                    
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(pred_labels)
                all_masks.append(raw_masks)

        # --- 合併所有模型與 TTA 結果，並執行最終 NMS ---
        if len(all_boxes) > 0:
            fb = torch.cat(all_boxes)
            fs = torch.cat(all_scores)
            fl = torch.cat(all_labels)
            fm = torch.cat(all_masks)
            
            # 🎯 關鍵步驟：使用 batched_nms 過濾來自不同模型、不同 TTA 的重複框
            # iou_threshold 設為 0.45 或 0.5 通常最穩定
            keep = batched_nms(fb, fs, fl, iou_threshold=0.45)
            
            final_boxes = fb[keep]
            final_scores = fs[keep].numpy()
            final_labels = fl[keep].numpy()
            final_masks_tensor = fm[keep]
            
            # 反向縮放
            if scale != 1.0:
                final_boxes /= scale
                final_masks_tensor = F.interpolate(final_masks_tensor, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            
            final_boxes = final_boxes.numpy()
            final_masks_raw = final_masks_tensor.squeeze(1).numpy()
            
            for i in range(len(final_scores)):
                score = final_scores[i]
                label = final_labels[i]
                thresh = MASK_THRESH.get(label, 0.5)
                
                bin_mask = (final_masks_raw[i] > thresh).astype(np.uint8)
                if bin_mask.sum() == 0: continue
                    
                x1, y1, x2, y2 = final_boxes[i]
                results.append({
                    'image_id': int(img_id),
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(score),
                    'category_id': int(label),
                    'segmentation': binary_mask_to_rle(bin_mask)
                })
                
        if (idx + 1) % 10 == 0:
            print(f"    進度: {idx + 1} / {len(test_images_info)}")

    print(f"[*] 寫入 {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f: json.dump(results, f)
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf: zipf.write(OUTPUT_JSON)
    print("✅ 終極模型融合推論完成！祝你順利突破分數！")