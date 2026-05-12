import os
import random
import numpy as np
import tifffile as tiff
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms.functional as TF

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import batched_nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# =====================================================
# 0. 環境與全域設定
# =====================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DIR = "./train"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================================
# 1. CBAM 注意力機制 — 插入 ResNet Stage 後（更有效的位置）
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

# CBAM 插入 ResNet body 的 layer2/3/4 後面（比插 FPN 輸出更有效）
class CBAMResNetWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.body = backbone
        # 預設回傳 4 層，PyTorch 會將 key 命名為 '0', '1', '2', '3'
        self.cbam1 = CBAM(256)   # 對應 layer1 (FPN P2)
        self.cbam2 = CBAM(512)   # 對應 layer2 (FPN P3)
        self.cbam3 = CBAM(1024)  # 對應 layer3 (FPN P4)
        self.cbam4 = CBAM(2048)  # 對應 layer4 (FPN P5)

    def forward(self, x):
        features = self.body(x)
        out = {}
        for k, v in features.items():
            if k == '0':
                out[k] = self.cbam1(v)
            elif k == '1':
                out[k] = self.cbam2(v)
            elif k == '2':
                out[k] = self.cbam3(v)
            elif k == '3':
                out[k] = self.cbam4(v)
            else:
                out[k] = v
        return out

# =====================================================
# 2. 資料集
# =====================================================
class MedicalCellDataset(Dataset):
    def __init__(self, sample_dirs, target_size=800, is_train=True):
        self.samples = sample_dirs
        self.target_size = target_size
        self.is_train = is_train

        # 加權抽樣：有 C3/C4 的圖給高權重
        self.sample_weights = []
        if self.is_train:
            for d in self.samples:
                self.sample_weights.append(self._compute_weight(d))

    def _compute_weight(self, sample_dir):
        for cid in [3, 4]:
            p = sample_dir / f"class{cid}.tif"
            if p.exists():
                m = tiff.imread(str(p))
                if np.any(m > 0):
                    return 6.0   # C3/C4 稀有類別，大幅提高抽樣率
        return 1.0

    def __len__(self):
        return len(self.samples)

    def _resize(self, img_t, masks_np, labels):
        orig_h, orig_w = img_t.shape[1], img_t.shape[2]
        max_dim = max(orig_h, orig_w)

        # 小圖不放大（避免 artifact），大圖才縮
        if max_dim <= self.target_size:
            new_h, new_w = orig_h, orig_w
        else:
            scale = self.target_size / float(max_dim)
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)

        img_t = TF.resize(img_t, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)

        new_masks, valid_labels = [], []
        for m, l in zip(masks_np, labels):
            m_t = torch.as_tensor(m, dtype=torch.uint8).unsqueeze(0)
            m_r = TF.resize(m_t, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
            if m_r.any():
                new_masks.append(m_r.numpy())
                valid_labels.append(l)

        return img_t, new_masks, valid_labels, new_h, new_w

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        img = tiff.imread(str(sample_dir / "image.tif"))[:, :, :3]

        # 讀取 mask
        masks_list, labels_list = [], []
        for class_id in range(1, 5):
            m_path = sample_dir / f"class{class_id}.tif"
            if not m_path.exists():
                continue
            mask = tiff.imread(str(m_path))
            for uid in np.unique(mask)[np.unique(mask) != 0]:
                inst = (mask == uid)
                if inst.sum() < 10:
                    continue
                masks_list.append(inst)
                labels_list.append(class_id)

        img_t = torch.as_tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # resize（小圖保原尺寸）
        img_t, masks_list, labels_list, new_h, new_w = self._resize(img_t, masks_list, labels_list)

        # ---- 資料增強 ----
        if self.is_train and len(masks_list) > 0:
            # 水平翻轉
            if random.random() > 0.5:
                img_t = TF.hflip(img_t)
                masks_list = [np.fliplr(m).copy() for m in masks_list]
            # 垂直翻轉
            if random.random() > 0.5:
                img_t = TF.vflip(img_t)
                masks_list = [np.flipud(m).copy() for m in masks_list]
            # 旋轉 90/180/270°（細胞無方向性，免費翻倍資料量）
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                img_t = torch.rot90(img_t, k, dims=[1, 2])
                masks_list = [np.rot90(m, k).copy() for m in masks_list]
            # 顏色抖動
            if random.random() > 0.3:
                img_t = TF.adjust_brightness(img_t, random.uniform(0.75, 1.25))
                img_t = TF.adjust_contrast(img_t, random.uniform(0.75, 1.25))
                img_t = TF.adjust_saturation(img_t, random.uniform(0.75, 1.25))
            # 高斯模糊（模擬對焦不准）
            if random.random() > 0.7:
                img_t = TF.gaussian_blur(img_t, kernel_size=5, sigma=random.uniform(0.1, 1.5))

        # ---- 計算 bbox ----
        boxes, final_masks, final_labels = [], [], []
        for m, l in zip(masks_list, labels_list):
            pos = np.where(m)
            if len(pos[0]) == 0:
                continue
            xmin, xmax = float(np.min(pos[1])), float(np.max(pos[1]))
            ymin, ymax = float(np.min(pos[0])), float(np.max(pos[0]))
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                final_masks.append(m)
                final_labels.append(l)

        if len(boxes) > 0:
            target = {
                "boxes":  torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(final_labels, dtype=torch.int64),
                "masks":  torch.as_tensor(np.array(final_masks), dtype=torch.uint8),
            }
        else:
            target = {
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks":  torch.zeros((0, new_h, new_w), dtype=torch.uint8),
            }

        return img_t, target


# =====================================================
# 3. 模型：MaskRCNN v2 + CBAM in ResNet stages
# =====================================================
def get_model(num_classes=5):
    # 先載入有 FPN 的 ResNet50 backbone
    backbone_with_fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=torchvision.models.ResNet50_Weights.DEFAULT,
        trainable_layers=4,          # 開放 layer1~4，讓更多層 fine-tune
        # 🚨 已經刪除 returned_layers=[2, 3, 4]，讓它恢復預設的 [1, 2, 3, 4] 保住微小細胞！
        extra_blocks=torchvision.ops.feature_pyramid_network.LastLevelMaxPool(),
    )

    # 把 ResNet body 換成帶 CBAM 的 wrapper
    backbone_with_fpn.body = CBAMResNetWrapper(backbone_with_fpn.body)

    model = torchvision.models.detection.MaskRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes,
        # RPN 設定
        rpn_anchor_generator=torchvision.models.detection.anchor_utils.AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),   # 對應小細胞（最小 35px 圖）
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        ),
        rpn_pre_nms_top_n_train=3000,
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1500,
        rpn_nms_thresh=0.7,
        # ROI 設定
        box_detections_per_img=500,
        box_nms_thresh=0.5,
        box_score_thresh=0.005,      # 訓練時低門檻讓 mAP 抓得到
        # 圖片大小：縮到 640 節省顯存
    )

    # 替換 head
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)

    return model


# =====================================================
# 4. 訓練主程式
# =====================================================
if __name__ == '__main__':

    all_dirs = sorted([p for p in Path(TRAIN_DIR).iterdir() if p.is_dir()])
    # 固定 seed 再 shuffle，確保每次切一樣
    rng = random.Random(SEED)
    rng.shuffle(all_dirs)
    split_idx = int(0.85 * len(all_dirs))

    TARGET_SIZE = 800

    train_ds = MedicalCellDataset(all_dirs[:split_idx], target_size=TARGET_SIZE, is_train=True)
    val_ds   = MedicalCellDataset(all_dirs[split_idx:], target_size=TARGET_SIZE, is_train=False)

    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=1, sampler=sampler,
        num_workers=4, collate_fn=lambda x: tuple(zip(*x)), pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=lambda x: tuple(zip(*x)),
    )

    model = get_model(num_classes=5).to(DEVICE)
    model.transform.resize = lambda images, targets: (images, targets)

    # 分組 lr：backbone 用小 lr，head 用大 lr
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5},
        {'params': head_params,     'lr': 1e-4},
    ], weight_decay=1e-3)

    MAX_EPOCHS  = 100
    ACCUM_STEPS = 8    # 等效 batch=8，顯存友好

    # Warmup 5 epoch + CosineAnnealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS - 5, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
    )

    scaler = torch.amp.GradScaler('cuda')

    best_ap50       = 0.0
    patience        = 15
    epochs_no_improve = 0

    print(f"🚀 開始訓練！(CBAM-in-ResNet + rot90 aug + batched_nms + warmup)")
    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | target_size: {TARGET_SIZE}")

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        train_loss_accum = 0.0
        num_batches = len(train_loader)

        for i, (imgs, targs) in enumerate(train_loader):
            imgs  = [img.to(DEVICE) for img in imgs]
            targs = [{k: v.to(DEVICE) for k, v in t.items()} for t in targs]

            with torch.amp.autocast('cuda'):
                loss_dict = model(imgs, targs)
                losses = sum(loss_dict.values()) / ACCUM_STEPS

            scaler.scale(losses).backward()
            train_loss_accum += losses.item() * ACCUM_STEPS

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # flush 殘留梯度（batch 數不整除 accum_steps 時）
        if num_batches % ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = train_loss_accum / num_batches

        # ---- Validation（滑動視窗） ----
        model.eval()
        metric = MeanAveragePrecision(iou_type="segm", class_metrics=True)

        PATCH_SIZE = 800
        STRIDE     = 640   # overlap=160，避免邊緣細胞被切破

        # 各類別 mask 門檻（稀有類別放寬）
        MASK_THRESH = {1: 0.5, 2: 0.5, 3: 0.35, 4: 0.35}

        with torch.no_grad():
            for imgs, targs in val_loader:
                img_t = imgs[0]
                _, H, W = img_t.shape

                all_boxes, all_scores, all_labels, all_masks = [], [], [], []

                for y in range(0, H, STRIDE):
                    for x in range(0, W, STRIDE):
                        y1 = min(y, max(0, H - PATCH_SIZE))
                        x1 = min(x, max(0, W - PATCH_SIZE))
                        y2 = min(y1 + PATCH_SIZE, H)
                        x2 = min(x1 + PATCH_SIZE, W)

                        patch = img_t[:, y1:y2, x1:x2].unsqueeze(0).to(DEVICE)

                        with torch.amp.autocast('cuda'):
                            preds = model(patch)[0]

                        if len(preds['scores']) == 0:
                            continue

                        boxes = preds['boxes'].cpu().clone()
                        boxes[:, [0, 2]] += x1
                        boxes[:, [1, 3]] += y1

                        # 各類別用不同 mask 門檻
                        pred_labels = preds['labels'].cpu()
                        raw_masks   = preds['masks'].cpu()
                        full_masks  = torch.zeros((len(raw_masks), H, W), dtype=torch.uint8)
                        for mi in range(len(raw_masks)):
                            lb = pred_labels[mi].item()
                            thresh = MASK_THRESH.get(lb, 0.5)
                            pm = (raw_masks[mi, 0] > thresh).to(torch.uint8)
                            full_masks[mi, y1:y2, x1:x2] = pm

                        all_boxes.append(boxes)
                        all_scores.append(preds['scores'].cpu())
                        all_labels.append(pred_labels)
                        all_masks.append(full_masks)

                if len(all_boxes) > 0:
                    fb = torch.cat(all_boxes)
                    fs = torch.cat(all_scores)
                    fl = torch.cat(all_labels)
                    fm = torch.cat(all_masks)

                    # ✅ batched_nms：各類別獨立做，不跨類互壓
                    keep = batched_nms(fb, fs, fl, iou_threshold=0.5)

                    cpu_preds = [{"boxes": fb[keep], "scores": fs[keep],
                                  "labels": fl[keep], "masks": fm[keep]}]
                else:
                    cpu_preds = [{"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)),
                                  "labels": torch.zeros((0,), dtype=torch.int64),
                                  "masks": torch.zeros((0, H, W), dtype=torch.uint8)}]

                cpu_targs = [{"boxes":  targs[0]["boxes"].cpu(),
                              "labels": targs[0]["labels"].cpu(),
                              "masks":  targs[0]["masks"].cpu()}]
                metric.update(cpu_preds, cpu_targs)

        val_results   = metric.compute()
        current_ap50  = val_results['map_50'].item()
        per_class_ap50 = val_results.get('map_50_per_class', val_results['map_per_class'])

        lr_now = optimizer.param_groups[1]['lr']
        print(f"\n🎯 Epoch {epoch+1:03d}/{MAX_EPOCHS} | Loss: {avg_train_loss:.4f} | "
              f"Val AP50: {current_ap50:.4f} | LR: {lr_now:.2e}")
        print(f"📊 各類別 AP50: {per_class_ap50.tolist() if torch.is_tensor(per_class_ap50) else per_class_ap50}")

        # ---- 儲存 & Early Stop ----
        if current_ap50 > best_ap50:
            best_ap50 = current_ap50
            epochs_no_improve = 0
            torch.save(model.state_dict(), "hw3_best_model.pth")
            print(f"🌟 新高分！儲存 Best Model (AP50: {best_ap50:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️  未突破，連續 {epochs_no_improve}/{patience} 次")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"hw3_epoch_{epoch+1:03d}.pth")

        torch.save(model.state_dict(), "hw3_latest_model.pth")

        scheduler.step()
        metric.reset()
        torch.cuda.empty_cache()

        if epochs_no_improve >= patience:
            print(f"\n🛑 Early Stopping！最佳 Val AP50: {best_ap50:.4f}")
            break