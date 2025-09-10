
# import torch
# import torch.nn as nn
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from .multitask_network import UNet_MultiTask
# from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

# class nnUNetTrainer_MultiTask(nnUNetTrainer):
#     def initialize_network(self):
#         """Initialize multi-task UNet"""
#         self.network = UNet_MultiTask(
#             in_channels=self.num_input_channels,
#             num_classes_seg=self.num_classes,
#             num_classes_clf=2  # Change based on your dataset
#         ).to(self.device)

#     def compute_loss(self, x, y):
#         """Compute combined loss for segmentation + classification"""
#         # Expect y = (seg_target, clf_target)
#         seg_target, clf_target = y
#         seg_pred, clf_pred = self.network(x)

#         seg_loss = DC_and_CE_loss(seg_pred, seg_target)
#         clf_loss = nn.CrossEntropyLoss()(clf_pred, clf_target)

#         total_loss = seg_loss + clf_loss
#         return total_loss















# import torch
# import torch.nn as nn
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from .multitask_network import UNet_MultiTask
# from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

# class nnUNetTrainer_MultiTask(nnUNetTrainer):
#     def initialize_network(self):
#         """Initialize multi-task UNet"""
#         self.network = UNet_MultiTask(
#             in_channels=self.num_input_channels,
#             num_classes_seg=self.num_classes,
#             num_classes_clf=2
#         ).to(self.device)

#     def compute_loss(self, x, y):
#         seg_target, clf_target = y
#         seg_pred, clf_pred = self.network(x)
#         seg_loss = DC_and_CE_loss(seg_pred, seg_target)
#         clf_loss = nn.CrossEntropyLoss()(clf_pred, clf_target)
#         return seg_loss + clf_loss

#     # Ensure inference returns only segmentation for nnUNet infrastructure
#     def predict(self, x):
#         seg_pred, clf_pred = self.network(x)
#         return seg_pred  # only segmentation for predictor









import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from multitask_network import UNetResEncMultiTask


class nnUNetTrainer_MultiTask(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, **kwargs):
        super().__init__(plans, configuration, fold, **kwargs)
        self.num_classes_clf = 2  # 🔧 adjust for your dataset

    def build_network(self):
        # Replace standard UNet with multi-task ResEnc M
        self.network = UNetResEncMultiTask(
            self.plans,
            self.num_input_channels,
            self.num_classes,
            self.num_classes_clf
        ).to(self.device)

    def compute_loss(self, data, target, output):
        seg_logits, clf_logits = output

        # Segmentation target: assume last channel is segmentation mask
        seg_target = target['seg']  # (B, H, W, D)
        seg_target = seg_target.long()

        # Classification target
        clf_target = target['clf'].long()

        # Segmentation loss (Dice + CE)
        ce_loss = F.cross_entropy(seg_logits, seg_target)
        dice_loss = self.loss(seg_logits, seg_target)  # nnUNet’s built-in dice loss
        seg_loss = ce_loss + dice_loss

        # Classification loss
        clf_loss = F.cross_entropy(clf_logits, clf_target)

        # Total multitask loss
        total_loss = seg_loss + clf_loss
        return total_loss

    def predict(self, x):
        # Only return segmentation to nnUNet pipeline
        seg_out, _ = self.network(x)
        return seg_out
