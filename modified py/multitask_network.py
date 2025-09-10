
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Simple 3D UNet backbone for demonstration
# class UNetBackbone(nn.Module):
#     def __init__(self, in_channels=1, base_channels=32):
#         super().__init__()
#         self.enc1 = nn.Sequential(nn.Conv3d(in_channels, base_channels, 3, padding=1),
#                                   nn.ReLU(),
#                                   nn.Conv3d(base_channels, base_channels, 3, padding=1),
#                                   nn.ReLU())
#         self.pool = nn.MaxPool3d(2)
#         self.enc2 = nn.Sequential(nn.Conv3d(base_channels, base_channels*2, 3, padding=1),
#                                   nn.ReLU(),
#                                   nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
#                                   nn.ReLU())

#         self.center = nn.Sequential(nn.Conv3d(base_channels*2, base_channels*4, 3, padding=1),
#                                     nn.ReLU(),
#                                     nn.Conv3d(base_channels*4, base_channels*4, 3, padding=1),
#                                     nn.ReLU())

#         self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
#         self.dec2 = nn.Sequential(nn.Conv3d(base_channels*4, base_channels*2, 3, padding=1),
#                                   nn.ReLU(),
#                                   nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
#                                   nn.ReLU())

#         self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2)
#         self.dec1 = nn.Sequential(nn.Conv3d(base_channels*2, base_channels, 3, padding=1),
#                                   nn.ReLU(),
#                                   nn.Conv3d(base_channels, base_channels, 3, padding=1),
#                                   nn.ReLU())

#     def forward(self, x):
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         center = self.center(self.pool(enc2))

#         dec2 = self.dec2(torch.cat([self.up2(center), enc2], dim=1))
#         dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
#         return dec1


# # Multi-task UNet: segmentation + classification
# class UNet_MultiTask(nn.Module):
#     def __init__(self, in_channels=1, num_classes_seg=2, num_classes_clf=2):
#         super().__init__()
#         self.backbone = UNetBackbone(in_channels)
#         base_channels = 32

#         # Segmentation head
#         self.seg_head = nn.Conv3d(base_channels, num_classes_seg, kernel_size=1)

#         # Classification head (global average pooling -> linear)
#         self.clf_head = nn.Sequential(
#             nn.AdaptiveAvgPool3d(1),
#             nn.Flatten(),
#             nn.Linear(base_channels, num_classes_clf)
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         seg_out = self.seg_head(features)
#         clf_out = self.clf_head(features)
#         return seg_out, clf_out






















# import torch
# import torch.nn as nn

# # --- 3D UNet backbone ---
# class UNetBackbone(nn.Module):
#     def __init__(self, in_channels=1, base_channels=32):
#         super().__init__()
#         self.enc1 = nn.Sequential(
#             nn.Conv3d(in_channels, base_channels, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(base_channels, base_channels, 3, padding=1),
#             nn.ReLU()
#         )
#         self.pool = nn.MaxPool3d(2)
#         self.enc2 = nn.Sequential(
#             nn.Conv3d(base_channels, base_channels*2, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
#             nn.ReLU()
#         )
#         self.center = nn.Sequential(
#             nn.Conv3d(base_channels*2, base_channels*4, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(base_channels*4, base_channels*4, 3, padding=1),
#             nn.ReLU()
#         )
#         self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
#         self.dec2 = nn.Sequential(
#             nn.Conv3d(base_channels*4, base_channels*2, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
#             nn.ReLU()
#         )
#         self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2)
#         self.dec1 = nn.Sequential(
#             nn.Conv3d(base_channels*2, base_channels, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(base_channels, base_channels, 3, padding=1),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         center = self.center(self.pool(enc2))
#         dec2 = self.dec2(torch.cat([self.up2(center), enc2], dim=1))
#         dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
#         return dec1

# # --- Multi-task UNet ---
# class UNet_MultiTask(nn.Module):
#     def __init__(self, in_channels=1, num_classes_seg=2, num_classes_clf=2):
#         super().__init__()
#         self.backbone = UNetBackbone(in_channels)
#         base_channels = 32

#         self.seg_head = nn.Conv3d(base_channels, num_classes_seg, kernel_size=1)
#         self.clf_head = nn.Sequential(
#             nn.AdaptiveAvgPool3d(1),
#             nn.Flatten(),
#             nn.Linear(base_channels, num_classes_clf)
#         )

#     def forward(self, x, return_seg_only=False):
#         features = self.backbone(x)
#         seg_out = self.seg_head(features)
#         clf_out = self.clf_head(features)
#         if return_seg_only:
#             return seg_out
#         return seg_out, clf_out















import torch
import torch.nn as nn
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


class UNetResEncMultiTask(nn.Module):
    def __init__(self, plans, num_input_channels, num_classes_seg, num_classes_clf):
        super().__init__()

        # Build segmentation backbone from plans (ResEnc M variant recommended)
        pm = PlansManager(plans)
        self.encoder_decoder = pm.build_network_architecture(
            num_input_channels=num_input_channels,
            num_classes=num_classes_seg,
            enable_deep_supervision=True
        )

        # Classification head: global pooled encoder features -> FC layer
        enc_channels = self.encoder_decoder.encoder.stages[-1].output_channels
        self.clf_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(enc_channels, num_classes_clf)
        )

        self.last_features = None

    def forward(self, x, return_seg_only=False):
        seg_logits = self.encoder_decoder(x)

        # Grab last encoder features
        self.last_features = self.encoder_decoder.encoder.stages[-1].output
        clf_logits = self.clf_head(self.last_features)

        if return_seg_only:
            return seg_logits
        return seg_logits, clf_logits
