"""SnapMix Utility Functions

Reference: https://github.com/Shaoli-Huang/SnapMix/
"""
import torch
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_spm(input_tensor, target, model, image_size, half: bool = False):
    bs = input_tensor.size(0)
    with torch.no_grad():
        fc = model.get_fc()
        if half:
            input_tensor = input_tensor.half()
        fms = model.extract_features(input_tensor.to(fc.weight.device))
        weight, bias = fc.weight, fc.bias
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)
        # pooled = F.adaptive_avg_pool2d(fms, (1, 1)).squeeze()
        # cls_logit = F.softmax(fc.forward(pooled))
        # logit_list = []
        # for i in range(bs):
        #     logit_list.append(cls_logit[i, target[i]])
        # cls_logit = torch.stack(logit_list)

        activations = F.conv2d(fms, weight, bias=bias)

        tmp = []
        for i in range(bs):
            target_activation = activations[i, target[i]]
            tmp.append(target_activation)

        target_activation_map = torch.stack(tmp)
        target_activation_map = target_activation_map.unsqueeze(1)
        target_activation_map = F.interpolate(
            target_activation_map, image_size,
            mode='bilinear', align_corners=False)

        target_activation_map = target_activation_map.squeeze(1)

        for i in range(bs):
            target_activation_map[i] -= target_activation_map[i].min()
            target_activation_map[i] /= target_activation_map[i].sum()

    return target_activation_map  # , cls_logit


# def snapmix(input, target, conf, model=None):
#     r = np.random.rand(1)
#     lam_a = torch.ones(input.size(0))
#     lam_b = 1 - lam_a
#     target_b = target.clone()

#     if r < conf.prob:
#         wfmaps, _ = get_spm(input, target, conf, model)
#         bs = input.size(0)
#         lam = np.random.beta(conf.beta, conf.beta)
#         lam1 = np.random.beta(conf.beta, conf.beta)
#         rand_index = torch.randperm(bs).cuda()
#         wfmaps_b = wfmaps[rand_index, :, :]
#         target_b = target[rand_index]

#         same_label = target == target_b
#         bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
#         bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)

#         area = (bby2-bby1)*(bbx2-bbx1)
#         area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

#         if area1 > 0 and area > 0:
#             ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
#             ncont = F.interpolate(ncont, size=(
#                 bbx2-bbx1, bby2-bby1), mode='bilinear', align_corners=True)
#             input[:, :, bbx1:bbx2, bby1:bby2] = ncont
#             lam_a = 1 - wfmaps[:, bbx1:bbx2,
#                                bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
#             lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(
#                 2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
#             tmp = lam_a.clone()
#             lam_a[same_label] += lam_b[same_label]
#             lam_b[same_label] += tmp[same_label]
#             lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
#                        (input.size()[-1] * input.size()[-2]))
#             lam_a[torch.isnan(lam_a)] = lam
#             lam_b[torch.isnan(lam_b)] = 1-lam

#     return input, target, target_b, lam_a.cuda(), lam_b.cuda()
