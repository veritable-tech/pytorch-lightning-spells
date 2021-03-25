"""SnapMix Utility Functions

Reference: `Shaoli-Huang/SnapMix <https://github.com/Shaoli-Huang/SnapMix/>`_
"""
import torch
import torch.nn.functional as F


def get_spm(input_tensor, target, model, image_size, half: bool = False):
    bs = input_tensor.size(0)
    with torch.no_grad():
        fc = model.get_fc()
        if half:
            input_tensor = input_tensor.half()
        fms = model.extract_features(input_tensor.to(fc.weight.device))
        weight, bias = fc.weight, fc.bias
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)

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

    return target_activation_map
