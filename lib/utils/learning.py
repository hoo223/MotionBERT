import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from lib.model.DSTformer import DSTformer
from lib.model.CanonDSTformer import CanonDSTformer1, CanonDSTformer2
from lib.model.TCPFormer import MemoryInducedTransformer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print('load_weight', len(matched_layers))
    return model

def partial_train_layers(model, partial_list):
    """Train partial layers of a given model."""
    for name, p in model.named_parameters():
        p.requires_grad = False
        for trainable in partial_list:
            if trainable in name:
                p.requires_grad = True
                break
    return model

def load_backbone(args):
    if not(hasattr(args, "backbone")):
        args.backbone = 'DSTformer' # Default
    if args.backbone=='DSTformer':
        model_backbone = DSTformer(dim_in=args.dim_in, dim_out=args.dim_out, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
    elif args.backbone=='TCN':
        from lib.model.model_tcn import PoseTCN
        model_backbone = PoseTCN()
    elif args.backbone=='poseformer':
        from lib.model.model_poseformer import PoseTransformer 
        model_backbone = PoseTransformer(num_frame=args.maxlen, num_joints=args.num_joints, in_chans=3, embed_dim_ratio=32, depth=4,
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0, attn_mask=None) 
    elif args.backbone=='mixste':
        from lib.model.model_mixste import MixSTE2 
        model_backbone = MixSTE2(num_frame=args.maxlen, num_joints=args.num_joints, in_chans=3, embed_dim_ratio=512, depth=8,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)
    elif args.backbone=='stgcn':
        from lib.model.model_stgcn import Model as STGCN 
        model_backbone = STGCN()
    elif args.backbone=='CanonDSTformer1':
        model_backbone = CanonDSTformer1(dim_in=args.dim_in, dim_out=args.dim_out, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
    elif args.backbone=='CanonDSTformer2':
        model_backbone = CanonDSTformer2(dim_in=args.dim_in, dim_out=args.dim_out, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
    elif args.backbone=='TCPFormer':
        from lib.utils.activation import SymSum
        act_mapper = {
            "gelu": nn.GELU,
            'relu': nn.ReLU,
            'symsum': SymSum
        }
        model_backbone = MemoryInducedTransformer(n_layers=args.n_layers,
                               dim_in=args.dim_in,
                               dim_feat=args.dim_feat,
                               dim_rep=args.dim_rep,
                               dim_out=args.dim_out,
                               mlp_ratio=args.mlp_ratio,
                               act_layer=act_mapper[args.act_layer],
                               attn_drop=args.attn_drop,
                               drop=args.drop,
                               drop_path=args.drop_path,
                               use_layer_scale=args.use_layer_scale,
                               layer_scale_init_value=args.layer_scale_init_value,
                               use_adaptive_fusion=args.use_adaptive_fusion,
                               num_heads=args.num_heads,
                               qkv_bias=args.qkv_bias,
                               qkv_scale=args.qkv_scale,
                               hierarchical=args.hierarchical,
                               num_joints=args.num_joints,
                               use_temporal_similarity=args.use_temporal_similarity,
                               temporal_connection_len=args.temporal_connection_len,
                               use_tcn=args.use_tcn,
                               graph_only=args.graph_only,
                               neighbour_num=args.neighbour_num,
                               n_frames=args.n_frames)
    else:
        raise Exception("Undefined backbone type.")
    return model_backbone