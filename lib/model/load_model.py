import os
import torch
import torch.nn as nn

from lib.utils.learning import load_backbone
from lib.model.DHDSTformer import DHDSTformer_total, DHDSTformer_total2, DHDSTformer_total3, DHDSTformer_total4, DHDSTformer_total5, DHDSTformer_total6, DHDSTformer_total7, DHDSTformer_total8, \
    DHDSTformer_limb, DHDSTformer_limb2, DHDSTformer_limb3, DHDSTformer_limb4, DHDSTformer_limb5, \
    DHDSTformer_right_arm, DHDSTformer_right_arm2, DHDSTformer_right_arm3, \
    DHDSTformer_torso, DHDSTformer_torso2, \
    DHDSTformer_torso_limb, \
    DHDSTformer_onevec

def load_model(opts, args):
    if opts.pretrained_backbone != '':
        print('Checkpoint for backbone', opts.pretrained_backbone)
        chk_backbone_filename = os.path.join(opts.pretrained_backbone, opts.selection)
    else:
        chk_backbone_filename = ''

    print(args.model)
    if 'DHDSTformer_total' == args.model: model_pos = DHDSTformer_total(chk_filename=chk_backbone_filename, args=args)
    elif 'DHDSTformer_total2' == args.model: model_pos = DHDSTformer_total2(chk_filename=chk_backbone_filename, args=args)
    elif 'DHDSTformer_total3' == args.model: model_pos = DHDSTformer_total3(chk_filename=chk_backbone_filename, args=args)
    elif 'DHDSTformer_total4' == args.model: model_pos = DHDSTformer_total4(args=args)
    elif 'DHDSTformer_total5' == args.model: model_pos = DHDSTformer_total5(args=args)
    elif 'DHDSTformer_total6' == args.model: model_pos = DHDSTformer_total6(chk_filename=chk_backbone_filename, args=args)
    elif 'DHDSTformer_total7' == args.model: model_pos = DHDSTformer_total7(chk_filename=chk_backbone_filename, args=args)
    elif 'DHDSTformer_total8' == args.model: model_pos = DHDSTformer_total8(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_torso': model_pos = DHDSTformer_torso(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_torso2': model_pos = DHDSTformer_torso2(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_limb': model_pos = DHDSTformer_limb(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_limb2': model_pos = DHDSTformer_limb2(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_limb3': model_pos = DHDSTformer_limb3(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_limb4': model_pos = DHDSTformer_limb4(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_limb5': model_pos = DHDSTformer_limb5(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_torso_limb': model_pos = DHDSTformer_torso_limb(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDST_onevec': model_pos = DHDSTformer_onevec(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_right_arm': model_pos = DHDSTformer_right_arm(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_right_arm2': model_pos = DHDSTformer_right_arm2(chk_filename=chk_backbone_filename, args=args)
    elif args.model == 'DHDSTformer_right_arm3': model_pos = DHDSTformer_right_arm3(chk_filename=chk_backbone_filename, args=args)
    else: 
        model_pos = load_backbone(args)
        
    if torch.cuda.is_available():
        model_pos = nn.DataParallel(model_pos)
        model_pos = model_pos.cuda()
    
    checkpoint = None
    if args.finetune:
        if opts.resume:
            chk_filename = opts.resume
            print('Loading checkpoint from resume', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_pos.load_state_dict(checkpoint['model_pos'], strict=True)
        elif opts.pretrained:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading checkpoint from pretrained', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_pos.load_state_dict(checkpoint['model_pos'], strict=True)    
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume:
            chk_filename = opts.resume
            print('Loading checkpoint from resume', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_pos.load_state_dict(checkpoint['model_pos'], strict=True)
    if opts.evaluate:
        chk_filename = os.path.join(opts.checkpoint, opts.evaluate)
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_pos.load_state_dict(checkpoint['model_pos'], strict=True)
    
    return model_pos, chk_filename, checkpoint