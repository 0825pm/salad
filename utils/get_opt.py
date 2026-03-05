import os
from os.path import join as pjoin
from argparse import Namespace
import re


POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def is_list(numStr):
    flag = False
    numStr = str(numStr).strip()
    if numStr.startswith("[") and numStr.endswith("]"):
        flag = True
    return flag


def get_opt(opt_path, device, **kwargs):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path, 'r') as f:
        for line in f:
            if line.strip() not in skip:
                key, value = line.strip('\n').split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                elif is_list(value):
                    opt_dict[key] = eval(value)
                else:
                    opt_dict[key] = str(value)

    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/humanml3d/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.pose_dim = 263
        opt.contact_joints = [7, 10, 8, 11]
        opt.fps = 20
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/kit-ml/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.pose_dim = 251
        opt.contact_joints = [19, 20, 14, 15]
        opt.fps = 12.5
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    ## ── PATCH: sign ──
    elif opt.dataset_name == 'sign':
        skeleton_mode = getattr(opt, 'skeleton_mode', 'sign10_vel')
        if skeleton_mode == 'sign10_vel':
            opt.joints_num = 10
            opt.pose_dim = 210
        elif skeleton_mode == 'sign10':
            opt.joints_num = 10
            opt.pose_dim = 120
        elif skeleton_mode == '7part':
            opt.joints_num = 7
            opt.pose_dim = 133
        opt.contact_joints = []
        opt.fps = 24
        opt.max_motion_length = getattr(opt, 'max_motion_length', 400)
        opt.max_motion_frame = opt.max_motion_length
        opt.max_motion_token = 55
    ## ── end PATCH ──
    else:
        raise KeyError('Dataset not recognized')

    if not hasattr(opt, 'unit_length'):
        opt.unit_length = 4
    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    opt_dict.update(kwargs)

    return opt