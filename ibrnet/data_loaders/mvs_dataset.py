# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os.path import join
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json
from glob import glob 
from tqdm import tqdm
import pandas as pd
import pickle

sys.path.append('../')

if __name__ == '__main__':
    from data_utils import rectify_inplane_rotation, get_nearest_pose_ids
else:
    from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


def read_cameras(pose_file):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, 'r') as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta['camera_angle_x'])
    rgb_files = []
    c2w_mats = []

    img = imageio.imread(os.path.join(basedir, meta['frames'][0]['file_path'] + '.png'))
    H, W = img.shape[:2]
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    intrinsics = get_intrinsics_from_hwf(H, W, focal)

    for i, frame in enumerate(meta['frames']):
        rgb_file = os.path.join(basedir, meta['frames'][i]['file_path'][2:] + '.png')
        rgb_files.append(rgb_file)
        c2w = np.array(frame['transform_matrix'])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics]*len(meta['frames'])), c2w_mats

def intrinsic_matrix(fx, fy, cx, cy):
    """ 3x3 Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.],
    ])

# https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/main/internal/datasets.py 
def load_blender_posedata(data_dir, downscale=4):
    """Load poses from `transforms.json` file, as used in Blender/NGP datasets."""
    pose_file = os.path.join(data_dir, 'transforms.json')

    with open(pose_file, 'r') as fp:
        meta = json.load(fp)

    names = []
    # poses = []
    c2w_mats = []

    for _, frame in enumerate(meta['frames']):
        filepath = os.path.join(data_dir, frame['file_path'])
        if os.path.exists(filepath):
            names.append(frame['file_path'].split('/')[-1])
            # poses.append(np.array(frame['transform_matrix'], dtype=np.float32))
            c2w = np.array(frame['transform_matrix'], dtype=np.float32)
            # w2c_blender = np.linalg.inv(c2w)
            # w2c_opencv = w2c_blender
            # w2c_opencv[1:3] *= -1
            # c2w_opencv = np.linalg.inv(w2c_opencv)
            c2w_mats.append(c2w)

    c2w_mats = np.array(c2w_mats)

    w = meta['w']
    h = meta['h']
    cx = meta['cx'] if 'cx' in meta else w / 2.
    cy = meta['cy'] if 'cy' in meta else h / 2.
    if 'fl_x' in meta:
        fx = meta['fl_x']
    else:
        fx = 0.5 * w / np.tan(0.5 * float(meta['camera_angle_x']))
    if 'fl_y' in meta:
        fy = meta['fl_y']
    else:
        fy = 0.5 * h / np.tan(0.5 * float(meta['camera_angle_y']))

    intrinsic = intrinsic_matrix(fx, fy, cx, cy)
    cam2pix = intrinsic
    pix2cam = np.linalg.inv(cam2pix)
    pix2cam = pix2cam @ np.diag([downscale, downscale, 1])
    intrinsic = np.linalg.inv(pix2cam)

    # change intrinsic to 4x4
    intrinsic = np.concatenate([intrinsic, np.zeros((3, 1))], axis=1)
    intrinsic = np.concatenate([intrinsic, np.zeros((1, 4))], axis=0)
    intrinsic[-1, -1] = 1.0

    n = len(names)
    intrinsics = [intrinsic] * n
    files = []
    # we use downfactor=4 for all datasets
    downfactor = 4
    for name in names:
        file_name = join(data_dir, f'images_{downfactor}', name)
        files.append(file_name)

    return files, intrinsics, c2w_mats


def load_blender_posedata_cache(data_folder, hash):
    cache_file = join(data_folder, 'cache', f'{hash}.bin')
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
        rgb_files = data['rgb_files']
        intrinsics = data['intrinsics']
        poses = data['poses']

    rgb_path = join(data_folder, 'compress', f'{hash}.npz')
    all_rgb = np.load(rgb_path)['data']

    return rgb_files, intrinsics, poses, all_rgb


def get_intrinsics_from_hwf(h, w, focal):
    return np.array([[focal, 0, 1.0*w/2, 0],
                     [0, focal, 1.0*h/2, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


class MVS_Dataset(Dataset):
    def __init__(self, args, mode,
                 scenes=(), **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/MVS/')
        self.rectify_inplane_rotation = args.rectify_inplane_rotation

        if mode == 'validation':
            mode = 'val'

        assert mode in ['train', 'val', 'test']
        self.mode = mode  # train / test / val
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        exp_setting = ['MVS_270', 'MVS_1K', 'MVS_2K']
        if 'MVS' in args.exp_name:
            assert args.exp_name in exp_setting, f'{args.exp_name} dataset not found'

        N = 270
        if args.exp_name == 'MVS_270':
            N = 270
        elif args.exp_name == 'MVS_1K':
            N = 1000
        elif args.exp_name == 'MVS_2K':
            N = 2000

        # import pdb; pdb.set_trace()
        scene_names = pd.read_csv(os.path.join(self.folder_path, 'meta.csv'))['hash'].tolist()[:N]
        cache = join(self.folder_path, 'cache', f'{args.exp_name}.bin' )
        os.makedirs(os.path.dirname(cache), exist_ok=True)

        print("loading {} for {}. Found {}".format(args.exp_name, mode, len(scene_names)))

        self.scene_maps = {}
        self.id_scene = {}
        self.id_index = {}

        if os.path.exists(cache):
            data = pickle.load(open(cache, 'rb'))
            self.scene_maps = data['scene_maps']
            self.id_scene = data['id_scene']
            self.id_index = data['id_index']

        else:
            for scene in tqdm(scene_names, desc='Init loading MVS scenes'):
                rgb_files, intrinsics, poses  = load_blender_posedata(join(self.folder_path, scene, 'data/outputs'))
                self.scene_maps[scene] = {
                    'rgb_files': rgb_files,
                    'intrinsics': intrinsics,
                    'poses': poses,
                }

                id = len(self.id_scene.keys())
                for i in range(len(rgb_files)):
                    self.id_scene[id + i] = scene
                    self.id_index[id + i] = i

            pickle.dump({'scene_maps': self.scene_maps,
                        'id_scene': self.id_scene,
                        'id_index': self.id_index}, open(cache, 'wb'))

            if self.mode != 'train':
                self.id_scene = {}
                for scene in tqdm(scene_names, desc='Loading MVS eval scenes'):
                    self.scene_maps[scene]['rgb_files'] = self.scene_maps[scene]['rgb_files'][::self.testskip]
                    self.scene_maps[scene]['intrinsics'] = self.scene_maps[scene]['intrinsics'][::self.testskip]
                    self.scene_maps[scene]['poses'] = self.scene_maps[scene]['poses'][::self.testskip]

                    id = len(self.id_scene)
                    for i in range(len(self.scene_maps[scene]['rgb_files'])):
                        self.id_scene[id + i] = scene
                        self.id_index[id + i] = i

        self.dbg = args.DBG
        self.dbg_nearest_pose_ids = None
        print('DBG: ', self.dbg)


    def __len__(self):
        # return 10
        # return len(self.render_rgb_files)
        if self.dbg:
            return 1
        else:
            return len(self.id_scene)


    def read_rgb(self, file):
        rgb = imageio.imread(file).astype(np.float32) / 255. 
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        return rgb


    def __getitem__(self, idx):
        scene = self.id_scene[idx]
        cur_id = self.id_index[idx]

        train_rgb_files = self.scene_maps[scene]['rgb_files']
        train_intrinsics = self.scene_maps[scene]['intrinsics']
        train_poses = self.scene_maps[scene]['poses']

        rgb_file = train_rgb_files[cur_id]
        render_pose = train_poses[cur_id]
        render_intrinsics = train_intrinsics[cur_id]

        if self.mode == 'train':
            id_render = cur_id
            assert id_render < len(train_poses), f'id_render out of bound. {id_render} vs {len(train_poses)} {rgb_file}'

            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
            if self.dbg:
                subsample_factor = 1

        else:
            id_render = -1
            subsample_factor = 1

        rgb = self.read_rgb(rgb_file)

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), render_intrinsics.flatten(), render_pose.flatten())).astype(np.float32)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                int(self.num_source_views*subsample_factor),
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.999, 0.001]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        if self.dbg:
            if self.dbg_nearest_pose_ids is None:
                self.dbg_nearest_pose_ids = nearest_pose_ids
            else:
                nearest_pose_ids = self.dbg_nearest_pose_ids

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = self.read_rgb(train_rgb_files[id])

            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(), train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)


        near_depth = 0.2
        far_depth = 100.0

        if self.dbg:
            near_depth = 0.2
            far_depth = 20.0
            

        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }


if __name__ == '__main__':
    data_dir = 'MVSDataset/training/'
    hash = '0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3'

    # mvs dataset
    train_rgb_files_, train_intrinsics_, train_poses_ = load_blender_posedata(join(data_dir, hash, 'data/outputs'))

    # nerf dataset
    train_rgb_files, train_intrinsics, train_poses = read_cameras('data/nerf_synthetic/lego/transforms_train.json')

    import pdb; pdb.set_trace()