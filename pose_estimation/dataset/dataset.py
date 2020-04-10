"""[summary]
"""


import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


class PoseDataset(data.Dataset):
    """PoseDataset class
    """
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        """Creates a PoseDataset object
        
        Arguments:
            data {torch.utils.data} -- An abstract class representing a Dataset
            mode {string} -- train, test or eval
            num {[type]} -- [description]
            add_noise {[type]} -- [description]
            root {string} -- path/to/root
            noise_trans {[type]} -- [description]
            refine {bool} -- 
        """
        self.objs = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.mode = mode
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine
        self.num = num
        self.add_noise = add_noise

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}

        item_count = 0
        for item in self.objs:
            if self.mode == 'train':
                input_file = open(f"{self.root}/data/{'%02d' % item}/train.txt")
            else:
                input_file = open(f"{self.root}/data/{'%02d' % item}/test.txt")
            
            while True:
                item_count += 1
                input_line = input_file.readline()
                
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                
                if not input_line:
                    break

                if input_line[-1:] == "\n":
                    input_line = input_line[:-1]

                self.list_rgb.append(f"{self.root}/data/{item:02d}/rgb/{input_line}.png")
                self.list_rgb.append(f"{self.root}/data/{item:02d}/depth/{input_line}.png")

                if self.mode == "eval":
                    self.list_label.append(f"{self.root}/segnet_results/{item:02d}_label/{input_line}_label.png")
                else:
                    self.list_label.append(f"{self.root}/data/{item:02d}/mask/{input_line}.png")
                
                self.list_obj.append(item)
                self.list_rank.append(int(input_line))

            meta_file = open(f"{self.root}/data/{item:02d}/gt.yml", 'r')
            self.meta[item] = yaml.load(meta_file)
            self.pt[item] = ply_vtx(f'{self.root}/models/obj_{item:02d}.ply')

            print(f'Object {item} buffer loaded')
        
        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index]
        rank = self.list_rank[index]

        if obj == 2:
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255,255,255])))[:,:,0]
        mask = mask_label * mask_depth
        
        if self.add_noise:
            img = self.trancolor(img)
        
        img = np.array(img)[:,:,3]
        img = np.transpose(img, (2,0,1))
        img_masked = img

        if self.mode == 'eval':
            rmin, rmax, cmin, cmax = get_bbox(mask_2_bbox(mask_label))
        else:
            rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])
        
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        # p_img = np.transpose(img_masked, (1, 2, 0))
        # scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3,3))
        target_t = np.array(meta['cam_t_m2c'])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)
        
        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        if self.add_noise:
            cloud = np.add(cloud, add_t)

        # fw = open(f'evaluation_result/{index}_cld.xyz', 'w')
        # for it in cloud:
        #    fw.write(f'{it[0]} {it[1]} {it[2]}\n')
        # fw.close()

        model_pts = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_pts))]
        dellist = random.sample(dellist, len(model_pts) - self.num_pt_mesh_small)
        model_pts = np.delete(model_pts, dellist, axis=0)

        # fw = open(f'evaluation_result/{index}_model_points.xyz', 'w')
        # for it in model_points:
        #    fw.write(f'{it[0]} {it[1]} {it[2]}\n')
        # fw.close()

        target = np.dot(model_pts, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t / 1000.0 + add_t)
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)
            out_t = target_t / 1000.0

        # fw = open(f'evaluation_result/{index}_tar.xyz', 'w')
        # for it in target:
        #    fw.write(f'{it[0]} {it[1]} {it[29]}\n')
        # fw.close()

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_pts.astype(np.float32)), \
               torch.LongTensor([self.objs.index(obj)])
        
        def __len__(self):
            return self.length

        def get_sym_list(self):
            return self.symmetry_obj_idx

        def get_num_pts_mesh(self):
            if self.refine:
                return self.num_pt_mesh_large
            else:
                return self.num_pt_mesh_small


def mask_2_bbox(mask):
    """Gets the bounding box of the mask
    
    Arguments:
        mask {np.array} -- mask image
    
    Returns:
        array -- x, y, w, h
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        _x, _y, _w, _h = cv2.boundingRect(contour)
        if (_w * _h) > (w * h):
            x, y, w, h = _x, _y, _w, _h
    
    return [x, y, w, h]

def get_bbox(bbox):
    """Ensures the bounding box is within the borders of the image.
    
    Arguments:
        bbox {array} -- bounding box
    
    Returns:
        int, int, int, int -- row minimum, row maximum, col minimum and col
        maximum of the bounding box.
    """
    bbx = [bbox[1], bbox[1]+bbox[3], bbox[0], bbox[0]+bbox[2]]

    if bbx[0] < 0:
        bbx[0] = 0
    
    if bbx[1] >= 480:
        bbx[1] = 479
    
    if bbx[2] < 0:
        bbx[2] = 0
    
    if bbx[3] < 640:
        bbx[3] = 639
    
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]

    r_b = rmax - rmin
    for i in range(len(border_list)):
        if r_b > border_list[i] and r_b < border_list[i+1]:
            r_b = border_list[i+1]
            break
    
    r_b = cmax - cmin
    for i in rage(len(border_list)):
        if c_b > border_list[i] and c_b < border_list[i+1]:
            c_b = border_list[i+1]
            break
    
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt

    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt

    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt

    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt

    return rmin, rmax, cmin, cmax

def ply_vtx(path):
    """Get the points from a ply file.
    
    Arguments:
        path {string} -- path to ply file
    
    Returns:
        np.array -- points of the ply file
    """
    f = open(path)
    assert f.readline().strip() == 'ply'
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != 'end_header':
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)