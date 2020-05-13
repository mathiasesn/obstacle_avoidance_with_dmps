"""Predict with DenseFusion
"""

import open3d as o3d
import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import numpy as np
import copy
import random
import cv2
import yaml
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from numpy import ma
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
from pose_estimation.dataset.linemod.dataset import ply_vtx, get_bbox, mask_2_bbox
from pose_estimation.dense_fusion.lib.network import PoseNet, PoseRefineNet
from pose_estimation.dense_fusion.lib.loss import Loss
from pose_estimation.dense_fusion.lib.loss_refiner import Loss_refine
from pose_estimation.dense_fusion.lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from pose_estimation.dense_fusion.lib.knn.__init__ import KNearestNeighbor


parser = argparse.ArgumentParser(description='Predict with DenseFusion')
parser.add_argument('--item', type=str, default='04', help='item to predict (default: 01 for ape)')
parser.add_argument('--data_root', type=str, default='pose_estimation/dataset/linemod/Linemod_preprocessed', help='path/to/dataset/root')
parser.add_argument('--posenet_model', type=str, default='pose_estimation/dense_fusion/trained_models/linemod/pose_model_9_0.01310166542980859.pth', help='path/to/posenet/model')
parser.add_argument('--refinenet_model', type=str, default='pose_estimation/dense_fusion/trained_models/linemod/pose_refine_model_29_0.006821325639856025.pth', help='path/to/refinenet/model')
parser.add_argument('--std', type=float, default=0.0, help='standard devitaion of the noise added to depth and image')
args = parser.parse_args()


class Visualize:
    def __init__(self, pcds, img_path, depth_path):
        self.pcds = pcds
        self.img_path = img_path
        self.depth_path = depth_path
        self.index = 0
        # self.get_rgbd_input()

    def get_rgbd_input(self):
        color_raw = o3d.io.read_image(self.img_path)
        depth_raw = o3d.io.read_image(self.depth_path)
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
        cam_mat = o3d.camera.PinholeCameraIntrinsic()
        cam_mat.set_intrinsics(640, 480, 572.41140, 573.57043, 325.26110, 242.04899)
        pcd_rgbd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, cam_mat)
        self.pcds.append(pcd_rgbd)

    def custom_draw_geometry_with_key_callback(self, win_name):
        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            return False

        def load_render_option(vis):
            vis.get_render_option().load_from_json("pose_estimation/dense_fusion/demonstration/renderoption.json")
            return False

        def capture_depth(vis):
            depth = vis.capture_depth_float_buffer()
            plt.imshow(np.asarray(depth))
            plt.show()
            return False

        def capture_image(vis):
            image = vis.capture_screen_float_buffer()
            plt.imsave(f'pose_estimation/dense_fusion/demonstration/data/{self.index:04d}.png', np.asarray(image), dpi=1)
            self.index += 1
            return False
        
        def rotate_view(vis):
            image = vis.capture_screen_float_buffer()
            plt.imsave(f'pose_estimation/dense_fusion/demonstration/data/{self.index:04d}.png', np.asarray(image), dpi=1)
            self.index += 1
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 5.0)
            return False

        key_to_callback = {}
        key_to_callback[ord("K")] = change_background_to_black
        key_to_callback[ord("R")] = load_render_option
        key_to_callback[ord(",")] = capture_depth
        key_to_callback[ord(".")] = capture_image
        key_to_callback[ord("S")] = rotate_view
        o3d.visualization.draw_geometries_with_key_callbacks(self.pcds, key_to_callback, window_name=win_name)
    
    def custom_draw_geometry_with_rotation(self, win_name):
        def rotate_view(vis):
            image = vis.capture_screen_float_buffer()
            plt.imsave(f'pose_estimation/dense_fusion/demonstration/data/{self.index:04d}.png', np.asarray(image), dpi=1)
            self.index += 1
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            return False

        o3d.visualization.draw_geometries_with_animation_callback(self.pcds, rotate_view)

    	
    def custom_draw_geometry(self, win_name):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=win_name)
        [vis.add_geometry(pcd) for pcd in self.pcds]
        vis.run()
        vis.destroy_window()

    def set_pcds(self, pcds):
        self.pcds = pcds  


class Dataset(Dataset):
    def __init__(self, root, item, num_points, sigma=0.0):
        self.sigma = sigma
        self.objs = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.num = num_points

        self.imgs = []
        self.depths = []
        self.labels = []
        self.objects = []
        self.ranks = []
        self.meta = {}
        self.pt = {}

        item_count = 0
        input_file = open(f'{root}/data/{item}/test.txt')

        while True:
            item_count += 1
            input_line = input_file.readline()

            if not input_line:
                break

            if input_line[-1:] == '\n':
                input_line = input_line[:-1]

            self.imgs.append(f'{root}/data/{item}/rgb/{input_line}.png')
            self.depths.append(f'{root}/data/{item}/depth/{input_line}.png')
            self.labels.append(f'{root}/segnet_results/{item}_label/{input_line}_label.png')
            self.objects.append(int(item))
            self.ranks.append(int(input_line))
        
        meta_file = open(f'{root}/data/{item}/gt.yml', 'r')
        self.meta[int(item)] = yaml.load(meta_file, Loader=yaml.FullLoader)
        self.pt[int(item)] = ply_vtx(f'{root}/models/obj_{item}.ply')

        self.length = len(self.imgs)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = num_points
        self.num_pt_mesh_small = num_points
        self.symmetry_obj_idx = [7, 8]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        ori_img = np.array(img)
        depth = np.array(Image.open(self.depths[index]))
        label = np.array(Image.open(self.labels[index]))
        obj = self.objects[index]
        rank = self.ranks[index]

        if obj == 2:
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        mask = mask_label * mask_depth

        img = np.array(img)[:, :, :3]

        # add noise to image
        rows, cols, chs = img.shape
        gauss = np.random.normal(0, self.sigma, (rows ,cols, chs)) * 255.0
        gauss = np.reshape(gauss.astype(img.dtype), (rows, cols, chs))
        img = img + gauss

        # # // show noisy image
        # cv2.imshow(f'Noisy img std {self.sigma}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # # //

        img = np.transpose(img, (2, 0, 1))
        masked_img = img

        masked_bbox = mask_2_bbox(mask_label)
        rmin, rmax, cmin, cmax = get_bbox(masked_bbox)
        masked_img = masked_img[:, rmin:rmax, cmin:cmax]

        # # // Uncomment to show cropped image
        # crop_img = np.transpose(masked_img, (1, 2, 0))
        # cv2.imshow('Cropped image', cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
        # # //

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3,3))
        target_t = np.array(meta['cam_t_m2c'])

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

        # add noise to cloud
        rows, cols = cloud.shape
        gauss = np.random.normal(0, self.sigma, (rows, cols)).astype(cloud.dtype)
        gauss = np.reshape(gauss, (rows, cols))
        cloud = cloud + gauss

        # # // show input cloud
        # pcd_input = o3d.geometry.PointCloud()
        # pcd_input.points = o3d.utility.Vector3dVector(cloud)
        # pcd_input.paint_uniform_color([0, 0, 1])
        # visualize(pcd_input, self.imgs[index], self.depths[index], 'Input point mask')
        # o3d.visualization.draw_geometries([pcd_input], window_name=f'Input cloud with sigma {self.sigma}')
        # # //

        model_pts = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_pts))]
        dellist = random.sample(dellist, len(model_pts) - self.num_pt_mesh_small)
        model_pts = np.delete(model_pts, dellist, axis=0)

        # # // show init pose
        # pcd_init = o3d.geometry.PointCloud()
        # pcd_init.points = o3d.utility.Vector3dVector(model_pts)
        # pcd_init.paint_uniform_color([1, 0, 0])
        # visualize(pcd_init, self.imgs[index], self.depths[index], 'Init')
        # o3d.visualization.draw_geometries([pcd_init], window_name='Init pose')
        # # //

        target = np.dot(model_pts, target_r.T)
        target = np.add(target, target_t / 1000.0)

        # # // show ground truth
        # pcd_target = o3d.geometry.PointCloud()
        # pcd_target.points = o3d.utility.Vector3dVector(target)
        # pcd_target.paint_uniform_color([1, 0, 0])
        # visualize(pcd_target, self.imgs[index], self.depths[index], 'Ground truth pose')
        # o3d.visualization.draw_geometries([pcd_target], window_name='Ground truth cloud')
        # # //

        out_t = target_t / 1000.0

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(masked_img.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_pts.astype(np.float32)), \
               torch.LongTensor([self.objs.index(obj)])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh_small

    def get_rgbd_path(self, index):
        return self.imgs[index], self.depths[index]

def main():
    num_objects = 13
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    num_points = 10000 #500
    iteration = 2
    bs = 1
    dataset_config_dir = 'pose_estimation/dataset/linemod/dataset_config'
    output_result_dir = ''
    knn = KNearestNeighbor(1)

    estimator = PoseNet(num_points, num_objects)
    estimator.cuda()
    estimator.load_state_dict(torch.load(args.posenet_model))
    estimator.eval()

    refiner = PoseRefineNet(num_points, num_objects)
    refiner.cuda()
    refiner.load_state_dict(torch.load(args.refinenet_model))
    refiner.eval()

    dataset = Dataset(args.data_root, args.item, num_points, sigma=args.std)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    sym_list = dataset.get_sym_list()
    num_points_mesh = dataset.get_num_points_mesh()
    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    diameter = []
    meta_file = open(f'{dataset_config_dir}/models_info.yml', 'r')
    meta = yaml.load(meta_file, Loader=yaml.FullLoader)
    for obj in objlist:
        obj_d = meta[obj]['diameter'] / 1000.0 * 0.1
        diameter.append(obj_d)

    success_count = [0 for i in range(num_objects)]
    num_count = [0 for i in range(num_objects)]
    times = []

    bar = tqdm(dataloader)
    for i, data in enumerate(bar, 0):
        points, choose, img, target, model_points, idx = data

        if len(points.size()) == 2:
            bar.set_description(f'No.{i} NOT Pass! Lost detection!')
            continue

        points = Variable(points).cuda()
        choose = Variable(choose).cuda()
        img = Variable(img).cuda()
        target = Variable(target).cuda()
        model_points = Variable(model_points).cuda()
        idx = Variable(idx).cuda()

        tic = time.time()

        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)

        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)

        pred_t = pred_t.view(bs*num_points, 1, 3)

        r = pred_r[0][which_max[0]]
        r = r.view(-1)
        r = r.cpu().data.numpy()
        
        t = (points.view(bs*num_points, 1, 3) + pred_t)
        t = t[which_max[0]]
        t = t.view(-1)
        t = t.cpu().data.numpy()
        
        pred = np.append(r, t)

        # # // show pose
        # pcd_img_path, pcd_depth_path = dataset.get_rgbd_path(i)
        # pcd_model_points = model_points[0].cpu().detach().numpy()
        # pcd_r = quaternion_matrix(r)
        # pcd_r = pcd_r[:3, :3]
        # pcd_pose = np.dot(pcd_model_points, pcd_r.T) + t
        # pcd_target = o3d.geometry.PointCloud()
        # pcd_target.points = o3d.utility.Vector3dVector(pcd_pose)
        # pcd_target.paint_uniform_color([0, 0, 1])

        # gt = target[0].cpu().detach().numpy()
        # pcd_gt = o3d.geometry.PointCloud()
        # pcd_gt.points = o3d.utility.Vector3dVector(gt)
        # pcd_gt.paint_uniform_color([1, 0, 0])

        # pcds = [pcd_target, pcd_gt]
        # vis = Visualize(pcds, pcd_img_path, pcd_depth_path)

        # bar.write(f'Pose from PoseNet\n Rotation\n{pcd_r}\n translation\n{t}')
        
        # error = np.mean(np.linalg.norm(pcd_pose - gt, axis=1))
        # bar.write(f'Add score --> {error}')

        # vis.custom_draw_geometry(f'Initial pose width ADD score {error:.4f}')
        # # //

        for j in range(0, iteration):
            T = t.astype(np.float32)
            T = torch.from_numpy(T)
            T = Variable(T).cuda()
            T = T.view(1, 3)
            T = T.repeat(num_points, 1).contiguous()
            T = T.view(1, num_points, 3)

            mat = quaternion_matrix(r)

            R = mat[:3, :3].astype(np.float32)
            R = torch.from_numpy(R)
            R = Variable(R).cuda()
            R = R.view(1, 3, 3)

            mat[0:3, 3] = t

            new_points = torch.bmm((points - T), R).contiguous()

            pred_r, pred_t = refiner(new_points, emb, idx)

            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, 1, 1)

            r2 = pred_r.view(-1).cpu().data.numpy()
            t2 = pred_t.view(-1).cpu().data.numpy()
            mat2 = quaternion_matrix(r2)
            mat2[0:3, 3] = t2

            final_mat = np.dot(mat, mat2)
            
            final_r = copy.deepcopy(final_mat)
            final_r[0:3, 3] = 0
            final_r = quaternion_from_matrix(final_r, True)

            final_t = np.array([
                final_mat[0][3],
                final_mat[1][3],
                final_mat[2][3]
            ])

            pred = np.append(final_r, final_t)
            
            r = final_r
            t = final_t

            # # // show pose
            # pcd_img_path, pcd_depth_path = dataset.get_rgbd_path(i)

            # pcd_model_points = model_points[0].cpu().detach().numpy()
            # pcd_r = quaternion_matrix(r)
            # pcd_r = pcd_r[:3, :3]
            # pcd_pose = np.dot(pcd_model_points, pcd_r.T) + t
            # bar.write(f'Pose from PoseRefineNet iteration {j}\n Rotation\n{pcd_r}\n translation\n{t}')

            # pcd_target = o3d.geometry.PointCloud()
            # pcd_target.points = o3d.utility.Vector3dVector(pcd_pose)
            # pcd_target.paint_uniform_color([0, 0, 1])

            # gt = target[0].cpu().detach().numpy()
            # pcd_gt = o3d.geometry.PointCloud()
            # pcd_gt.points = o3d.utility.Vector3dVector(gt)
            # pcd_gt.paint_uniform_color([1, 0, 0])

            # pcds = [pcd_target, pcd_gt]
            # vis.set_pcds(pcds)

            # error = np.mean(np.linalg.norm(pcd_pose - gt, axis=1))
            # bar.write(f'Add score --> {error}')

            # vis.custom_draw_geometry(f'Refinement {j+1} with ADD score {error:.4f}')
            # # //
        
        model_points = model_points[0].cpu().detach().numpy()
        r = quaternion_matrix(r)
        r = r[:3, :3]
        pose = np.dot(model_points, r.T) + t

        target = target[0].cpu().detach().numpy()

        # // show pose
        bar.write(f'Final pose\n Rotation\n{r}\n translation\n{t}')

        pcd_img_path, pcd_depth_path = dataset.get_rgbd_path(i)

        est_pcd = o3d.geometry.PointCloud()
        est_pcd.points = o3d.utility.Vector3dVector(pose)
        est_pcd.paint_uniform_color([0, 0, 1])


        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        target_pcd.paint_uniform_color([1, 0, 0])

        pcds = [est_pcd, target_pcd]
        # vis = Visualize(pcds, pcd_img_path, pcd_depth_path)
        vis.set_pcds(pcds)
        # //

        if idx[0].item() in sym_list:
            pose = pose.astype(np.float32)
            pose = torch.from_numpy(pose)
            pose = pose.cuda()
            pose = pose.transpose(1, 0)
            pose = pose.contiguous()

            target = target.astype(np.float32)
            target = torch.from_numpy(target)
            target = target.cuda()
            target = target.transpose(1, 0)
            target = target.contiguous()

            # inds = knn(target.unsqueeze(0), pose.unsqueeze(0))
            inds = knn.forward(target.unsqueeze(0), pose.unsqueeze(0))
            inds = inds.view(-1)

            target = torch.index_select(target, 1, inds-1)

            dis = pose.transpose(1, 0) - target.transpose(1, 0)
            dis = torch.norm(dis, dim=1)
            dis = torch.mean(dis, dim=0)
            dis = dis.item()
        else:
            dis = np.mean(np.linalg.norm(pose - target, axis=1))

        toc = time.time()
        times.append((toc - tic))

        if dis < diameter[idx[0].item()]:
            success_count[idx[0].item()] += 1
        
        num_count[idx[0].item()] += 1

        success_rate = float(np.sum(success_count)) / np.sum(num_count)
        bar.set_description(f'Succes rate {success_rate:.4f} Distance {dis:.4f}')

        # // show pose
        vis.custom_draw_geometry(f'Final pose with ADD score {dis:.4f}')
        # //

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f'Prediction time: Mean {avg_time:.4f} Std {std_time:.4f}')

    # for i in range(num_objects):
    #     success_rate = float(success_count[i]) / num_count[i]
    #     obj = objlist[i]
    #     print(f'Object {obj} Success rate {success_rate:.4f}')

    success_rate = float(np.sum(success_count)) / np.sum(num_count)
    print(f'Success rate for item {args.item} is {success_rate}')


if __name__ == '__main__':
    main()
