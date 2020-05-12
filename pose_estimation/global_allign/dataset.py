from PIL import Image
import numpy as np
import numpy.ma as ma
import cv2
import yaml
import os
import os.path
import sys
import open3d as o3d
import random


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

class dataset_global_align():


    def __init__(self, num_points, root, show=False, sigma=0.0):
        self.list_of_objs = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.num_points = num_points
        self.root = root
        self.show = show
        self.sigma = sigma


        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}

        item_count = 0
        
        for item in self.list_of_objs:
            input_file = open(f"{self.root}/data/{'%02d' % item}/test.txt")
            bob = 0
            while True:
                item_count += 1
                bob += 1
                input_line = input_file.readline()

                # TEMPT TEST DELETE AFTER
                # if bob > 2:
                #     break
                
                if not input_line:
                    break

                if input_line[-1:] == "\n":
                    input_line = input_line[:-1]

                self.list_depth.append(f'{self.root}/data/{item:02d}/depth/{input_line}.png')


                self.list_label.append(f"{self.root}/segnet_results/{item:02d}_label/{input_line}_label.png")
                
                self.list_obj.append(item)
                self.list_rank.append(int(input_line))

            meta_file = open(f"{self.root}/data/{item:02d}/gt.yml", 'r')
            self.meta[item] = yaml.load(meta_file, Loader=yaml.FullLoader)
            self.pt[item] = ply_vtx(f'{self.root}/models/obj_{item:02d}.ply')

            print(f'Object {item} buffer loaded')

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.length = len(self.list_depth)
        
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
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
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        mask = mask_label * mask_depth

        rmin, rmax, cmin, cmax = get_bbox(mask_2_bbox(mask_label))

        # Ground truth
        target_r = np.resize(np.array(meta['cam_R_m2c']), (3,3))
        target_t = np.array(meta['cam_t_m2c'])
        
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        
        # no points in mask
        if len(choose) == 0:
            return None, None, None, None

        # not enough points in mask
        if len(choose) > self.num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

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

        # Getting model points and making it the same metric as imported data points
        model_pts = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_pts))]
        dellist = random.sample(dellist, len(model_pts) - self.num_pt_mesh_small)
        model_pts = np.delete(model_pts, dellist, axis=0)
        
        # target is ground truth
        target = np.dot(model_pts, target_r.T)
        target = np.add(target, target_t / 1000.0)
        out_t = target_t / 1000.0

        if self.show:
            depth_cv2 = cv2.imread(self.list_depth[index], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            min_val, max_val, _, _ = cv2.minMaxLoc(depth_cv2)
            depth_cv2 = cv2.convertScaleAbs(depth_cv2, 0, 255/max_val)
            cv2.imshow('Depth image', depth_cv2)

            depth_roi_cv2 = cv2.cvtColor(depth_cv2, cv2.COLOR_GRAY2BGR)
            depth_roi_cv2 = cv2.rectangle(depth_roi_cv2, (cmin,rmin), (cmax,rmax), (0,255,0), 3)
            cv2.imshow('Depth image with region of interest', depth_roi_cv2)

            scene_cloud = o3d.geometry.PointCloud()
            model_cloud = o3d.geometry.PointCloud()
            target_cloud = o3d.geometry.PointCloud()

            scene_cloud.points = o3d.utility.Vector3dVector(cloud)
            model_cloud.points = o3d.utility.Vector3dVector(model_pts)
            target_cloud.points = o3d.utility.Vector3dVector(target)

            o3d.visualization.draw_geometries([scene_cloud, target_cloud])

        return cloud, model_pts, target, obj # cloud is world with object cropped out, model_pts is model trying to fit and target is ground truth. 

def mask_2_bbox(mask):
    """Gets the bounding box of the mask
    
    Arguments:
        mask {np.array} -- mask image
    
    Returns:
        array -- x, y, w, h
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]

def get_bbox(bbox):
    """Ensures the bounding box is within the borders of the image.
    
    Arguments:
        bbox {array} -- bounding box
    
    Returns:
        int, int, int, int -- row minimum, row maximum, col minimum and col
        maximum of the bounding box.
    """
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
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
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)


if __name__ == '__main__':
    data_root = '/home/mikkel/Documents/dataset/linemod/Linemod_preprocessed'
    dataset = dataset_global_align(500, data_root, show=True)

    from tqdm import tqdm
    bar = tqdm(dataset)
    for i, data in enumerate(bar):
        cloud, model_pts, target, idx = data

        key = cv2.waitKey(0)
        if key == 27:
            print('Terminating because of key press')
            break