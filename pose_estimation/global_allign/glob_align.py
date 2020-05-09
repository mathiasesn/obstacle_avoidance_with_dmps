from tqdm import tqdm
import open3d as o3d
from dataset import dataset_global_align
from scipy import spatial
import numpy as np
import yaml
import copy
import time
import csv
import sys

sym_list = [7,8]# list of which object is non-symmetric and symmetric


class global_alignment():

    def __init__(self, dataset_root):

        # Loading and making pointsclouds
        self.data_root = dataset_root 
        self.dataset = dataset_global_align(5000, self.data_root + '/Linemod_preprocessed')

    def make_alignment(self, scene, model, target, obj_idx):
        # Getting pointclouds at index
        scene_cloud = o3d.geometry.PointCloud()
        model_cloud = o3d.geometry.PointCloud()
        target_cloud = o3d.geometry.PointCloud()

        scene_cloud.points = o3d.utility.Vector3dVector(scene)
        model_cloud.points = o3d.utility.Vector3dVector(model)
        target_cloud.points = o3d.utility.Vector3dVector(target)

        scene_fpfh = self.preprocess_point_cloud(scene_cloud)
        model_fpfh = self.preprocess_point_cloud(model_cloud)
        target_fpfh = self.preprocess_point_cloud(target_cloud)
        

        #start = time.time()
        # FAST 
       # global_trans = self.execute_fast_global_registration(model_cloud, scene_cloud, model_fpfh, scene_fpfh)

        # GLOBAL 
        global_trans = self.execute_global_registration(model_cloud, scene_cloud, model_fpfh, scene_fpfh)

        # # ICP/LOCAL #
        local_trans = self.refine_registration(model_cloud, scene_cloud, global_trans).transformation

        #print(f'Time took {time.time()-start:.4f} sec')

        if self.dataset.list_of_objs.index(obj_idx) in self.dataset.symmetry_obj_idx:
            return self.calculate_ADDS_error(model_cloud, target_cloud, local_trans) 
        else:
            return self.calculate_ADD_error(model_cloud, target_cloud, local_trans)

            

        #self.draw_registration_result(model_cloud, target_cloud, local_trans)


    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        """standard global allignment
        Inspiration:
            http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html
        Arguments:
            source_down {[o3d.geometry.PointCloud]} -- [model point cloud]
            target_down {[o3d.geometry.PointCloud]} -- [scene point cloud]
            source_fpfh {[o3d.geometry.PointCloud]} -- [model feature descriptor]
            target_fpfh {[o3d.geometry.PointCloud]} -- [scene feature discriptor]

        Returns:
            [o3d.geometry.Transformation] -- [Global Transformation to allign model and scene]
        """
        distance_threshold = 0.01

        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            ransac_n=3, checkers=[ 
                o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold) # inlier distance
            ], criteria=o3d.registration.RANSACConvergenceCriteria(4000000, 10**4))
        return result


    def execute_fast_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        """fast global allignment
        Inspiration:
            http://www.open3d.org/docs/release/tutorial/Advanced/fast_global_registration.html
        Arguments:
            source_down {[o3d.geometry.PointCloud]} -- [model point cloud]
            target_down {[o3d.geometry.PointCloud]} -- [scene point cloud]
            source_fpfh {[o3d.geometry.PointCloud]} -- [model feature descriptor]
            target_fpfh {[o3d.geometry.PointCloud]} -- [scene feature discriptor]

        Returns:
            [o3d.geometry.Transformation] -- [Global Transformation to allign model and scene]
        """
        distance_threshold = 0.01
        result = o3d.registration.registration_fast_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh, o3d.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
        return result

    def refine_registration(self, source, target, result_ransac):
        """Local allignment using ICP
        Inspiration:
            http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html
        Arguments:
            source {[o3d.geometry.PointCloud]} -- [model point cloud]
            target {[o3d.geometry.PointCloud]} -- [scene point cloud]
            result_ransac {[o3d.geometry.Transformation]} -- [transformation result from global allignment]

        Returns:
            [o3d.geometry.Transformation] -- [Local Transformation to allign model and scene]
        """
        distance_threshold = 0.01
        result = o3d.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation, o3d.registration.TransformationEstimationPointToPoint())
        return result

    def preprocess_point_cloud(self, pcd):
        """Makes features descriptors on a point cloud

        Arguments:
            pcd {[o3d.geometry.PointCloud]} -- [input point cloud]

        Returns:
            [o3d.geometry.PointCloud] -- [feature descriptor point cloud]
        """
        radius_normal = 0.25
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=15))

        radius_feature = 0.5
        pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))

        return pcd_fpfh

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def calculate_ADDS_error(self, source, target, transform): 
        """Error for symmetric objects ADD-S
        Inspiration:
            https://github.com/yuxng/PoseCNN/blob/master/lib/utils/pose_error.py
        Arguments:
            source {[o3d.geometry.PointCloud]} -- [pointcloud of model]
            target {[o3d.geometry.PointCloud]} -- [pointcloud of ground truth]
            transform {[o3d.geometry.Transformation]} -- [transformation of model to fit ground truth in scene]

        Returns:
            [float] -- [error]
        """
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transform)

        source_xyz = np.asarray(source_temp.points)
        target_xyz = np.asarray(target_temp.points)

        # Calculate distances to the nearest neighbors from pts_gt to pts_est
        nn_index = spatial.cKDTree(target_xyz)
        nn_dists, _ = nn_index.query(source_xyz, k=1)

        error = nn_dists.mean()
        return error


    def calculate_ADD_error(self, source, target, transform):
        """
        Error for non-symmetric objects ADD
        """
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.transform(transform)

        source_xyz = np.asarray(source_temp.points)
        target_xyz = np.asarray(target_temp.points)

        return np.mean(np.linalg.norm(source_xyz - target_xyz, axis=1))

    def find_obj_diameter(self, obj_idx):
        diameter = 0
        path = self.data_root + '/dataset_config'
        meta_file = open(f'{path}/models_info.yml', 'r')
        meta = yaml.load(meta_file, Loader=yaml.FullLoader)
        diameter = (meta[obj_idx]['diameter'] / 1000.0 * 0.1)
        return diameter

    def __write_results(self, name, result):
        np.savetxt("logs/"+name, result, newline=",")

    def __write_raw_data(self, name, raw_data):
        csv_writer = csv.writer(open("logs/"+name, "w", newline=''))
            
        for i,row in enumerate(raw_data):
            csv_writer.writerow((i, row))

    def __calc_mean_score(self, acc_score):
        mean_score = []
        for acc_obj in acc_score:
            mean_score.append(np.mean(acc_obj))

        return mean_score

    def __calc_match_score_mean(self, math_score):
        match_mean = []
        for match_obj in math_score:
            mean = np.mean(match_obj)
            match_mean.append(mean)

        return match_mean

    def __calc_match_score_std(self, math_score):
        match_std = []
        for match_obj in math_score:
            mean = np.std(match_obj)
            match_std.append(mean)

        return match_std

    def evalute_data(self):
        bar = tqdm(range(len(self.dataset)), ascii=True)
        num_of_obj = 0
        sucess = 0
        accuracy_score = [[] for i in range(len(self.dataset.list_of_objs))] # accuracy score
        matching_score = [[] for i in range(len(self.dataset.list_of_objs))] # ADD and ADD-S mathcing score.

        prev_obj_idx = None # Used to see if new object
        score_idx = 0


        for index in bar:
            scene, model, target, obj_idx = self.dataset[index] # getting pointsclouds for object at index
            if (type(scene) is not np.ndarray) and (type(model) is not np.ndarray) and (type(target) is not np.ndarray) and (type(obj_idx) is not np.ndarray): # No points found when masking
                continue

            error = self.make_alignment(scene, model, target, obj_idx)
            diameter = self.find_obj_diameter(obj_idx)

            score_idx = self.dataset.list_of_objs.index(obj_idx)

            if error < diameter:
                accuracy_score[score_idx].append(1)
            else:
                accuracy_score[score_idx].append(0)

            matching_score[score_idx].append(error)

            bar.set_description_str(f"Sucess rate {(np.mean(accuracy_score[score_idx]))*100:.2f}%") 



        mean_score = self.__calc_mean_score(accuracy_score)
        total_score = np.mean(mean_score)
        match_score_mean = self.__calc_match_score_mean(matching_score)
        match_score_std = self.__calc_match_score_std(matching_score)

        print(f"Accuracy pr object {mean_score}")
        print(f"Mean ADD/ADD-S pr object {match_score_mean}")
        print(f"Standard-deviation ADD/ADD-S pr object {match_score_std}")
        print(f"Total score is {total_score}")
        
        self.__write_results("Accuracy.txt", np.append(mean_score, total_score))
        self.__write_results("ADD_ADD-S_mean.txt", match_score_mean)
        self.__write_results("ADD_ADD-S_std.txt", match_score_mean)

        self.__write_raw_data("acc_raw",accuracy_score)
        self.__write_raw_data("match_raw",matching_score)


   

if __name__ == '__main__':
    glob_align = global_alignment('/home/mikkel/Documents/dataset/linemod')
    glob_align.evalute_data()
    #glob_align.make_alignment(0)
