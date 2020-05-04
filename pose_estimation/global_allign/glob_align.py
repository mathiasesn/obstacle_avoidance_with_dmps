from tqdm import tqdm
import open3d as o3d
from dataset import dataset_global_align
import numpy as np
import yaml
import copy
import time
import sys

class global_alignment():

    def __init__(self, dataset_root):

        # Loading and making pointsclouds
        self.data_root = dataset_root 
        self.dataset = dataset_global_align(5000, self.data_root + '/Linemod_preprocessed')


        # from tqdm import tqdm
        # bar = tqdm(dataset)
        # for i, data in enumerate(bar, 0):
        #     cloud, model_pts, target, idx = data
        #     self.objs_cropped.append(cloud)
        #     self.models.append(model_pts)
        #     self.ground_truths.append(target)
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
        # FAST #
        #global_trans = self.execute_fast_global_registration(model_cloud, scene_cloud, model_fpfh, scene_fpfh)

        # GLOBAL 
        global_trans = self.execute_global_registration(model_cloud, scene_cloud, model_fpfh, scene_fpfh)

        # ICP/LOCAL #
        local_trans = self.refine_registration(model_cloud, scene_cloud, global_trans).transformation

        #print(f'Time took {time.time()-start:.4f} sec')
        return self.calculate_trans_error(model_cloud, target_cloud, local_trans)

        #self.draw_registration_result(model_cloud, target_cloud, local_trans)


    def execute_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        distance_threshold = 0.01

        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 3, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.1),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 5000))
        return result


    def execute_fast_global_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        distance_threshold = 0.01
        result = o3d.registration.registration_fast_based_on_feature_matching( source_down, target_down, source_fpfh, target_fpfh, o3d.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
        return result

    def refine_registration(self, source, target, result_ransac):
        distance_threshold = 0.01
        result = o3d.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation, o3d.registration.TransformationEstimationPointToPoint())
        return result

    def preprocess_point_cloud(self, pcd):
        radius_normal = 0.25
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))

        radius_feature = 0.5
       
        pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=10))
        return pcd_fpfh

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def calculate_trans_error(self, source, target, transform):
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

    def evalute_data(self):
        bar = tqdm(range(len(self.dataset)), ascii=True)
        num_of_obj = 0
        sucess = 0
        for index in bar:
            scene, model, target, obj_idx = self.dataset[index] # getting pointsclouds for object at index
            if (type(scene) is not np.ndarray) and (type(model) is not np.ndarray) and (type(target) is not np.ndarray) and (type(obj_idx) is not np.ndarray): # No points found when masking
                continue

            error = self.make_alignment(scene, model, target, obj_idx)
            diameter = self.find_obj_diameter(obj_idx)
            
            num_of_obj += 1
            if error < diameter:
                sucess += 1
            
            bar.set_description_str(f"Sucess rate {(sucess/(num_of_obj))*100:.2f}%") 
        


if __name__ == '__main__':
    glob_align = global_alignment('/home/mikkel/Documents/dataset/linemod')
    glob_align.evalute_data()
    #glob_align.make_alignment(0)
