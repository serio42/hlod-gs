#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIterationGivenLod, searchForMaxLod
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel,
                 load_iteration=None, load_lod=None,
                 shuffle=True, resolution_scales=[1.0], load_only_test_images=False,
                 load_image_device="cpu", load_by_image_names=None, load_ply=True):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.loaded_lod = None
        self.gaussians = gaussians

        if load_iteration:
            if load_lod != -1:
                self.loaded_lod = load_lod
            else:
                self.loaded_lod = searchForMaxLod(os.path.join(self.model_path, "point_cloud"))
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIterationGivenLod(os.path.join(self.model_path, "point_cloud"), self.loaded_lod)
            else:
                self.loaded_iter = load_iteration
                            
        self.train_cameras = {}
        self.test_cameras = {}
                    
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,
                                                         llffhold=args.llffhold, eval_proximity=args.eval_proximity,
                                                         load_ply=load_ply)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, load_ply=load_ply)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras) 
            random.shuffle(scene_info.test_cameras) 

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        if load_by_image_names is not None:
            filter_train_cameras = [camera for camera in scene_info.train_cameras if camera.image_name in load_by_image_names]

            for resolution_scale in resolution_scales:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(filter_train_cameras, resolution_scale, args, load_image_device)

        else:
            if load_only_test_images:
                for resolution_scale in resolution_scales:
                    self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, load_image_device)

            else:
                for resolution_scale in resolution_scales:
                    self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, load_image_device)
                    self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, load_image_device)

        if self.gaussians != None:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            f"lod_{self.loaded_lod}_iteration_{self.loaded_iter}",
                                                            "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, args.init_opacity)


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def save(self, iteration, lod):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/lod_{}_iteration_{}".format(lod, iteration))
        save_chunks = (
            getattr(self.gaussians, "_save_hierarchy_chunks", False)
            and self.gaussians._birth_lod is not None
            and self.gaussians._parent_idx is not None
        )
        if save_chunks:
            self.gaussians.save_hierarchy_chunks(
                point_cloud_path,
                export_flat_legacy=getattr(self.gaussians, "_export_flat_legacy", False),
            )
        else:
            self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # Hierarchical training metadata
        if self.gaussians._parent_idx is not None and not save_chunks:
            torch.save(self.gaussians._hierarchy_meta_payload(storage_format="flat"),
                       os.path.join(point_cloud_path, "hierarchy_meta.pt"))
        elif self.gaussians._mip_floor is not None:
            # Persist the anti-aliasing floor for plain PLY exports
            # (save_ply does not bake it).
            torch.save(self.gaussians._mip_floor.cpu(),
                       os.path.join(point_cloud_path, "mip_floor.pt"))
