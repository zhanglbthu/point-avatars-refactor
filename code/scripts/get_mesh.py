import sys
sys.path.append('./')

from pyhocon import ConfigFactory

import torch
import torch.nn as nn

import utils.general as utils
from utils.mesh import generate_grid_points
from utils.mesh import save_mesh
from utils.plots import save_pcl_to_ply

from model.point_avatar_model import PointAvatar

import mcubes

import numpy as np

# get_mesh class
class MeshExtractor():
    def __init__(self):
        self.resolution = 128
        self.bound_min = torch.tensor([-1.0, -1.0, -1.0])
        self.bound_max = torch.tensor([1.0, 1.0, 1.0])
        self.threshold = 0.0
        
        self._init_model()
        # if torch.cuda.is_available():
        #     self.model.cuda()
        
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=min(int(self.conf.get_int('train.max_points_training') /self.model.pc.points.shape[0]),self.conf.get_int('train.max_batch',default='10')),
                                                           shuffle=False,
                                                           collate_fn=self.plot_dataset.collate_fn
                                                          )
        
        # obj path
        self.output_path = '../mesh/d_8.11.obj'
        
        # checkpoint path
        self.checkpoint_path = '../checkpoints/63.pth'
        
        # ply path
        self.ply_path = '../ply'
        
        # * update geometry_network, deformer_network, and rendering_network
        checkpoint = torch.load(self.checkpoint_path)['model_state_dict']
        n_points = checkpoint['pc.points'].shape[0]
        self.model.pc.init(n_points)
        self.model.pc = self.model.pc.cuda() 
        
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
    def _init_model(self):
        conf_file = "confs/subject1.conf"
        self.conf = ConfigFactory.parse_file(conf_file)
        self.conf.put('dataset.test.subsample', 1)
        self.conf.put('dataset.test.load_images', False)
        self.use_background = self.conf.get_bool('dataset.use_background', default=False)
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                          subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                          json_name=self.conf.get_string('dataset.json_name'),
                                                                                          use_mean_expression=self.conf.get_bool('dataset.use_mean_expression', default=False),
                                                                                          use_var_expression=self.conf.get_bool('dataset.use_var_expression', default=False),
                                                                                          use_background=self.use_background,
                                                                                          is_eval=False,
                                                                                          only_json=True,
                                                                                          **self.conf.get_config('dataset.train'))
        
        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_folder=self.conf.get_string('dataset.data_folder'),
                                                                                        subject_name=self.conf.get_string('dataset.subject_name'),
                                                                                        json_name=self.conf.get_string('dataset.json_name'),
                                                                                        use_background=self.use_background,
                                                                                        is_eval=True,
                                                                                        **self.conf.get_config('dataset.test'))   
        
        self.model = PointAvatar(conf=self.conf.get_config('model'),
                                shape_params=self.plot_dataset.shape_params,
                                img_res=self.plot_dataset.img_res,
                                canonical_expression=self.train_dataset.mean_expression,
                                canonical_pose=self.conf.get_float('dataset.canonical_pose', default=0.2),
                                use_background=self.use_background)
    
    def _print_model_parameters(self):
        print('Model parameters:')
        for name, param in self.model.named_parameters():
            print(name, param.shape)
        print("print model parameters done!")
    
    def _print_model_key_parameters(self):
        print('Model key parameters:')
        for name, param in self.model.named_parameters():
            if 'geometry_network' in name:
                print(name, param)
        print("print model key parameters done!")
        
    def extract_mesh(self):
        return self.model.get_mesh(self.output_path, self.resolution, self.bound_min, self.bound_max, self.threshold)
    
    def get_deformer_output(self, vertices):
        return self.model.deformer_network.query_weights(vertices)
    
    def get_points_sdf(self, points):
        geometry_output = self.model.geometry_network(points)
        sdf = geometry_output[:, 0]
        return sdf
    
    def extract_fields(self):
        N =64
        X = torch.linspace(self.bound_min[0], self.bound_max[0], self.resolution).split(N)
        Y = torch.linspace(self.bound_min[1], self.bound_max[1], self.resolution).split(N)
        Z = torch.linspace(self.bound_min[2], self.bound_max[2], self.resolution).split(N)
        
        u = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)
        
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        # print progress
                        print(f'Extracting field {xi * len(Y) * len(Z) + yi * len(Z) + zi + 1} / {len(X) * len(Y) * len(Z)}')
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        geometry_output = self.model.geometry_network(pts)
                        sdf_values = geometry_output[:, 0]
                        sdf_values = sdf_values.reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = sdf_values
        return u
    def extract_geometry(self, voxels):
        vertices, triangles = mcubes.marching_cubes(-voxels, self.threshold)
        b_min_np = self.bound_min.detach().cpu().numpy()
        b_max_np = self.bound_max.detach().cpu().numpy()
        vertices = vertices / (self.resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        
        vertices = torch.from_numpy(vertices.astype(np.float32))
        triangles = torch.from_numpy(triangles.astype(np.int32))
        
        return vertices, triangles
    
if __name__ == '__main__':
    mesh_extractor = MeshExtractor()
    #region get_canonical_code
    voxels = mesh_extractor.extract_fields()
    vertices, triangles = mesh_extractor.extract_geometry(voxels)
    print("vertices.shape:", vertices.shape)
    print("triangles.shape:", triangles.shape)
    points = nn.Parameter(vertices)
    mesh_extractor.model.pc.points = points
    print("points_shape:", mesh_extractor.model.pc.points.shape)
    #endregion
    #region get_flame_code
    # shapedirs, posedirs, lbs_weights, pnts_c_flame = mesh_extractor.get_deformer_output(vertices)
    # vertices = pnts_c_flame
    # save_mesh(mesh_extractor.output_path, vertices, triangles)
    #endregion
    #region get_deformed_code
    eval_iterator = iter(mesh_extractor.plot_dataloader)
    index = 175
    for i in range(index + 1):
        # print progress
        print(f'Extracting mesh {i + 1} / {index + 1}')
        indices, model_input, ground_truth = next(eval_iterator)
    
    if torch.cuda.is_available():   
        mesh_extractor.model.cuda()
    for k, v in model_input.items():
        try:
            model_input[k] = v.cuda()
        except:
            model_input[k] = v
    for k, v in ground_truth.items():
        try:
            ground_truth[k] = v.cuda()
        except:
            ground_truth[k] = v
    
    batch_size = model_input['expression'].shape[0]
    model_output = mesh_extractor.model(model_input)
    # deformed_points = model_output['deformed_points']
    # # extract one of the deformed points
    # deformed_points = deformed_points.reshape(batch_size, -1, 3)[0]

    # save_mesh(mesh_extractor.output_path, deformed_points, triangles)
    deformed_normals = model_output['pnts_normal_deformed']
    #endregion