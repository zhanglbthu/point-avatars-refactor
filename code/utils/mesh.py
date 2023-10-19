import torch
import torch.nn as nn
import numpy as np
import mcubes

from pytorch3d.io import save_obj

def generate_grid_points(bound_min, bound_max, resolution):
    
    X = torch.linspace(bound_min[0], bound_max[0], resolution)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution)

    grid_points = []
    for x in X:
        for y in Y:
            for z in Z:
                grid_points.append([x, y, z])

    return torch.tensor(grid_points)

# initialize sphere 
def generate_sphere_points(radius, n_init_points):
    init_points = torch.rand(n_init_points, 3) * 2.0 - 1.0
    init_normals = nn.functional.normalize(init_points, dim=1)
    init_points = init_normals * radius
    
    return init_points

def extract_geometry(output_path, sdf_grids, bound_min, bound_max, threshold=0.0, resolution=64):
    vertices, triangles = mcubes.marching_cubes(-sdf_grids, threshold)
    print("vertices: ", vertices.shape)
    print("triangles: ", triangles.shape)
    vertices = vertices / (resolution - 1.0) * 2.0 - 1.0
    
    # convert to torch tensor
    vertices = torch.from_numpy(vertices.astype(np.float32))
    triangles = torch.from_numpy(triangles.astype(np.int32))
    
    save_obj(output_path, vertices, triangles)

    return vertices, triangles

def save_mesh(output_path, vertices, triangles):
    save_obj(output_path, vertices, triangles)
    
if __name__ == '__main__':
    # generate sphere points
    init_points = generate_sphere_points(0.5, 1000)
    print(init_points.shape)
    print(init_points)