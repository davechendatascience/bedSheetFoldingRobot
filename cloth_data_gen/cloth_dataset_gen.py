import bpy
import numpy as np
import os
import random
from mathutils import Vector, Euler

# Parameters
output_dir = "./output"
n_samples = 10
length_range = (1.5, 3.0)
width_range = (1.0, 2.5)
res = 40  # Subdivisions per side

def ensure_dirs():
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/keypoints", exist_ok=True)

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_cloth(length, width):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=res, y_subdivisions=res, size=1)
    cloth = bpy.context.active_object
    cloth.scale = (length/2, width/2, 1)
    cloth.name = "Cloth"
    bpy.ops.object.shade_smooth()
    return cloth

def assign_random_color(obj):
    # Create and assign a random colored material
    mat = bpy.data.materials.new("ClothColorMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    color = (random.random(), random.random(), random.random(), 1)
    bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.clear()
    obj.data.materials.append(mat)

def bend_cloth(obj):
    bend = obj.modifiers.new(name="Bend", type='SIMPLE_DEFORM')
    bend.deform_method = 'BEND'
    bend.angle = np.radians(random.uniform(60, 150))
    bend.origin = None
    bend.limits = (0, 1)
    bend.deform_axis = random.choice(['X', 'Y'])

def fold_cloth(obj):
    disp = obj.modifiers.new(name="Fold", type='DISPLACE')
    tex = bpy.data.textures.new("FoldTex", type='CLOUDS')
    disp.texture = tex
    disp.strength = random.uniform(0.12, 0.33)

def curve_cloth(obj):
    # Apply a random bend (curve) to the cloth around a random axis and with a random angle
    bend = obj.modifiers.new(name="CurveBend", type='SIMPLE_DEFORM')
    bend.deform_method = 'BEND'
    # Choose a random bending axis (X, Y, or Z)
    bend.deform_axis = random.choice(['X', 'Y', 'Z'])
    # Apply a random angle (positive or negative bend)
    bend.angle = np.radians(random.uniform(-100, 100))
    # Optionally: position the object for a visible arch (move origin), or leave as default
    bend.origin = None

def random_deform(obj):
    # Existing random bend/fold
    if random.random() < 0.5:
        bend_cloth(obj)
    if random.random() < 0.6:
        fold_cloth(obj)
    # Add random curving with 50% probability
    if random.random() < 0.5:
        curve_cloth(obj)

def get_corner_indices(grid_res):
    n = grid_res
    return [
        0,               # Bottom-left
        n - 1,           # Bottom-right
        n * n - 1,     # Top-left
        n * n - 2        # Top-right
    ]

def apply_random_rotation(obj):
    # Apply a random 3D rotation
    rot_x = np.radians(random.uniform(-10, 10))      # keep cloth mostly upright, but slight x/y tilt is ok
    rot_y = np.radians(random.uniform(-10, 10))
    rot_z = np.radians(random.uniform(0, 360))
    obj.rotation_euler = Euler((rot_x, rot_y, rot_z), 'XYZ')

def visualize_keypoints(obj, indices):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()
    mat = bpy.data.materials.new("RedMat")
    mat.diffuse_color = (1, 0, 0, 1)
    for idx in indices:
        co = mesh_eval.vertices[idx].co
        glob = obj.matrix_world @ co
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.06, location=glob)
        kp = bpy.context.active_object
        kp.data.materials.append(mat)
        kp.name = "Keypoint"
    obj_eval.to_mesh_clear()

def setup_camera():
    bpy.ops.object.camera_add(location=(0, -6, 3), rotation=(1.25, 0, 0))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    bpy.context.scene.render.resolution_x = 400
    bpy.context.scene.render.resolution_y = 400
    return cam

def world_to_pixel(coord_world, camera, scene):
    # Manual conversion from 3D world to 2D pixel coordinates
    co_local = camera.matrix_world.normalized().inverted() @ coord_world
    z = -co_local.z
    if z == 0.0:
        return -1, -1
    frame = camera.data.view_frame(scene=scene)
    right = (frame[1] - frame[0]).length / 2
    top = (frame[2] - frame[1]).length / 2
    x_pixel = int((co_local.x / (-co_local.z)) * (scene.render.resolution_x / 2) / right + (scene.render.resolution_x / 2))
    y_pixel = int((co_local.y / (-co_local.z)) * (scene.render.resolution_y / 2) / top + (scene.render.resolution_y / 2))
    return (x_pixel, y_pixel)

def render_sample(i, cloth, cam, indices):
    scene = bpy.context.scene
    bpy.context.scene.render.filepath = f"{output_dir}/images/cloth_{i:04d}.png"
    bpy.ops.render.render(write_still=True)
    # Save 3D and 2D (pixel-space) keypoints
    with open(f"{output_dir}/keypoints/cloth_{i:04d}.txt", "w") as f:
        f.write("x_world, y_world, z_world, x_pixel, y_pixel\n")
        for idx in indices:
            global_co = cloth.matrix_world @ cloth.data.vertices[idx].co
            px, py = world_to_pixel(global_co, cam, scene)
            f.write(f"{global_co.x:.4f}, {global_co.y:.4f}, {global_co.z:.4f}, {px}, {py}\n")

def main():
    ensure_dirs()
    for i in range(n_samples):
        clear_scene()
        l = random.uniform(*length_range)
        w = random.uniform(*width_range)
        cloth = create_cloth(l, w)
        assign_random_color(cloth)              # Assign random color to cloth
        random_deform(cloth)
        apply_random_rotation(cloth)            # Apply random 3D rotation before rendering
        cam = setup_camera()
        indices = get_corner_indices(res)
        visualize_keypoints(cloth, indices)
        render_sample(i, cloth, cam, indices)
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
