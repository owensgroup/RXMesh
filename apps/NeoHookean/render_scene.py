#!/usr/bin/env python3
"""
Blender script to render OBJ files with consistent colors for different sphere instances.
Usage: blender --background --python render_scene.py -- --input output/scene_step_10.obj --output render_step_10.png
"""

import bpy
import bmesh
import sys
import argparse
from pathlib import Path
import colorsys
import random
import hashlib


def parse_args():
    """Parse command line arguments after the '--' separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render OBJ file in Blender with colored instances")
    parser.add_argument("--input", type=str, required=True, help="Input OBJ file path")
    parser.add_argument("--output", type=str, default="render.png", help="Output image path")
    parser.add_argument("--width", type=int, default=1920, help="Image width")
    parser.add_argument("--height", type=int, default=1080, help="Image height")
    parser.add_argument("--samples", type=int, default=128, help="Render samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for color generation")

    return parser.parse_args(argv)


def get_color_for_instance(instance_id, total_instances, seed=42):
    """Generate a consistent color for an instance ID using HSV color space."""
    # Use hash to ensure consistency across different runs
    hash_obj = hashlib.md5(f"{seed}_{instance_id}".encode())
    hash_int = int(hash_obj.hexdigest()[:8], 16)

    # Use golden ratio for better color distribution
    golden_ratio = 0.618033988749895
    hue = (hash_int * golden_ratio) % 1.0

    # Use high saturation and value for vibrant colors
    saturation = 0.7 + (hash_int % 30) / 100.0  # 0.7 to 1.0
    value = 0.8 + (hash_int % 20) / 100.0       # 0.8 to 1.0

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b, 1.0)


def separate_mesh_islands(obj):
    """Separate a mesh into disconnected components (islands)."""
    # Switch to object mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Separate by loose parts
    bpy.ops.mesh.separate(type='LOOSE')

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Return all selected objects (the separated parts)
    return [o for o in bpy.context.selected_objects]


def create_material(name, color):
    """Create a material with the given color."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    nodes.clear()

    # Create shader nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Set color
    principled_node.inputs['Base Color'].default_value = color
    principled_node.inputs['Metallic'].default_value = 0.2
    principled_node.inputs['Roughness'].default_value = 0.4

    # Link nodes
    links = mat.node_tree.links
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    return mat


def setup_scene():
    """Set up the Blender scene with camera and lighting."""
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Add camera
    bpy.ops.object.camera_add(location=(10, -10, 8))
    camera = bpy.context.object
    camera.rotation_euler = (1.1, 0, 0.785)  # Look at center
    bpy.context.scene.camera = camera

    # Add key area light
    bpy.ops.object.light_add(type='AREA', location=(8, -8, 10))
    key_light = bpy.context.object
    key_light.data.energy = 500
    key_light.data.size = 8
    key_light.rotation_euler = (0.8, 0, 0.785)

    # Add fill area light
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 6))
    fill_light = bpy.context.object
    fill_light.data.energy = 200
    fill_light.data.size = 6
    fill_light.rotation_euler = (1.0, 0, -0.5)

    # Add back area light for rim lighting
    bpy.ops.object.light_add(type='AREA', location=(0, 8, 5))
    back_light = bpy.context.object
    back_light.data.energy = 150
    back_light.data.size = 5
    back_light.rotation_euler = (1.57, 0, 3.14)

    # Add floor plane
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Floor"

    # Create floor material
    floor_mat = bpy.data.materials.new(name="Floor_Material")
    floor_mat.use_nodes = True
    floor_nodes = floor_mat.node_tree.nodes
    floor_nodes.clear()

    floor_output = floor_nodes.new(type='ShaderNodeOutputMaterial')
    floor_principled = floor_nodes.new(type='ShaderNodeBsdfPrincipled')

    # Set floor properties - light gray, slightly rough
    floor_principled.inputs['Base Color'].default_value = (0.4, 0.4, 0.4, 1.0)
    floor_principled.inputs['Metallic'].default_value = 0.0
    floor_principled.inputs['Roughness'].default_value = 0.6

    floor_links = floor_mat.node_tree.links
    floor_links.new(floor_principled.outputs['BSDF'], floor_output.inputs['Surface'])

    # Assign floor material
    floor.data.materials.append(floor_mat)

    # Set up world background
    world = bpy.context.scene.world
    world.use_nodes = True
    bg_node = world.node_tree.nodes['Background']
    bg_node.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1.0)
    bg_node.inputs['Strength'].default_value = 0.5


def render_obj_file(input_path, output_path, width=1920, height=1080, samples=128, seed=42):
    """Load OBJ file, separate instances, assign colors, and render."""
    print(f"Loading OBJ file: {input_path}")

    # Set up scene
    setup_scene()

    # Import OBJ file
    bpy.ops.wm.obj_import(filepath=str(input_path))

    # Get the imported object
    imported_objs = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']

    if not imported_objs:
        print("Error: No mesh objects found in OBJ file")
        return

    print(f"Found {len(imported_objs)} imported objects")

    # Separate into islands
    all_instances = []
    for obj in imported_objs:
        instances = separate_mesh_islands(obj)
        all_instances.extend(instances)

    print(f"Separated into {len(all_instances)} instances")

    # Sort instances by location for consistency
    all_instances.sort(key=lambda obj: (obj.location.x, obj.location.y, obj.location.z))

    # Assign colors to each instance
    for idx, obj in enumerate(all_instances):
        color = get_color_for_instance(idx, len(all_instances), seed)
        mat_name = f"Material_Instance_{idx}"

        # Create and assign material
        mat = create_material(mat_name, color)

        # Assign material to object
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        # Enable smooth shading
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.shade_smooth()

        print(f"Instance {idx}: color RGB{color[:3]}")

    # Set render settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.filepath = str(output_path)

    # Frame all objects in camera view
    bpy.ops.object.select_all(action='DESELECT')
    for obj in all_instances:
        obj.select_set(True)
    bpy.ops.view3d.camera_to_view_selected()

    # Render
    print(f"Rendering to: {output_path}")
    bpy.ops.render.render(write_still=True)
    print("Render complete!")

    # Save color mapping to file for consistency
    mapping_file = Path(output_path).with_suffix('.colors.txt')
    with open(mapping_file, 'w') as f:
        f.write(f"# Color mapping for {input_path}\n")
        f.write(f"# Seed: {seed}\n")
        f.write(f"# Total instances: {len(all_instances)}\n\n")
        for idx in range(len(all_instances)):
            color = get_color_for_instance(idx, len(all_instances), seed)
            f.write(f"Instance {idx}: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})\n")
    print(f"Color mapping saved to: {mapping_file}")


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_obj_file(
        input_path=input_path,
        output_path=output_path,
        width=args.width,
        height=args.height,
        samples=args.samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
