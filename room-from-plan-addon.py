bl_info = {
    "name": "Room by plan",
    "author": "Alexander Ermakov",
    "version": (0, 1),
    "blender": (3, 5, 0),
    "location": "View3D > N > Create Room",
    "description": "Creates an interior scene from a processed plan image",
    "warning": "",
    "doc_url": "",
    "category": "Add Mesh",
}

import bpy
import bpy_extras
import numpy as np
import json
import math
from mathutils import Vector
# ImportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty
from bpy.types import Operator


    
# Windows
window_values = []

# Doors
door_values = []

# Furniture
furniture_values = []

image_location = None
horizontal_image_size = None
vertical_image_size = None
walls = None
windows = None
doors = None
furniture_list = None


def import_reference_image():
    
    collection_name = 'Reference Image'
    
    if collection_name not in bpy.data.collections:
        reference_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(reference_collection)
    else:
        reference_collection = bpy.data.collections.get(collection_name)    
    
    reference_collection.hide_viewport = False
    reference_collection.hide_select = False
    
    bpy.ops.view3d.snap_cursor_to_center()

    # Import reference image as a background image
    bpy.ops.object.load_reference_image(filepath=image_location)
    reference_image = bpy.context.object
    reference_image.name = "PlanReferenceImage"
    reference_image.show_empty_image_perspective = True
    reference_image.show_empty_image_only_axis_aligned = False
    reference_image.empty_image_depth = 'BACK'
    reference_image.empty_image_side = 'FRONT'
    reference_image.use_empty_image_alpha = True
    reference_image.color[3] = 0.5

    reference_image.scale.x = horizontal_image_size / 5
    reference_image.scale.y = vertical_image_size / 5
    reference_image.hide_render = True  # Hide the image in renders
    reference_image.rotation_euler = (0, 0, 0)
    
    reference_collection.objects.link(reference_image)
    bpy.context.scene.collection.objects.unlink(reference_image)


def create_walls_floor_and_ceiling():
    
    global walls
    
    collection_name = 'Walls Floor and Ceiling'
    
    if collection_name not in bpy.data.collections:
        room_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(room_collection)
    else:
        room_collection = bpy.data.collections.get(collection_name)
    
    room_collection.hide_viewport = False
    room_collection.hide_select = False
    
    
    # Create walls as edges
    verts = []
    edges = []

    for wall in walls:
        wall_verts = []
        for point in wall:
            x = point[0]
            y = point[1]
            z = 0
            vertex = [x, y, z]
            if vertex not in verts:
                verts.append(vertex)
                wall_verts.append(len(verts) - 1)  # Store the index of the added vertex
            else:
                wall_verts.append(verts.index(vertex))  # Get the index of the existing vertex

        # Connect the vertices of the wall to form edges
        for i in range(len(wall_verts) - 1):
            edges.append([wall_verts[i], wall_verts[i + 1]])

    # Create the mesh from the vertices and edges
    mesh = bpy.data.meshes.new("Walls")
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    
    # Create object for the walls
    walls = bpy.data.objects.new("Walls", mesh)
    node_tree_name = "HolesCutter"
    node_tree = bpy.data.node_groups.get(node_tree_name)
    walls_modifier = walls.modifiers.new(name='Hole Cutter', type='NODES')
    walls_modifier.node_group = node_tree
    
    room_collection.objects.link(walls)

    # Create floor and ceiling objects
    floor_mesh = mesh.copy()
    floor_mesh.name = "Floor"
    floor_mesh.transform(walls.matrix_world)

    ceiling_mesh = mesh.copy()
    ceiling_mesh.name = "Ceiling"
    ceiling_mesh.transform(walls.matrix_world)

    floor_obj = bpy.data.objects.new("Floor", floor_mesh)
    ceiling_obj = bpy.data.objects.new("Ceiling", ceiling_mesh)

    room_collection.objects.link(floor_obj)
    room_collection.objects.link(ceiling_obj)

    # Set floor and ceiling materials
    floor_material = bpy.data.materials.new(name="Floor Material")
    floor_obj.data.materials.append(floor_material)
    ceiling_material = bpy.data.materials.new(name="Ceiling Material")
    ceiling_obj.data.materials.append(ceiling_material)

    # Assign materials to floor and ceiling
    floor_material.diffuse_color = (0.2, 0.2, 0.2, 1.0)  # Adjust floor color
    ceiling_material.diffuse_color = (0.8, 0.8, 0.8, 1.0)  # Adjust ceiling color

    # Set floor and ceiling location
    floor_obj.location.z = 0
    ceiling_obj.location.z = 2.5  # Adjust ceiling height as needed

    # Extrude walls to create wall objects
    bpy.context.view_layer.objects.active = walls
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(
        MESH_OT_extrude_region={"mirror": True},
        TRANSFORM_OT_translate={"value": (0, 0, context.scene.my_tool.wall_height)}  # Extrude walls to a height of 2.5 units
    )
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.context.view_layer.objects.active = floor_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill()
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.context.view_layer.objects.active = ceiling_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill()
    bpy.ops.object.mode_set(mode='OBJECT')


def process_data(data):
    global image_location, horizontal_image_size, vertical_image_size, walls, windows, doors, furniture_list
    # Extract necessary information from the JSON data
    image_location = data.get("image_location")
    print(image_location)
    horizontal_image_size = data.get("horizontal_image_size")
    vertical_image_size = data.get("vertical_image_size")
    walls = data.get("walls")
    windows = data.get("windows")
    doors = data.get("doors")
    furniture_list = data.get("furniture")

    for i, window in enumerate(windows, 1):
        window_values.append({
            "center": window.get(f"{i}_center"),
            "size": window.get(f"{i}_size"),
            "rotation_angle": window.get(f"{i}_rotation_angle"),
            "casements": window.get(f"{i}_casements")
        })

    # Print the window values
    for window in window_values:
        
#        obj_copy = bpy.data.objects['Chair'].copy()
#        obj_copy.name = window['furniture_name']
#        location = window['center']
#        obj_copy.location = [window['center'][0] / 1000, window['center'][1] / 1000, 0]
#        # [201, 75] .modifiers["Chair geometry nodes"]["Input_3"] = 1.34
#        # widht
#        obj_copy.modifiers["Chair geometry nodes"]["Input_3"] = window['size'][0] / 1000
#        # length
#        obj_copy.modifiers["Chair geometry nodes"]["Input_4"] = window['size'][1] / 1000
#        obj_copy.rotation_euler[2] = math.radians(window['rotation_angle'])

#        bpy.context.collection.objects.link(obj_copy)
        
        
        print("Window Center:", window["center"])
        print("Window Size:", window["size"])
        print("Window Rotation Angle:", window["rotation_angle"])
        print("Window Furniture Name: window")
        print()
    
    for i, door in enumerate(doors, 1):
        door_values.append({
            "center": door.get(f"{i}_center"),
            "size": door.get(f"{i}_size"),
            "rotation_angle": door.get(f"{i}_rotation_angle"),
            "door_type": door.get(f"{i}_door_type")
        })
        
    for door in door_values:    
        print("Door Center:", door["center"])
        print("Door Size:", door["size"])
        print("Door Rotation Angle:", door["rotation_angle"])
        print("Door Furniture Name: door")
        print()
    
    for i, furniture in enumerate(furniture_list, 1):
        furniture_values.append({
            "center": furniture.get(f"{i}_center"),
            "size": furniture.get(f"{i}_size"),
            "rotation_angle": furniture.get(f"{i}_rotation_angle"),
            "furniture_name": furniture.get(f"{i}_furniture_name")
        })
        
    for furniture in furniture_values:    
        print("Furniture Center:", furniture["center"])
        print("Furniture Size:", furniture["size"])
        print("Furniture Rotation Angle:", furniture["rotation_angle"])
        print("Furniture Name:", furniture["furniture_name"])
        print()

def create_wall(wall, verts_raw, edges_raw):
    wall_verts = []
    wall_edges = []

    # Create vertices
    for point in wall:
        vert_name = "a" + str(len(verts_raw))
        vert_coords = [vert_name] + point
        verts_raw.append(vert_coords)
        wall_verts.append(vert_name)

    # Create edges
    for i in range(len(wall_verts) - 1):
        edge = (wall_verts[i], wall_verts[i + 1])
        edges_raw.append(edge)
        wall_edges.append(edge)

    # Connect the last and first vertices to close the loop
    edge = (wall_verts[-1], wall_verts[0])
    edges_raw.append(edge)
    wall_edges.append(edge)

def read_json_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
        process_data(data)

class ImportDataOperator(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    bl_idname = "object.import_data_operator"
    bl_label = "Import Data"
    
    filter_glob: bpy.props.StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        filepath = self.filepath
        read_json_data(filepath)
        return {'FINISHED'}


class AddFurnitureOperator(bpy.types.Operator):
    bl_idname = "object.add_furniture_operator"
    bl_label = "Add Furniture"
    bl_description = "Adds furniture to the scene at the cursor position"

    preset_enum: bpy.props.EnumProperty(
        name = "Furniture selector",
        description = "Select an option",
        items = [
            ('OP1', "Sofa", "Add a tabouret to the scene"),
            ('OP2', "Table", "Add a table to the scene"),
            ('OP3', "Bed", "Add a shelf to the scene"),
            ('OP4', "Chair", "Add a chair to the scene" )
        ]
    )
    
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preset_enum")

    def execute(self, context):
        cursor_location = context.scene.cursor.location
        bpy.ops.object.select_all(action='DESELECT')
        if self.preset_enum == 'OP1':
            obj_copy = bpy.data.objects["Sofa"].copy()
            obj_copy.location = cursor_location
            obj_copy.name = "Pasted sofa"
            bpy.context.collection.objects.link(obj_copy)
        elif self.preset_enum == 'OP3':
            obj_copy = bpy.data.objects["Bed"].copy()
            obj_copy.location = cursor_location
            obj_copy.name = "Pasted bed"
            bpy.context.collection.objects.link(obj_copy)

        return {'FINISHED'}


class CreateRoomSceneOperator(bpy.types.Operator):
    bl_idname = "object.create_room_scene_operator"
    bl_label = "Create Room Scene"
    bl_description = "Creates scene from imported data"

    def execute(self, context):
        
        print(image_location)
        
        import_reference_image()
        
        # Set the wall height
        wall_height = context.scene.wall_height
        create_walls_floor_and_ceiling()
        
        return {'FINISHED'}
    
class MyProperties(bpy.types.PropertyGroup):
    wall_height: FloatProperty(
        name="Wall Height",
        description="Set the height of the walls",
        default=2.5,
        min=2.4,
        soft_max=10.0
    )


class CreateRoomPanel(bpy.types.Panel):
    bl_label = "Create Room Panel"
    bl_idname = "OBJECT_PT_room_generator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Create Room"

    def draw(self, context):
        layout = self.layout

        obj = context.object

        row = layout.row()
        row.label(text="Create room from processed plan image data", icon='MESH_CUBE')
        
#        row = layout.row()
#        row.label(text="Proces data and generate room", icon="TIME")
        
        row = layout.row()
        row.operator(ImportDataOperator.bl_idname, text="Import Data", icon="FILE")
        
        layout.prop(context.scene.wall_height, 'wall_height')
#        layout.operator('object.apply_wall_height')
        
        row = layout.row()
        row.operator("object.create_room_scene_operator")
        
        row = layout.row()
        row.operator("object.add_furniture_operator", icon="EVENT_F")
        
        
        

_classes = [
    # ImportSomeData,
    ImportDataOperator,
    CreateRoomPanel,
    AddFurnitureOperator,
    CreateRoomSceneOperator,
    MyProperties
]

def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
        
    bpy.types.Scene.wall_height = bpy.props.PointerProperty(type=MyProperties)
    


def unregister():
    for cls in _classes:
        bpy.utils.unregister_class(cls)
        
    del bpy.types.Scene.wall_height


if __name__ == "__main__":
    register()
