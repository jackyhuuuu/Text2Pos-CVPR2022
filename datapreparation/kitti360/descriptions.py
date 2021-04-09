from typing import List
import numpy as np
from sklearn.cluster import DBSCAN

from datapreparation.kitti360.imports import Object3d, Cell, Description
from datapreparation.kitti360.utils import STUFF_CLASSES

def get_mask(points, cell_bbox):
    mask = np.bitwise_and.reduce((
        points[:, 0] >= cell_bbox[0],
        points[:, 1] >= cell_bbox[1],
        points[:, 2] >= cell_bbox[2],
        points[:, 0] <= cell_bbox[3],
        points[:, 1] <= cell_bbox[4],
        points[:, 2] <= cell_bbox[5],
    ))   
    return mask 

def cluster_stuff_object(obj, stuff_min, eps=0.75):
    """ Perform DBSCAN cluster, thresh objects by points again
    """
    # cluster = DBSCAN(eps=1.5, min_samples=300, leaf_size=30, n_jobs=-1).fit(obj.xyz)
    cluster = DBSCAN(eps=eps, n_jobs=-1).fit(obj.xyz)
    clustered_objects = []

    for i, label_value in enumerate(range(0, np.max(cluster.labels_) + 1)):
        mask = cluster.labels_ == label_value
        if np.sum(mask) < stuff_min:
            continue

        c_obj = obj.mask_points(mask)
        clustered_objects.append(c_obj)

    return clustered_objects

printed = False
# TODO: if instance-objects too slow, add simple center-based thresholding first
def describe_cell(bbox, scene_objects: List[Object3d], pose, scene_name, inside_fraction=1/3, stuff_min=500, num_mentioned=6):
    """Create the cell using all the objects in the scene.
    Instance-objects are threshed in/outside the scene (all points are retained)
    Stuff-objects' points are threshed inside the cell, clustered and then saved with new IDs
    CARE: object-ids are completely re-set after gathering and can repeat across cells!

    Args:
        bbox: Cell bbox
        scene_objects: Objects in scene
        pose: Pose
    """
    # Gather and cluster the objects
    global printed
    # if not printed:
    #     print('CARE: not using instance-obj!')
    #     printed = True

    cell_objects = []
    for obj in scene_objects:
        assert obj.id < 1e7

        mask = get_mask(obj.xyz, bbox)
        if obj.label in STUFF_CLASSES:
            if np.sum(mask) < stuff_min:
                continue

            cell_obj = obj.mask_points(mask)
            clustered_objects = cluster_stuff_object(cell_obj, stuff_min)
            cell_objects.extend(clustered_objects)
        else:
            if np.sum(mask) / len(mask) < inside_fraction:
                continue
            cell_objects.append(obj) # DEBUG: comment out

    # assert len(cell_objects) >= num_mentioned
    if len(cell_objects) < num_mentioned:
        return None

    # Reset all ids
    for id, obj in enumerate(cell_objects):
        obj.id = id + 1

    # Normalize objects, pose and cell based on the largest cell-edge ∈ [0, 1] (instance-objects can reach over edge)
    cell_size = np.max(bbox[3:6] - bbox[0:3])
    for obj in cell_objects:
        obj.xyz = (obj.xyz - bbox[0:3]) / cell_size
    pose = (pose - bbox[0:3]) / cell_size
    bbox = (bbox - np.hstack((bbox[0:3], bbox[0:3]))) / cell_size

    # Describe the post based on the clostest objects
    # Alternatives: describe in each direction, try to get many classes
    descriptions = []
    distances = np.linalg.norm([obj.get_closest_point(pose) - pose for obj in cell_objects], axis=1)
    closest_indices = np.argsort(distances)

    mentioned_objects = [cell_objects[idx] for idx in closest_indices[0:num_mentioned]]
    for obj in mentioned_objects:
        obj2pose = pose - obj.get_closest_point(pose) # e.g. "The pose is south of a car."
        if np.linalg.norm(obj2pose[0:2]) < 0.5 / cell_size: # Say 'on-top' if the object is very close (e.g. road), only calculated in x-y-plane!
            direction = 'on-top'
        else:
            if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]>=0: direction='east'
            if abs(obj2pose[0])>=abs(obj2pose[1]) and obj2pose[0]<=0: direction='west'
            if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]>=0: direction='north'
            if abs(obj2pose[0])<=abs(obj2pose[1]) and obj2pose[1]<=0: direction='south' 

        descriptions.append(Description(obj.id, direction, obj.label))

    return Cell(scene_name, cell_objects, descriptions, pose)



