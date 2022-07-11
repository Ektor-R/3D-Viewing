from typing import Tuple
import numpy as np



def affine_transform(point: np.ndarray, angle = 0., axis: np.ndarray = np.nan, translation = np.array([0, 0, 0])) -> np.ndarray:
    """
        Perform 3D Affine Transformation

        Arguments:
            point: Point(s) to transform.
            angle: Angle of rotation.
            axis: Axis of rotation
            translation: Translation of point(s)

        Returns:
            new point(s) coordinates
    """

    # Pass rotation if axis is not defined or angle is 0
    if axis is not np.nan or angle != 0:
        # Create rotation matrix
        rotationMatrix = _rotation_matrix(angle, axis)

        # Rotate
        # Transpose coordinates before and after rotation in case multiple points have been given
        point = np.matmul(rotationMatrix, point.transpose()).transpose()

    # Translate
    point = np.add(point, translation)

    return point



def system_transform(point: np.ndarray, rotationMatrix: np.ndarray = np.nan, center = np.array([0, 0, 0])) -> np.ndarray:
    """
        Coordinate System Transformation

        Arguments:
            point: Point(s) to transform.
            rotationMatrix: Rotation matrix
            center: Center of new co. system

        Returns:
            point(s) coordinates in the new system
    """

    # Translate point to new center
    point = affine_transform(point, translation = -center)

    # Pass rotation if rotationMatrix is not defined
    if rotationMatrix is not np.nan:
        point = np.matmul(rotationMatrix, point.transpose()).transpose()

    return point



def project_cam(f: float, center: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
        Find projection of point(s) on camera

        Arguments:
            f: Camera f
            center: Camera center
            x: Camera x direction vector
            y: Camera y direction vector
            z: Camera z direction vector
            point: Point(s) to project
        Returns:
            point(s) projection on camera
            point(s) depth
    """

    # Transform to camera system
    point = system_transform(point, np.array([x, y, z]), center)
    
    # If point is vector, make it array.
    if point.ndim == 1:
        point = point[None]

    # Point projection: (x', y') = f * (x, y) / z
    verts2d = f * ( point[:, [0, 1]] / point[:, 2, None] )

    # Depth
    depth = point[:,2]

    return verts2d, depth



def project_cam_lookat(f: float, center: np.ndarray, lookat: np.ndarray, up: np.ndarray, verts3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
        Find projection of point(s) on camera

        Arguments:
            f: Camera f
            center: Camera center
            lookat: Camera lookat point
            up: Camera up unit vector
            verts3d: Coordinates of 3D points
        Returns:
            point(s) projection on camera
            point(s) depth
    """

    # Find unit vectors for the camera's system
    z = (lookat - center) / np.linalg.norm(lookat - center)

    t = up - np.dot(up, z) * z
    y = t / np.linalg.norm(t)

    x = np.cross(y, z)
    
    return project_cam(f, center, x, y, z, verts3d)



def rasterize(verts2d: np.ndarray, imgHeight: int, imgWidth: int, camHeight: float, camWidth: float) -> np.ndarray:
    """
        Project from camera to image

        Argument:
            verts2d: Coordinates of point(s)
            imgHeight: Image height
            imgWidth: Image width
            camHeight: Camera height
            camWidth: Camera width
        Returns:
            point(s) on image
    """
    
    # Stretch
    verts2d[:, 0] = verts2d[:, 0] * imgWidth / camWidth
    verts2d[:, 1] = verts2d[:, 1] * imgHeight / camHeight

    # Transform to image coordinates and get integer values
    verts2d = system_transform(verts2d, center = np.array([-imgWidth/2, -imgHeight/2])).round()
    # From bottom-top to top-bottom y
    verts2d[:, 1] = imgHeight - verts2d[:, 1]
    
    return verts2d



def _rotation_matrix(angle = 0., axis: np.ndarray = np.nan) -> np.ndarray:
    """
        Create a 3D rotation matrix

        Arguments:
            angle: Angle of rotation.
            axis: Axis of rotation
        Returns:
            rotation matrix
    """

    if axis is np.nan or angle == 0:
        return np.identity(3)
    
    # Get unit vector of axis
    u = axis / np.linalg.norm(axis)

    sinA = np.sin(angle)
    cosA = np.cos(angle)

    rotationMatrix = np.array([
        [
            (u[0]**2) * (1 - cosA) + cosA,
            u[0] * u[1] * (1 - cosA) - u[2] * sinA,
            u[0] * u[2] * (1 - cosA) + u[1] * sinA
        ],
        [
            u[1] * u[0] * (1 - cosA) + u[2] * sinA,
            (u[1]**2) * (1 - cosA) + cosA,
            u[1] * u[2] * (1 - cosA) - u[0] * sinA
        ],
        [
            u[2] * u[0] * (1 - cosA) - u[1] * sinA,
            u[2] * u[1] * (1 - cosA) + u[0] * sinA,
            (u[2]**2) * (1 - cosA) + cosA
        ]
    ])

    return rotationMatrix