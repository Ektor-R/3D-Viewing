import numpy as np

import triangle_rasterizer
import projection

# Conf
N_PHONG = 1



def ambient_light(ka: float, Ia: np.ndarray) -> np.ndarray:
    """
        Calculate ambient light

        Arguments:
            ka: Ambient reflection coefficient
            Ia: Global ambient illumination
        
        Returns:
            ambient light
    """
    
    return ka * Ia



def diffuse_light(
        point: np.ndarray,
        N: np.ndarray, 
        color: np.ndarray, 
        kd: float, 
        lightPositions: np.ndarray, 
        lightIntensities: np.ndarray
    ) -> np.ndarray:
        """
            Calculate diffuse reflection on point.

            Arguments:
                point: Point to calculate reflection on.
                N: surface perpendicular vector on point.
                color: color of point.
                kd: light reflection coefficient.
                lightPositions: List with coordinates of light sources.
                lightIntensities: List with light intensity of each color source
                
            Returns:
                color of point including light reflection 
        """

        # If lightPositions is vector, make it array.
        if lightPositions.ndim == 1:
            lightPositions = lightPositions[None]
        if lightIntensities.ndim == 1:
            lightIntensities = lightIntensities[None]

        # Calculate unit vectors of each light source with point as the start
        lightVectors = lightPositions - point
        lightVectors = lightVectors/np.linalg.norm(lightVectors, axis=1)[:,None]

        # Calculate (cosine of) angle of each light source with N (dot product of each light vector with N)
        lightAngles = np.einsum('j,ij->i', N, lightVectors)

        # Reflection created by each light source
        reflections = lightIntensities * kd * lightAngles[:,None]

        return color + np.sum(reflections, axis=0)



def specular_light(
        point: np.ndarray,
        N: np.ndarray, 
        color: np.ndarray, 
        cameraPosition: np.ndarray, 
        ks: float,
        n: float,
        lightPositions: np.ndarray,
        lightIntensities: np.ndarray
    ) -> np.ndarray:
        """
            Calculate specular reflection on point

            Arguments:
                point: Point to calculate reflection on.
                N: Surface perpendicular vector on point.
                color: Color of point.
                cameraPosition: Camera center coordinates
                ks: Light reflection coefficient.
                n: Phong coefficient.
                lightPositions: List with coordinates of light sources.
                lightIntensities: List with light intensity of each color source
                
            Returns:
                color of point including light reflection 
        """
        
        # If lightPositions is vector, make it array.
        if lightPositions.ndim == 1:
            lightPositions = lightPositions[None]
        if lightIntensities.ndim == 1:
            lightIntensities = lightIntensities[None]

        # Calculate unit vectors of each light source with point as the start
        lightVectors = lightPositions - point
        lightVectors = lightVectors / np.linalg.norm(lightVectors, axis=1)[:,None]

        # Calculate unit vector of camera center with point as the start
        cameraVector = cameraPosition - point
        cameraVector = cameraVector / np.linalg.norm(cameraVector)

        lightAngles = np.einsum(
                'j,ij->i', 
                cameraVector, 
                2 * N * np.einsum('j,ij->i', N, lightVectors)[:,None] - lightVectors
            )

        # Reflection created by each light source
        reflections = lightIntensities * ks * (np.absolute(lightAngles[:,None])**n) * np.sign(lightAngles)[:,None]

        return color + np.sum(reflections, axis=0)



def calculate_normals(vertices: np.ndarray, faceIndices: np.ndarray) -> np.ndarray:
    """
        Calculate normal vector on each vertex

        Arguments:
            vertices: Array with coordinates of vertices
            faceIndices: Array with the indexes of each vertices of the triangles

        Returns:
            array with the normal vectors on each vertex
    """

    # If faceIndices is vector, make it array.
    if faceIndices.ndim == 1:
        faceIndices = faceIndices[None]
    
    triangleNormals = np.cross(
        vertices[faceIndices[:,1]] - vertices[faceIndices[:,0]], 
        vertices[faceIndices[:,2]] - vertices[faceIndices[:,0]]
    )
    triangleNormals = triangleNormals / np.linalg.norm(triangleNormals, axis=1)[:,None]

    normals = np.zeros([vertices.shape[0],3])

    for index, face in enumerate(faceIndices):
        normals[face] += triangleNormals[index]

    return normals / np.linalg.norm(normals, axis=1)[:,None]



def render_object(
        shader: str,
        focal: float, 
        eye: np.ndarray, 
        lookat: np.ndarray, 
        up: np.ndarray, 
        backgroundColor: np.ndarray, 
        imageM: int, 
        imageN: int, 
        cameraHeight: float,
        cameraWidth: float,
        verts: np.ndarray,
        vertsColors: np.ndarray,
        faceIndices: np.ndarray,
        ka: float,
        kd: float,
        ks: float,
        lightPositions: np.ndarray,
        lightIntensities: np.ndarray,
        Ia: np.ndarray
    ) -> np.ndarray:
        """
            Render object

            Arguments:
                shader: Shader type. gouraud or phong.
                focal: Camera focal length.
                eye: Camera center.
                lookat: Camera lookat.
                up: Camera up.
                backgroundColor: Image background color.
                imageM: image row size.
                imageN: Image column size.
                cameraHeight: Camera height.
                cameraWidth: Camera width.
                verts: Coordinates of vertices.
                vertsColors: Colors of vertices.
                faceIndices: Indexing of vertices of triangles.
                ka: Ambient coefficient.
                kd: Diffuse coefficient.
                ks: Specular coefficient.
                lightPositions: Coordinates of light sources.
                lightIntensities: Intensity of light sources.
                Ia: Intensity of ambient light.

            Returns:
                rgb image
        """

        normals = calculate_normals(verts, faceIndices)

        [verts2d, depth] = projection.project_cam_lookat(focal, eye, lookat, up, verts)
        verts2d = projection.rasterize(verts2d, imageM, imageN, cameraHeight, cameraWidth)

        # Sort triangles based on depth
        faceIndices = triangle_rasterizer._sort_faces(faceIndices, depth)

        # Initialize image.
        img = np.full( (imageM, imageN, 3) , backgroundColor)

        if shader == 'gouraud':
            for face in faceIndices:
                img = shade_gouraud(
                        verts2d[face], 
                        normals[face], 
                        vertsColors[face], 
                        np.mean(verts[face], axis=0), 
                        eye, 
                        ka, 
                        kd, 
                        ks, 
                        lightPositions, 
                        lightIntensities, 
                        Ia, 
                        img
                    )
        elif shader == 'phong':
            for face in faceIndices:
                img = shade_phong(
                        verts2d[face], 
                        normals[face], 
                        vertsColors[face], 
                        np.mean(verts[face], axis=0), 
                        eye, 
                        ka, 
                        kd, 
                        ks, 
                        lightPositions, 
                        lightIntensities, 
                        Ia, 
                        img
                    )

            img = np.clip(img, 0., 1.)

        return img



def shade_gouraud(
        verts2d: np.ndarray,
        normals: np.ndarray,
        vertsColors: np.ndarray,
        bcoords: np.ndarray,
        cameraPosition: np.ndarray,
        ka: float,
        kd: float,
        ks: float,
        lightPositions: np.ndarray,
        lightIntensities: np.ndarray,
        Ia: np.ndarray,
        X: np.ndarray
    ):
        """
            Shade triangle with gouraud algorithm

            Arguments:
                verts2d: Triangle vertices coordinates on camera.
                normals: Normal vectors of vertices.
                vertsColors: Triangle vertices colors.
                bcoords: Centroid of triangle.
                cameraPosition: Camera center.
                ka: Ambient coefficient.
                kd: Diffuse coefficient.
                ks: Specular coefficient.
                lightPositions: Coordinates of light sources.
                lightIntensities: Intensity of light sources.
                Ia: Intensity of ambient light.
                X: Image.

            Returns:
                image X with the new triangle
        """

        vertsColors = vertsColors + ambient_light(ka, Ia)

        for i in (0,2):
            vertsColors[i] = diffuse_light(bcoords, normals[i], vertsColors[i], kd, lightPositions, lightIntensities)
            vertsColors[i] = specular_light(bcoords, normals[i], vertsColors[i], cameraPosition, ks, N_PHONG, lightPositions, lightIntensities)

        # Clip color
        vertsColors = np.clip(vertsColors, 0., 1.)

        return triangle_rasterizer.shade_triangle(X, verts2d, vertsColors, 'gouraud')



def shade_phong(
        verts2d: np.ndarray,
        normals: np.ndarray,
        vertsColors: np.ndarray,
        bcoords: np.ndarray,
        cameraPosition: np.ndarray,
        ka: float,
        kd: float,
        ks: float,
        lightPositions: np.ndarray,
        lightIntensities: np.ndarray,
        Ia: np.ndarray,
        X: np.ndarray
    ):
        """
            Shade triangle with phong algorithm

            Arguments:
                verts2d: Triangle vertices coordinates on camera.
                normals: Normal vectors of vertices.
                vertsColors: Triangle vertices colors.
                bcoords: Centroid of triangle.
                cameraPosition: Camera center.
                ka: Ambient coefficient.
                kd: Diffuse coefficient.
                ks: Specular coefficient.
                lightPositions: Coordinates of light sources.
                lightIntensities: Intensity of light sources.
                Ia: Intensity of ambient light.
                X: Image.

            Returns:
                image X with the new triangle
        """

        return triangle_rasterizer.shade_triangle(
                X, 
                verts2d, 
                vertsColors, 
                'phong',
                normals,
                bcoords,
                cameraPosition,
                ka,
                kd,
                ks,
                N_PHONG,
                lightPositions,
                lightIntensities,
                Ia
            )