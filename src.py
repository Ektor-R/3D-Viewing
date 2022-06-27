import numpy as np
import triangle_rasterizer
import projection

# Conf



def ambient_light(k: float, I: np.ndarray) -> np.ndarray:
    """
        Calculate ambient light

        Arguments:
            k: Ambient reflection coefficient
            I: Global ambient illumination
        
        Returns:
            ambient light
    """
    
    return k * I



def diffuse_light(
        point: np.ndarray,
        N: np.ndarray, 
        color: np.ndarray, 
        k: float, 
        lightPositions: np.ndarray, 
        lightIntensities: np.ndarray
    ) -> np.ndarray:
        """
            Calculate diffuse reflection on point.

            Arguments:
                point: Point to calculate reflection on.
                N: surface perpendicular vector on point.
                color: color of point.
                k: light reflection coefficient.
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
        # TODO lightVector = 0 (point is a source)
        lightVectors = lightVectors/np.linalg.norm(lightVectors, axis=1)[:,None]

        # Calculate (cosine of) angle of each light source with N (dot product of each light vector with N)
        lightAngles = np.einsum('j,ij->i', N, lightVectors)

        #lightDistances = np.linalg.norm(lightPositions-point, axis=1)
        #attenuationFactors = 1/(lightDistances**2)

        # Reflection created by each light source
        reflections = lightIntensities * k * lightAngles[:,None] # * attenuationFactors[:,None]

        # TODO reflection larger than 1

        return color + np.sum(reflections, axis=0)



def specular_light(
        point: np.ndarray,
        N: np.ndarray, 
        color: np.ndarray, 
        cameraPosition: np.ndarray, 
        k: float,
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
                k: Light reflection coefficient.
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
        # TODO lightVector = 0 (point is a source)
        lightVectors = lightVectors / np.linalg.norm(lightVectors, axis=1)[:,None]

        # Calculate unit vector of camera center with point as the start
        cameraVector = cameraPosition - point
        # TODO cameraVector = 0 (point is camera center)
        cameraVector = cameraVector / np.linalg.norm(cameraVector)

        lightAngles = np.einsum(
                'j,ij->i', 
                cameraVector, 
                2 * N * np.einsum('j,ij->i', N, lightVectors)[:,None] - lightVectors
            )

        # Reflection created by each light source
        reflections = lightIntensities * k * (lightAngles[:,None]**n)

        # TODO reflection larger than 1

        return color + np.sum(reflections, axis=0)



def calculate_normals(vertices: np.ndarray, faceIndices: np.ndarray) -> np.ndarray:
    """
        Calculate normal vector of all triangles on each of their vertices

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

    normals = np.zeros([vertices.shape[0],3])

    normals[faceIndices[:,0]] = triangleNormals + vertices[faceIndices[:,0]]
    normals[faceIndices[:,1]] = triangleNormals + vertices[faceIndices[:,1]]
    normals[faceIndices[:,2]] = triangleNormals + vertices[faceIndices[:,2]]

    # TODO normal = 0

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
            # TODO n?
            vertsColors[i] = specular_light(bcoords, normals[i], vertsColors[i], cameraPosition, ks, 1, lightPositions, lightIntensities)

        # Clip color
        vertsColors[vertsColors < 0] = 0
        vertsColors[vertsColors > 1.] = 1.

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

        pass