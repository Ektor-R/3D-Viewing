import numpy as np

# Conf
IMG_WIDTH = 512



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

        lightDistances = np.linalg.norm(lightPositions-point, axis=1)
        attenuationFactors = 1/(lightDistances**2)

        # Reflection created by each light source
        reflections = lightIntensities * attenuationFactors[:,None] * k * lightAngles[:,None]

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

def calculateNormals(vertices: np.ndarray, faceIndices: np.ndarray) -> np.ndarray:
    """
        Calculate normal vector of all triangles on each of their vertices

        Arguments:
            vertices: Array with coordinates of vertices
            faceIndices: Array with the indexes of each vertices of the triangles

        Returns:
            array with the normal vectors on each vertex
    """
    
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