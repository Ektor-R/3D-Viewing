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