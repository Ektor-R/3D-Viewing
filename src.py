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