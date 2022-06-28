import numpy as np

# Conf
BACKGROUND = [1., 1., 1.]
M = 512
N = 512

def interpolate_color(x1: float, x2: float, x: float, C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    """
        Interpolate colour on a specified point between two points

        Arguments:
            x1: x coordinate (or y) of point no. 1
            x2: x coordinate (or y) of point no. 2
            x: x coordinate (or y) of point to calculate colour on
            C1: colour of point no. 1 (x1)
            C2: colour of point no. 2 (x2)

        Returns:
            colour for point x
    """

    if x1==x2:
        l = 1
    else:    
        l = abs(x2-x)/abs(x2-x1)
    
    return np.add( np.multiply(l, C1), np.multiply(1-l, C2) )



def shade_triangle(img: np.ndarray, verts2d: np.ndarray, vcolors: np.ndarray, shade_t: str) -> np.ndarray:
    """
        Draw triangle

        Arguments:
            img: Existing image to draw on
            verts2d:
            vcolors:
            shade_t: flat or gouraud
        
        Returns:
            new image
    """

    # Initialise some variables
    sidesHaveVerts = np.empty([3,2], 'int')     # Index represents each of the three sides. 
                                                # Values are the indexes for the vertices in verts2d (two for each side)
    Ymin = np.empty(3)                          # Index -> sides. Values -> Minimum Y coordinate of the two vertices
    Ymax = np.empty(3)                          # Index -> sides. Values -> Minimum Y coordinate of the two vertices
    sideGradient = np.empty(3)                  # Index -> sides. Values -> Gradient of each side

    activeMarginalPoints = np.array([np.nan, np.nan, np.nan])   # Index represents each of the three sides.
                                                                # Values are the x coordinate of the point where
                                                                # scan line meets the side. nan when side is inactive

    # Calculate triangle colour for flat algorithm
    if shade_t == 'flat':
        flatColour = np.array([
            np.sum(vcolors[:,0])/3,
            np.sum(vcolors[:,1])/3,
            np.sum(vcolors[:,2])/3
        ])

    # Store info for each side.
        #Sides are:
        #Side 0: vertice 0 to 1
        #Side 1: vertice 1 to 0
        #Side 2: vertice 2 to 0
    for k in range(3):
        sideStart = k
        sidesHaveVerts[k][0] = sideStart
        sideEnd = (k+1)<=2 and k+1 or 0         # If k+1=3: vertice=0
        sidesHaveVerts[k][1] = sideEnd

        Ymin[k] = min(verts2d[sideStart][1], verts2d[sideEnd][1])
        Ymax[k] = max(verts2d[sideStart][1], verts2d[sideEnd][1])

        if verts2d[sideEnd][0] == verts2d[sideStart][0]:    # When side is vertical.
            sideGradient[k] = np.inf
        else:
            sideGradient[k] = (                                     # ( yEnd-yStart ) / ( xend-xstart )
                (verts2d[sideEnd][1] - verts2d[sideStart][1])/
                (verts2d[sideEnd][0] - verts2d[sideStart][0])
            )
    
    # First scan line begins on minimum Y.
    activeSides = np.where(Ymin == Ymin.min())[0]
    for side in activeSides:
        activeMarginalPoints[side] = min(
            verts2d[
                sidesHaveVerts[side][
                    np.where( verts2d[ sidesHaveVerts[side] ][:,1] == Ymin.min() )
                ]
            ][:,0]
        )

    # Scan lines from minimum Y to maximum Y
    for Y in range(round(Ymin.min()), round(Ymax.max()) + 1): 
        # Clip if out of image size
        if 0 <= Y < np.shape(img)[0]:
            # Calculate line colour extremes for gouraud algorithm
            if shade_t == 'gouraud':
                startingLine = np.nanargmin(activeMarginalPoints)
                finishLine = np.nanargmax(activeMarginalPoints)

                scanLineStartColour = interpolate_color(
                    round(verts2d[ sidesHaveVerts[startingLine][0] ][1]),
                    round(verts2d[ sidesHaveVerts[startingLine][1] ][1]),
                    Y,
                    vcolors[ sidesHaveVerts[startingLine][0] ],
                    vcolors[ sidesHaveVerts[startingLine][1] ])

                scanLineEndColour = interpolate_color(
                    round(verts2d[ sidesHaveVerts[finishLine][0] ][1]),
                    round(verts2d[ sidesHaveVerts[finishLine][1] ][1]),
                    Y,
                    vcolors[ sidesHaveVerts[finishLine][0] ],
                    vcolors[ sidesHaveVerts[finishLine][1] ])

            # Scan line Y
            # Draw between min to max marginal points.
            for X in range(round(np.nanmin(activeMarginalPoints)), round(np.nanmax(activeMarginalPoints)) + 1):
                # Clip if out of image size
                if 0 <= X < np.shape(img)[1]:
                    if shade_t == 'flat':
                        img[int(Y)][int(X)] = flatColour
                    elif shade_t == 'gouraud':
                        img[int(Y)][int(X)] = interpolate_color(
                            round(np.nanmin(activeMarginalPoints)),
                            round(np.nanmax(activeMarginalPoints)),
                            X,
                            scanLineStartColour,
                            scanLineEndColour
                        )

        # Update active sides and marginal points
        for side in activeSides:
            if Ymax[side] == Y: # End of this line. Remove it and its marginal point
                activeSides = np.delete(activeSides, np.where(activeSides == side))
                activeMarginalPoints[side] = np.nan

        # Increase each remaining marginal points by 1/sideGradient
        for side, point in enumerate(activeMarginalPoints):
            if np.isnan(point) or np.isinf(sideGradient[side]): # Continue if no point or side is vertical
                continue
            activeMarginalPoints[side] = point + 1/sideGradient[side]

        # Add sides that have their minY = scanY+1
        # Add their marginal point
        for side in np.where(Ymin == Y+1)[0]:
            activeSides = np.append(activeSides, side)
            activeMarginalPoints[side] = min(
            verts2d[
                sidesHaveVerts[side][
                    np.where( verts2d[ sidesHaveVerts[side] ][:,1] == (Y+1) )
                ]
            ][:,0]
        )

    return img

    



def render(verts2d: np.ndarray, faces: np.ndarray, vcolors: np.ndarray, depth: np.ndarray, shade_t: str) -> np.ndarray:
    """
        Render image

        Arguments:
            verts2d:
            faces:
            vcolors:
            depth:
            shade_t: flat or gouraud

        Returns:
            rendered image
    """

    # Initialize image. Size and background colour is stored in constants
    img = np.full( (M, N, 3) , BACKGROUND)

    # Iterate through sorted faces by descending depth order and draw the triangles
    for face in _sort_faces(faces, depth):
        img = shade_triangle(
            img, 
            np.array([ verts2d[face[0]], verts2d[face[1]], verts2d[face[2]] ]),
            np.array([ vcolors[face[0]], vcolors[face[1]], vcolors[face[2]] ]),
            shade_t
        )

    return img



def _sort_faces(faces: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
        Sort faces by face depth (descending)

        Arguments:
            faces:
            depth:

        Returns:
            faces sorted
    """
    facesDepth = np.empty(np.shape(faces)[0])

    for index, face in enumerate(faces):
        facesDepth[index] = ( depth[face[0]] + depth[face[1]] + depth[face[2]] )/3
    
    return faces[np.argsort(-facesDepth)]
