import numpy as np

# inter-sensor distance
def intersensor_distance(ref_Nord, ref_East, coordinates_N, coordinates_E):
    """ Calculates the intersensor distance from the reference station. Code
    valid for coordinates in [m].

    Parameters
    ----------
    ref_Nord    :   float
        Nord coordinates of the reference station
    ref_East    :   float
        East coordinates of the reference station
    coordinates_N   :   array_like
        list of the Nord coordinates of the stations in the array_like
    coordinates_E   :   array_like
        list of the East coordinates of the stations in the array_like
    """

    nb_sens = len(coordinates_N)
    distance = np.zeros(nb_sens)

    for sensor in range(nb_sens):
        dNord = ref_Nord - coordinates_N[sensor]
        dEast = ref_East - coordinates_E[sensor]
        distance[sensor] = sqrt(dNord ** 2 + dEast ** 2)

    return distance
