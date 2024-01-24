
import math
def cheungsmethod (df_data, drivechain, total_mass, gravity, crr, density, CdA):
    """
    Calculate alt_diff based on time series data.

    Parameters:
    - data: DataFrame with columns 'power', 'speed', 'wind'.
    - drivechain: Drivechain efficiency
    - total_mass: Total mass (in kg)
    - gravity: Acceleration due to gravity (in m/s^2)
    - crr: Coefficient of rolling resistance
    - density: Air density (in kg/m^3)
    - CdA: Drag coefficient times frontal area (CdA)

    Returns:
    - alt_diff_series: Series of alt_diff values
    """

    # Calculate alt_diff for each time step, starting from row 1
    alt_diff_series = (
                              (df_data['power'][1:] * drivechain) -
                              (0.5 * total_mass * (
                                      (df_data['speed'][0:]) ** 2 - (df_data['speed'].shift(1)[0:]) ** 2)) -
                              (total_mass * gravity * df_data['speed'][0:] * crr) -
                              (0.5 * density * CdA * ((df_data['speed'][0:] + df_data['wind'][0:]) ** 2) * df_data[
                                                                                                               'speed'][
                                                                                                           0:])
                      ) / (total_mass * gravity)

    alt_diff_series.at[0] = df_data['alt'][0]

    return alt_diff_series


def add_wind(speed, angle):
    """
    Calculates the wind speed in a different direction based on the given wind speed and angle.
    """
    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle)

    # Calculate the wind speed in the desired direction
    wind_speed = speed * math.cos(angle_rad)

    return wind_speed