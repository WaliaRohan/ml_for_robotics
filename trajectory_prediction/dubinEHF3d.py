import numpy as np


def dubinEHF3d(east1, north1, alt1, psi1, east2, north2, r, step, gamma):
    """
    dubinEHF3d - Computes the minimum-length Dubin's curve between start (east1, north1, alt1, psi1)
    and end (east2, north2) positions with a free end heading. The path considers a constant flight
    path angle gamma, and step size between points.

    Parameters:
    - east1, north1, alt1: Start position
    - psi1: Initial heading
    - east2, north2: End position
    - r: Turn radius
    - step: Step size between path points
    - gamma: Flight path angle

    Returns:
    - path: Array of shape (N, 3) for path points (east, north, alt)
    - psi_end: Final heading at the end point
    - num_path_points: Number of points in the computed path
    """
    
    MAX_NUM_PATH_POINTS = 1000
    path = np.zeros((MAX_NUM_PATH_POINTS, 3))  # Initialize the path array
    r_sq = r**2

    # Normalize psi1 to the range [0, 2*pi)
    psi1 = psi1 % (2 * np.pi)

    # Left and right circles about p1
    theta_l = psi1 + np.pi / 2
    eastc_l = east1 + r * np.cos(theta_l)
    northc_l = north1 + r * np.sin(theta_l)

    theta_r = psi1 - np.pi / 2
    eastc_r = east1 + r * np.cos(theta_r)
    northc_r = north1 + r * np.sin(theta_r)

    # Distance from p2 to circle centers
    d2c_l_sq = (east2 - eastc_l)**2 + (north2 - northc_l)**2
    d2c_r_sq = (east2 - eastc_r)**2 + (north2 - northc_r)**2
    d2c_l = np.sqrt(d2c_l_sq)
    d2c_r = np.sqrt(d2c_r_sq)

    if d2c_l < r or d2c_r < r:
        print('No solution: distance of p1 and p2 is lower than turn radius.')
        return path, 0, 0

    # Angle from circle centers to p2
    theta_c_l = np.arctan2(north2 - northc_l, east2 - eastc_l) % (2 * np.pi)
    theta_c_r = np.arctan2(north2 - northc_r, east2 - eastc_r) % (2 * np.pi)

    # Length of tangent lines
    lt_l_sq = d2c_l_sq - r_sq
    lt_r_sq = d2c_r_sq - r_sq
    lt_l = np.sqrt(lt_l_sq)
    lt_r = np.sqrt(lt_r_sq)

    # Start and end angles on circles
    theta_start_l = theta_r
    theta_start_r = theta_l

    theta_d_l = np.arccos(r / d2c_l)
    theta_end_l = (theta_c_l - theta_d_l) % (2 * np.pi)

    theta_d_r = np.arccos(r / d2c_r)
    theta_end_r = (theta_c_r + theta_d_r) % (2 * np.pi)

    # Adjust angles to be within the correct range
    theta_end_l = (theta_end_l + 2 * np.pi) % (2 * np.pi)
    theta_end_r = (theta_end_r + 2 * np.pi) % (2 * np.pi)

    # Arc lengths
    arc_l = abs(theta_end_l - theta_start_l)
    arc_r = abs(theta_end_r - theta_start_r)
    arc_length_l = r * arc_l
    arc_length_r = r * arc_r

    total_length_l = arc_length_l + lt_l
    total_length_r = arc_length_r + lt_r

    # Path calculation
    if total_length_l < total_length_r:
        # Left turn arc
        if arc_length_l > 0.1:
            theta_step = step / r
            num_arc_seg = max(2, int(np.ceil(arc_l / theta_step)))
            angles = np.linspace(theta_start_l, theta_end_l, num_arc_seg)
            alt_end = alt1 + arc_length_l * np.tan(gamma)
            altitude = np.linspace(alt1, alt_end, num_arc_seg)
            arc_traj = np.array([[eastc_l + r * np.cos(a), northc_l + r * np.sin(a), alt] for a, alt in zip(angles, altitude)])
        else:
            arc_traj = np.array([[east1, north1, alt1]])
            num_arc_seg = 1

        # Straight line
        if lt_l > 0.1:
            num_line_seg = max(2, int(np.ceil(lt_l / step)))
            alt_begin = arc_traj[-1, 2]
            alt_end = alt_begin + lt_l * np.tan(gamma)
            line_traj = np.column_stack((
                np.linspace(arc_traj[-1, 0], east2, num_line_seg),
                np.linspace(arc_traj[-1, 1], north2, num_line_seg),
                np.linspace(alt_begin, alt_end, num_line_seg)
            ))
        else:
            line_traj = np.zeros((1, 3))
            num_line_seg = 0

    else:
        # Right turn arc
        if arc_length_r > 0.1:
            theta_step = step / r
            num_arc_seg = max(2, int(np.ceil(arc_r / theta_step)))
            angles = np.linspace(theta_start_r, theta_end_r, num_arc_seg)
            alt_end = alt1 + arc_length_r * np.tan(gamma)
            altitude = np.linspace(alt1, alt_end, num_arc_seg)
            arc_traj = np.array([[eastc_r + r * np.cos(a), northc_r + r * np.sin(a), alt] for a, alt in zip(angles, altitude)])
        else:
            arc_traj = np.array([[east1, north1, alt1]])
            num_arc_seg = 1

        # Straight line
        if lt_r > 0.1:
            num_line_seg = max(2, int(np.ceil(lt_r / step)))
            alt_begin = arc_traj[-1, 2]
            alt_end = alt_begin + lt_r * np.tan(gamma)
            line_traj = np.column_stack((
                np.linspace(arc_traj[-1, 0], east2, num_line_seg),
                np.linspace(arc_traj[-1, 1], north2, num_line_seg),
                np.linspace(alt_begin, alt_end, num_line_seg)
            ))
        else:
            line_traj = np.zeros((1, 3))
            num_line_seg = 0

    # Combine arc and line trajectories
    if num_line_seg > 1:
        num_path_points = num_arc_seg + num_line_seg - 1
        path[:num_path_points, :] = np.vstack((arc_traj, line_traj[1:]))
    else:
        num_path_points = num_arc_seg
        path[:num_path_points, :] = arc_traj

    # Compute final heading
    psi_end = np.arctan2(north2 - arc_traj[-1, 1], east2 - arc_traj[-1, 0])

    return path[:num_path_points], psi_end, num_path_points


