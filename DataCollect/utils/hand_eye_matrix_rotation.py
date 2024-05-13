import numpy as np
import csv

# Step 1: Read the matrix from the CSV file
def read_matrix_from_csv(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append([float(value) for value in row])
    return np.array(matrix)

def rotation_matrix_x(theta):
    """Return the rotation matrix for a rotation of theta degrees about the X-axis."""
    theta = np.radians(theta)
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(phi):
    """Return the rotation matrix for a rotation of phi degrees about the Y-axis."""
    phi = np.radians(phi)
    return np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

def rotation_matrix_z(psi):
    """Return the rotation matrix for a rotation of psi degrees about the Z-axis."""
    psi = np.radians(psi)
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

def rotate_matrix(matrix, theta, phi, psi):
    """Rotate the matrix about the X, Y, and Z axes."""
    Rx = rotation_matrix_x(theta)
    Ry = rotation_matrix_y(phi)
    Rz = rotation_matrix_z(psi)
    
    # Combine rotations
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # Multiply only the rotation part of the matrix
    rotated_submatrix = np.dot(matrix[:3, :3], R.T)
    
    # Update the input matrix's rotation part
    matrix[:3, :3] = rotated_submatrix
    
    return matrix



# Step 3: Save the matrix to a new CSV file
def save_matrix_to_csv(matrix, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in matrix:
            writer.writerow(row)

if __name__ == "__main__":
    # Read the matrix
    matrix = read_matrix_from_csv("C:/Users/camp/GIT/arthro_nerf/camera_calibration/handeye_matrix.csv")
    
    theta = 1.7  # Rotation about X
    phi = 3.4   # Rotation about Y
    psi = 89   # Rotation about Z
    rotated_matrix = rotate_matrix(matrix, theta, phi, psi)



    
    # Save the rotated matrix to a new CSV file
    save_matrix_to_csv(rotated_matrix, "C:/Users/camp/GIT/arthro_nerf/camera_calibration/handeye_matrix_new_2.csv")

