import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.face_model import FaceModel

def main():
    # Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Facial Model')

    # Get coordinates for full facial model
    full_face_coords = FaceModel().get()
    full_face_x = full_face_coords[:, 0]
    full_face_y = full_face_coords[:, 1]
    full_face_z = full_face_coords[:, 2]

    # Get coordinates for simplified facial model
    reye_indices = np.array([33, 133])
    leye_indices = np.array([362, 263])
    mouth_indices = np.array([185, 409])
    simplified_face_coords = full_face_coords[np.concatenate([reye_indices, leye_indices, mouth_indices])]
    simplified_face_x = simplified_face_coords[:, 0]
    simplified_face_y = simplified_face_coords[:, 1]
    simplified_face_z = simplified_face_coords[:, 2]

    # Plot full facial model and simplified model
    ax.plot_trisurf(full_face_x, full_face_y, full_face_z, linewidth=0, antialiased=True, alpha=0.3, label='Full facial model')
    ax.plot_trisurf(simplified_face_x, simplified_face_y, simplified_face_z, linewidth=0, antialiased=True, alpha=0.8, label='Simplified facial model')

    # Plot center of the face
    center_x, center_y, center_z = simplified_face_coords.mean(axis=0)
    ax.scatter(center_x, center_y, center_z, label='Center of the face')


    # Show plot
    plt.show()

if __name__ == '__main__':
    main()
