#In this code, the estimation of the rotation angle of the maglev is done

import cv2
import aruco_library
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\Tesi Alessandra\Tesi\images\magnete27.mp4')

frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
angles = []  # List to store angles for each frame
times = []  # List to store corresponding times

while True:
    ret, frame = cap.read()
    frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Get the current frame number

    if not ret:
        break

    # Detect markers and calculate orientation
    Detected_ArUco_markers = aruco_library.detect_ArUco(frame)
    angle = aruco_library.Calculate_orientation(Detected_ArUco_markers)  # Calculate orientation
    
    # If the marker with ID 1 is detected, store the angle and the time (adjust to available videos)
    if 1 in Detected_ArUco_markers:
        marker_angle = angle.get(1)  # Get the angle of the marker with ID 1
        if marker_angle is not None:
            angles.append(marker_angle)  # Add the angle to the list
            times.append(frame_no / frame_rate)  # Calculate the time corresponding to the frame

    # Exit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()

plt.plot(times, angles, label='Angle of Marker 1 (degrees)')
plt.xlabel('Time (seconds)')
plt.ylabel('Angle (degrees)')
plt.title('Rotation of Marker 1 as a Function of Time')
plt.legend()
plt.grid(True)
plt.show()
