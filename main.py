import mss
import numpy as np
import cv2 as cv

# Function to capture a portion of the screen
def capture_screen():
    with mss.mss() as sct:
        # Define the monitor/screen area to capture
        monitor = {"top": 103, "left": 550, "width": 179, "height": 60}
        sct_img = np.array(sct.grab(monitor))
        return cv.cvtColor(sct_img, cv.COLOR_BGRA2BGR)

# Main logic with template matching
def main():
    print(cv.__version__)
    # Load the "Defeat" template image in grayscale
    defeat_template = cv.imread('TemplatePicture.png', cv.IMREAD_GRAYSCALE)
    w, h = defeat_template.shape[::-1]  # Get width and height of the template
    
    while True:
        # Capture the screen region where "Defeat" might appear
        frame = capture_screen()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert to grayscale for matching

        # Perform template matching
        result = cv.matchTemplate(frame_gray, defeat_template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        # If max_val is above a certain threshold, consider it a match
        threshold = 0.3  # Adjust this threshold as needed
        if max_val > threshold:
            print("Defeat detected!")
            # Draw a rectangle around the matched region
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Show the captured image with (or without) the rectangle
        cv.imshow('Captured Screen', frame)

        # Press 'q' to quit the window
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
