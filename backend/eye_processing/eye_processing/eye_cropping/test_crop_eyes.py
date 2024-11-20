from crop_eyes import crop_eyes_with_space

if __name__ == "__main__":
    # Change 'face_image.jpg' to your actual image path
    combined_eye_image = crop_eyes_with_space("face_image.jpg")

    if combined_eye_image is not None:
        print("Both eyes cropped and displayed successfully!")
    else:
        print("Not both eyes detected.")
