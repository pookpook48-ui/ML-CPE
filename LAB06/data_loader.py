import os
import cv2

def load_image_paths(base_dir):
    image_paths = []
    labels = []
    classes = {'Cat': 0, 'Dog': 1}
    
    for class_name, label in classes.items():
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, fname)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        image_paths.append(img_path)
                        labels.append(label)
                except Exception:
                    continue
    return image_paths, labels, classes