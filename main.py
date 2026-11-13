import os
import cv2
import numpy as np

os.makedirs("images", exist_ok=True)
filenames = [
    "images/0.png", "images/1.png", "images/2.png",
    "images/3.png", "images/4.png", "images/5.png"
]

FACE_LABELS = ["W", "G", "R", "B", "O", "Y"]

def get_cell_lab(cube_crop, row, col):
    h, w, _ = cube_crop.shape
    cell_h, cell_w = h // 3, w // 3

    y1 = int((row + 0.30) * cell_h)
    y2 = int((row + 0.70) * cell_h)
    x1 = int((col + 0.30) * cell_w)
    x2 = int((col + 0.70) * cell_w)

    patch = cube_crop[y1:y2, x1:x2]
    median_bgr = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
    lab_pixel = cv2.cvtColor(np.uint8([[median_bgr]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
    return lab_pixel, (x1, y1, x2, y2)

def reconcile_colors(non_center_labs, reference_colors):
    num_stickers = len(non_center_labs)  # 48
    dist_matrix = np.zeros((num_stickers, 6), dtype=np.float32)
    for i, lab in enumerate(non_center_labs):
        for c_idx, label in enumerate(FACE_LABELS):
            dist_matrix[i, c_idx] = np.linalg.norm(lab - reference_colors[label])

    remaining_quota = {label: 8 for label in FACE_LABELS}
    assigned_labels = [None] * num_stickers
    unassigned = set(range(num_stickers))

    while unassigned:
        best_sticker = None
        best_color = None
        largest_margin = -1.0

        for idx in unassigned:
            available = [
                (dist_matrix[idx, c_idx], label)
                for c_idx, label in enumerate(FACE_LABELS)
                if remaining_quota[label] > 0
            ]
            available.sort(key=lambda x: x[0])

            margin = (available[1][0] - available[0][0]) if len(available) > 1 else 9999.0
            if margin > largest_margin:
                largest_margin = margin
                best_sticker = idx
                best_color = available[0][1]

        assigned_labels[best_sticker] = best_color
        remaining_quota[best_color] -= 1
        unassigned.remove(best_sticker)

    return assigned_labels

def processImages():
    images = []
    reference_colors = {}

    for idx, img_path in enumerate(filenames):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Missing {img_path}. Capture all 6 faces first.")
            return None
        images.append(img)

        center_lab, _ = get_cell_lab(img, 1, 1)
        reference_colors[FACE_LABELS[idx]] = center_lab

    non_center_labs = []
    non_center_meta = []  

    for face_idx, img in enumerate(images):
        for row in range(3):
            for col in range(3):
                if row == 1 and col == 1:
                    continue  
                lab, box = get_cell_lab(img, row, col)
                non_center_labs.append(lab)
                non_center_meta.append((face_idx, row, col, box))

    reconciled_labels = reconcile_colors(non_center_labs, reference_colors)

    all_faces = [[[None for _ in range(3)] for _ in range(3)] for _ in range(6)]
    display_images = [img.copy() for img in images]

    for face_idx in range(6):
        all_faces[face_idx][1][1] = FACE_LABELS[face_idx]
        _, (x1, y1, x2, y2) = get_cell_lab(images[face_idx], 1, 1)
        cv2.rectangle(display_images[face_idx], (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(display_images[face_idx], FACE_LABELS[face_idx], (x1 + 5, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(display_images[face_idx], FACE_LABELS[face_idx], (x1 + 5, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for (face_idx, row, col, (x1, y1, x2, y2)), label in zip(non_center_meta, reconciled_labels):
        all_faces[face_idx][row][col] = label
        cv2.rectangle(display_images[face_idx], (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_images[face_idx], label, (x1 + 5, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(display_images[face_idx], label, (x1 + 5, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    flat_colors = [c for face in all_faces for row in face for c in row]
    counts = {lbl: flat_colors.count(lbl) for lbl in FACE_LABELS}
    print("\nReconciled Color Counts:", counts)

    for face_idx in range(6):
        print(f"\n--- Face {face_idx + 1} ({FACE_LABELS[face_idx]} Center) ---")
        for row in all_faces[face_idx]:
            print(row)
        cv2.imshow(f"Reconciled Face {face_idx + 1}", display_images[face_idx])

    print("\nPress any key on an image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return all_faces

def extractColors():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Align the cube inside the green 3x3 box.")
    print(f"Press 1-6 to capture faces in order {FACE_LABELS}:")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w, _ = display.shape

        box_size = 480
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i in range(1, 3):
            step = (box_size * i) // 3
            cv2.line(display, (x1 + step, y1), (x1 + step, y2), (0, 255, 0), 1)
            cv2.line(display, (x1, y1 + step), (x2, y1 + step), (0, 255, 0), 1)

        cv2.imshow('Webcam - Align Cube in Box', display)
        key = cv2.waitKey(1) & 0xFF

        if ord('1') <= key <= ord('6'):
            idx = key - ord('0') - 1
            cube_crop = frame[y1:y2, x1:x2]
            cv2.imwrite(filenames[idx], cube_crop)
            print(f"Saved {filenames[idx]} ({FACE_LABELS[idx]} face)")

        if key == ord('6') or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return processImages()

def constructCube(faces):
    cornerList = []
    edgeList = []

    return [cornerList, edgeList]

def main():
    all_faces = extractColors()

    cube = constructCube(all_faces)

if __name__ == '__main__':
    main()
