import os
import cv2
import numpy as np

os.makedirs("images", exist_ok=True)
filenames = [
    "images/0.png", "images/1.png", "images/2.png",
    "images/3.png", "images/4.png", "images/5.png"
]

FACE_LABELS = ["W", "G", "R", "B", "O", "Y"]

CAPTURE_INSTRUCTIONS = [
    "1: WHITE center  (Keep GREEN facing you / on Bottom)",
    "2: GREEN center  (Keep WHITE on Top)",
    "3: RED center    (Keep WHITE on Top)",
    "4: BLUE center   (Keep WHITE on Top)",
    "5: ORANGE center (Keep WHITE on Top)",
    "6: YELLOW center (Keep GREEN on Top)"
]

# Face indices: 0=W(U), 1=G(F), 2=R(R), 3=B(B), 4=O(L), 5=Y(D)
CORNER_SLOTS = [
    ((0, 2, 2), (2, 0, 0), (1, 0, 2)),  # 0: URF (W-R-G)
    ((0, 0, 2), (3, 0, 0), (2, 0, 2)),  # 1: UBR (W-B-R)
    ((0, 0, 0), (4, 0, 0), (3, 0, 2)),  # 2: ULB (W-O-B)
    ((0, 2, 0), (1, 0, 0), (4, 0, 2)),  # 3: UFL (W-G-O)
    ((5, 0, 2), (1, 2, 2), (2, 2, 0)),  # 4: DFR (Y-G-R)
    ((5, 2, 2), (2, 2, 2), (3, 2, 0)),  # 5: DRB (Y-R-B)
    ((5, 2, 0), (3, 2, 2), (4, 2, 0)),  # 6: DBL (Y-B-O)
    ((5, 0, 0), (4, 2, 2), (1, 2, 0)),  # 7: DLF (Y-O-G)
]

EDGE_SLOTS = [
    ((0, 1, 2), (2, 0, 1)),  # 0: UR (W-R)
    ((0, 0, 1), (3, 0, 1)),  # 1: UB (W-B)
    ((0, 1, 0), (4, 0, 1)),  # 2: UL (W-O)
    ((0, 2, 1), (1, 0, 1)),  # 3: UF (W-G)
    ((5, 1, 2), (2, 2, 1)),  # 4: DR (Y-R)
    ((5, 2, 1), (3, 2, 1)),  # 5: DB (Y-B)
    ((5, 1, 0), (4, 2, 1)),  # 6: DL (Y-O)
    ((5, 0, 1), (1, 2, 1)),  # 7: DF (Y-G)
    ((1, 1, 2), (2, 1, 0)),  # 8: FR (G-R)
    ((1, 1, 0), (4, 1, 2)),  # 9: FL (G-O)
    ((3, 1, 0), (2, 1, 2)),  # 10: BR (B-R)
    ((3, 1, 2), (4, 1, 0)),  # 11: BL (B-O)
]

SOLVED_CORNERS = [
    {"W", "R", "G"}, {"W", "B", "R"}, {"W", "O", "B"}, {"W", "G", "O"},
    {"Y", "G", "R"}, {"Y", "R", "B"}, {"Y", "B", "O"}, {"Y", "O", "G"}
]

SOLVED_EDGES = [
    {"W", "R"}, {"W", "B"}, {"W", "O"}, {"W", "G"},
    {"Y", "R"}, {"Y", "B"}, {"Y", "O"}, {"Y", "G"},
    {"G", "R"}, {"G", "O"}, {"B", "R"}, {"B", "O"}
]


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
    num_stickers = len(non_center_labs)
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


def process_images():
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
        cv2.imshow(f"Face {face_idx + 1} ({FACE_LABELS[face_idx]})", display_images[face_idx])

    print("Verify the detected labels on screen, then press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return all_faces


def extract_colors():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    captured = [False] * 6
    print("\nAlign the cube inside the green 3x3 box.")
    print("Press keys 1-6 to capture each face (or 'q' to quit):")
    for line in CAPTURE_INSTRUCTIONS:
        print("  " + line)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w, _ = display.shape

        box_size = min(480, h - 80, w - 80)
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for i in range(1, 3):
            step = (box_size * i) // 3
            cv2.line(display, (x1 + step, y1), (x1 + step, y2), (0, 255, 0), 1)
            cv2.line(display, (x1, y1 + step), (x2, y1 + step), (0, 255, 0), 1)

        status_text = "Captured: " + " ".join(
            f"[{FACE_LABELS[i]}]" if captured[i] else FACE_LABELS[i] for i in range(6)
        )
        cv2.putText(display, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Webcam - Align Cube in Box', display)
        key = cv2.waitKey(1) & 0xFF

        if ord('1') <= key <= ord('6'):
            idx = key - ord('0') - 1
            cube_crop = frame[y1:y2, x1:x2]
            cv2.imwrite(filenames[idx], cube_crop)
            captured[idx] = True
            print(f"Saved {filenames[idx]} -> {CAPTURE_INSTRUCTIONS[idx]}")

        if all(captured) or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not all(captured):
        print("Capture incomplete. Exiting.")
        return None

    return process_images()


def construct_cube(faces):
    if faces is None:
        return None

    corners = []
    seen_corners = set()

    for slot_idx, slot_coords in enumerate(CORNER_SLOTS):
        stickers = [faces[f][r][c] for (f, r, c) in slot_coords]
        sticker_set = set(stickers)

        if sticker_set not in SOLVED_CORNERS:
            raise ValueError(f"Invalid corner colors {stickers} at slot {slot_idx}. Rescan cube.")

        piece_id = SOLVED_CORNERS.index(sticker_set)
        if piece_id in seen_corners:
            raise ValueError(f"Duplicate corner piece ID {piece_id} detected. Rescan cube.")
        seen_corners.add(piece_id)

        orientation = next(i for i, color in enumerate(stickers) if color in ("W", "Y"))
        corners.append((piece_id, orientation))

    edges = []
    seen_edges = set()

    for slot_idx, slot_coords in enumerate(EDGE_SLOTS):
        stickers = [faces[f][r][c] for (f, r, c) in slot_coords]
        sticker_set = set(stickers)

        if sticker_set not in SOLVED_EDGES:
            raise ValueError(f"Invalid edge colors {stickers} at slot {slot_idx}. Rescan cube.")

        piece_id = SOLVED_EDGES.index(sticker_set)
        if piece_id in seen_edges:
            raise ValueError(f"Duplicate edge piece ID {piece_id} detected. Rescan cube.")
        seen_edges.add(piece_id)

        if "W" in stickers or "Y" in stickers:
            orientation = 0 if stickers[0] in ("W", "Y") else 1
        else:
            orientation = 0 if stickers[0] in ("G", "B") else 1
        edges.append((piece_id, orientation))

    if sum(ori for _, ori in corners) % 3 != 0:
        raise ValueError("Invalid corner twist parity (sum % 3 != 0). Check face orientation during scan.")

    if sum(ori for _, ori in edges) % 2 != 0:
        raise ValueError("Invalid edge flip parity (sum % 2 != 0). Check face orientation during scan.")
    with open("shared_memory.txt", "w") as f:
        f.write(" ".join(f"{pid}:{ori}" for pid, ori in corners) + "\n")
        f.write(" ".join(f"{pid}:{ori}" for pid, ori in edges) + "\n")
    print("\nSuccess! Validated cube state saved to shared_memory.txt")
    return [corners, edges]
def main():
    all_faces = extract_colors()
    if all_faces is None:
        return
    try:
        cube = construct_cube(all_faces)
        print("Corners (id, ori):", cube[0])
        print("Edges   (id, ori):", cube[1])
    except ValueError as e:
        print(f"\n[VALIDATION ERROR] {e}")

if __name__ == '__main__':
    main()