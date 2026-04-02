import streamlit as st
import tempfile
import os
import csv
import cv2
import ast
import numpy as np
import pandas as pd
import base64
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv
from add_missing_data import interpolate_bounding_boxes




def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_encoded}");
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# دلوقتي تقدر تنادي عليها عادي

set_background('./sample.png')


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANPR System",
    page_icon="🚗",
    layout="wide"
)


@st.cache_resource
def load_models():
    coco = YOLO("yolov8n.pt")
    lp = YOLO("./model/best.pt")
    return coco, lp


coco_model, lp_model = load_models()
VEHICLES = [2, 3, 5, 7]


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10,
                line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img



st.title(" ANPR System 🚗")
st.markdown("Upload a video and the system will detect all license plates automatically.")

uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    video_path = tfile.name

    if st.button("▶ Run Detection", type="primary"):


        results = {}
        mot_tracker = Sort()
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress = st.progress(0, text="Detecting plates...")
        frame_nmr = -1
        ret = True

        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if not ret:
                break

            results[frame_nmr] = {}

            detections = coco_model(frame)[0]
            detections_ = []
            for det in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = det
                if int(class_id) in VEHICLES:
                    detections_.append([x1, y1, x2, y2, score])

            track_ids = mot_tracker.update(np.asarray(detections_)) if detections_ else np.empty((0, 5))

            license_plates = lp_model(frame)[0]
            for lp in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = lp
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
                if car_id == -1:
                    continue

                crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
                text, text_score = read_license_plate(thresh)

                if text:
                    results[frame_nmr][car_id] = {
                        "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                        "license_plate": {
                            "bbox": [x1, y1, x2, y2],
                            "text": text,
                            "bbox_score": score,
                            "text_score": text_score,
                        },
                    }

            progress.progress(
                min((frame_nmr + 1) / total_frames, 1.0),
                text=f"Detecting... frame {frame_nmr + 1} / {total_frames}"
            )

        cap.release()
        progress.empty()


        write_csv(results, "./test.csv")

        with open("test.csv", "r") as f:
            data = list(csv.DictReader(f))

        if data:
            interpolated = interpolate_bounding_boxes(data)
            with open("test_interpolated.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "frame_nmr", "car_id", "car_bbox", "license_plate_bbox",
                    "license_plate_bbox_score", "license_number", "license_number_score"
                ])
                writer.writeheader()
                writer.writerows(interpolated)


        progress2 = st.progress(0, text="Rendering output video...")

        df_results = pd.read_csv("./test_interpolated.csv")
        out_path = os.path.join(tempfile.gettempdir(), "anpr_out.mp4")

        cap2 = cv2.VideoCapture(video_path)
        fps = cap2.get(cv2.CAP_PROP_FPS)
        width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


        license_plate = {}
        for car_id in np.unique(df_results["car_id"]):
            max_score = np.amax(df_results[df_results["car_id"] == car_id]["license_number_score"])
            best_row = df_results[
                (df_results["car_id"] == car_id) &
                (df_results["license_number_score"] == max_score)
                ]
            license_plate[car_id] = {
                "license_crop": None,
                "license_plate_number": best_row["license_number"].iloc[0],
            }

            cap2.set(cv2.CAP_PROP_POS_FRAMES, best_row["frame_nmr"].iloc[0])
            ret, frame = cap2.read()

            x1, y1, x2, y2 = ast.literal_eval(
                best_row["license_plate_bbox"].iloc[0]
                .replace("[ ", "[").replace("   ", " ").replace("  ", " ").replace(" ", ",")
            )
            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
            license_plate[car_id]["license_crop"] = license_crop


        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fn = -1
        ret = True
        while ret:
            ret, frame = cap2.read()
            fn += 1
            if not ret:
                break

            df_ = df_results[df_results["frame_nmr"] == fn]
            for row_indx in range(len(df_)):

                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                    df_.iloc[row_indx]["car_bbox"]
                    .replace("[ ", "[").replace("   ", " ").replace("  ", " ").replace(" ", ",")
                )
                draw_border(frame,
                            (int(car_x1), int(car_y1)),
                            (int(car_x2), int(car_y2)),
                            (0, 255, 0), 25,
                            line_length_x=200, line_length_y=200)


                x1, y1, x2, y2 = ast.literal_eval(
                    df_.iloc[row_indx]["license_plate_bbox"]
                    .replace("[ ", "[").replace("   ", " ").replace("  ", " ").replace(" ", ",")
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)


                car_id = df_.iloc[row_indx]["car_id"]
                license_crop = license_plate[car_id]["license_crop"]
                H, W, _ = license_crop.shape

                try:
                    frame[int(car_y1) - H - 100:int(car_y1) - 100,
                    int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                    frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                    int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[car_id]["license_plate_number"],
                        cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17
                    )
                    cv2.putText(
                        frame,
                        license_plate[car_id]["license_plate_number"],
                        (int((car_x2 + car_x1 - text_width) / 2),
                         int(car_y1 - H - 250 + (text_height / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3, (0, 0, 0), 17
                    )
                except:
                    pass

            out.write(frame)
            progress2.progress(
                min((fn + 1) / total_frames, 1.0),
                text=f"Rendering frame {fn + 1} / {total_frames}"
            )

        cap2.release()
        out.release()
        progress2.empty()

        st.success("✅ Done!")
        st.video(out_path)


        os.unlink(video_path)