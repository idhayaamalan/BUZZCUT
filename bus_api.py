# bus_api.py
# Flask API for bus pipeline: IN/OUT counting + ETA prediction + retraining

import os
import math
import json
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
        "supports_credentials": True
    }
})

# Alternative: Enable CORS for specific origins only (more secure for production)
# CORS(app, resources={
#     r"/*": {
#         "origins": ["http://localhost:3000", "http://localhost:8080", "https://yourdomain.com"],
#         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#         "allow_headers": ["Content-Type", "Authorization"],
#     }
# })

# ==============================
# Shared config
# ==============================
MODEL_PATH = "bus_eta_linear_model.pkl"
ENC_PATH = "label_encoders.pkl"

CROWD_MODEL_PATH = "bus_crowd_model.pkl"
CROWD_ENC_PATH = "crowd_label_encoders.pkl"

ETA_FEATURES = [
    "distance_km", "hour", "day_of_week", "is_weekend", "is_peak_hour",
    "current_passengers", "bus_capacity", "stops_remaining", "avg_stop_time",
    "traffic_multiplier", "weather_delay_factor", "buses_per_hour_at_stop",
    "route_type_encoded", "origin_encoded", "destination_encoded", "weather_encoded",
    "passengers_per_km", "capacity_utilization", "total_stop_time"
]

CROWD_FEATURES = [
    "distance_km", "hour", "day_of_week", "is_weekend", "is_peak_hour",
    "stops_remaining", "avg_stop_time", "traffic_multiplier",
    "weather_delay_factor", "buses_per_hour_at_stop",
    "route_type_encoded", "origin_encoded", "destination_encoded", "weather_encoded",
    "passengers_per_km", "total_stop_time"
]

CAT_COLS = ["route_type", "origin", "destination", "weather"]

# ==============================
# Utility: ETA helpers
# ==============================
def _load_or_init_model_and_encoders():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENC_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENC_PATH)
    else:
        model = LinearRegression()
        encoders = {}
    return model, encoders

def _load_or_init_crowd_model_and_encoders():
    if os.path.exists(CROWD_MODEL_PATH) and os.path.exists(CROWD_ENC_PATH):
        model = joblib.load(CROWD_MODEL_PATH)
        encoders = joblib.load(CROWD_ENC_PATH)
    else:
        model = LinearRegression()
        encoders = {}
    return model, encoders

def _ensure_encoders(encoders, df):
    for col in CAT_COLS:
        if col not in encoders:
            le = LabelEncoder()
            encoders[col] = le.fit(df[col].astype(str).fillna("UNKNOWN"))
        le = encoders[col]
        known = set(le.classes_.tolist())
        def trans(val):
            s = str(val) if pd.notna(val) else "UNKNOWN"
            if s in known:
                return le.transform([s])[0]
            else:
                return -1
        df[col + "_encoded"] = df[col].apply(trans)
    return encoders, df

def _add_derived_features(df):
    df = df.copy()
    df["passengers_per_km"] = df["current_passengers"] / df["distance_km"].replace(0, np.nan)
    df["passengers_per_km"] = df["passengers_per_km"].fillna(0.0)
    df["capacity_utilization"] = df["current_passengers"] / df["bus_capacity"].replace(0, np.nan)
    df["capacity_utilization"] = df["capacity_utilization"].fillna(0.0)
    df["total_stop_time"] = df["stops_remaining"] * df["avg_stop_time"]
    return df

def _predict_eta_from_sample_dict(sample, model, encoders):
    df = pd.DataFrame([sample])
    encoders, df = _ensure_encoders(encoders, df)
    df = _add_derived_features(df)
    X = pd.DataFrame([df.iloc[0][ETA_FEATURES]])
    pred = float(model.predict(X)[0])
    return round(pred, 2)

def _predict_crowd_from_sample_dict(sample, model, encoders):
    df = pd.DataFrame([sample])
    encoders, df = _ensure_encoders(encoders, df)
    df = _add_derived_features(df)
    # crowd percentage = passengers / capacity * 100
    if df["bus_capacity"].iloc[0] > 0:
        df["crowd_percentage"] = (df["current_passengers"] / df["bus_capacity"]) * 100
    else:
        df["crowd_percentage"] = 0
    X = pd.DataFrame([df.iloc[0][CROWD_FEATURES]])
    pred = float(model.predict(X)[0])
    return round(pred, 2)

# ==============================
# Utility: IN/OUT Counter (YOLOv8)
# ==============================
def signed_distance_to_line(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    vx, vy = bx - ax, by - ay
    nx, ny = -vy, vx
    apx, apy = px - ax, py - ay
    return apx * nx + apy * ny

def run_counter(video_path, save_path="annotated.mp4", log_path="log.csv",
                line=((100, 100), (400, 100)), model_name="yolov8s.pt",
                conf=0.35, swap=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    import csv
    log_f = open(log_path, "w", newline="")
    log_writer = csv.writer(log_f)
    log_writer.writerow(["timestamp", "frame_index", "delta_in", "delta_out", "cum_in", "cum_out"])

    model = YOLO(model_name)

    last_side = {}
    last_centroid = {}
    last_cross_frame = defaultdict(lambda: -9999)
    cooldown = int(fps * 0.3)
    min_move = 8
    eps = 3
    in_count, out_count, frame_index = 0, 0, 0

    a, b = line
    results_gen = model.track(source=video_path, stream=True, persist=True,
                              verbose=False, conf=conf, iou=0.5, classes=[0])

    for res in results_gen:
        frame = res.orig_img.copy()
        delta_in, delta_out = 0, 0
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                if clss[i] != 0:
                    continue
                x1, y1, x2, y2 = xyxy[i].astype(int)
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                tid = int(ids[i]) if ids is not None and i < len(ids) else None
                if tid is None: continue
                side = signed_distance_to_line((cx, cy), a, b)
                if tid not in last_side:
                    last_side[tid] = side
                    last_centroid[tid] = (cx, cy)
                    continue
                prev_side = last_side[tid]
                pcx, pcy = last_centroid[tid]
                move_dist = math.hypot(cx-pcx, cy-pcy)
                crossed = False
                now_pos, prev_pos = side >= eps, prev_side >= eps
                now_neg, prev_neg = side <= -eps, prev_side <= -eps
                crossing_dir = None
                if prev_neg and now_pos: crossed, crossing_dir = True, "neg_to_pos"
                elif prev_pos and now_neg: crossed, crossing_dir = True, "pos_to_neg"
                if crossed and move_dist>=min_move and (frame_index-last_cross_frame[tid]>cooldown):
                    if swap:
                        if crossing_dir=="neg_to_pos": out_count+=1; delta_out+=1
                        elif crossing_dir=="pos_to_neg": in_count+=1; delta_in+=1
                    else:
                        if crossing_dir=="neg_to_pos": in_count+=1; delta_in+=1
                        elif crossing_dir=="pos_to_neg": out_count+=1; delta_out+=1
                    last_cross_frame[tid] = frame_index
                last_side[tid] = side
                last_centroid[tid] = (cx, cy)
        writer.write(frame)
        log_writer.writerow([datetime.now().isoformat(), frame_index, delta_in, delta_out, in_count, out_count])
        frame_index += 1

    writer.release()
    log_f.close()
    return {"in": in_count, "out": out_count, "video": save_path, "log": log_path}

# ==============================
# Error Handlers
# ==============================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request"}), 400

# ==============================
# Flask Endpoints
# ==============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Bus API Server",
        "version": "1.0.0",
        "endpoints": {
            "/start": "GET - Health check",
            "/predict": "POST - Predict ETA",
            "/predict_crowd": "POST - Predict crowd percentage",
            "/count": "POST - Count people in/out from video",
            "/retrain": "POST - Retrain ETA model",
            "/retrain_crowd": "POST - Retrain crowd model"
        }
    })

@app.route("/start", methods=["GET", "POST"])
def start():
    return jsonify({
        'message': 'Connected!',
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight"}), 200
    
    try:
        sample = request.get_json(force=True)
        if not sample:
            return jsonify({"error": "No JSON data provided"}), 400
            
        model, encoders = _load_or_init_model_and_encoders()
        if not hasattr(model, "predict"):
            return jsonify({"error": "Model not trained yet. Use /retrain first"}), 400
            
        eta = _predict_eta_from_sample_dict(sample, model, encoders)
        return jsonify({
            "eta_minutes": eta,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/retrain", methods=["POST", "OPTIONS"])
def retrain():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight"}), 200
    
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        new_csv = data.get("new_csv")
        old_csv = data.get("old_csv")
        test_size = float(data.get("test_size", 0.2))
        
        if not new_csv:
            return jsonify({"error": "new_csv parameter is required"}), 400
            
        if not os.path.exists(new_csv):
            return jsonify({"error": f"new_csv not found: {new_csv}"}), 400
            
        new_df = pd.read_csv(new_csv)
        if old_csv and os.path.exists(old_csv):
            full_df = pd.concat([pd.read_csv(old_csv), new_df], ignore_index=True)
        else:
            full_df = new_df
            
        encoders = {}
        encoders, full_df = _ensure_encoders(encoders, full_df)
        full_df = _add_derived_features(full_df)
        
        X, y = full_df[ETA_FEATURES], full_df["eta_minutes"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENC_PATH)
        
        return jsonify({
            "mae": mae,
            "rmse": rmse,
            "model_path": MODEL_PATH,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"Retraining failed: {str(e)}"}), 500

@app.route("/predict_crowd", methods=["POST", "OPTIONS"])
def predict_crowd():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight"}), 200
    
    try:
        sample = request.get_json(force=True)
        if not sample:
            return jsonify({"error": "No JSON data provided"}), 400
            
        model, encoders = _load_or_init_crowd_model_and_encoders()
        if not hasattr(model, "predict"):
            return jsonify({"error": "Crowd model not trained yet. Use /retrain_crowd first"}), 400
            
        crowd = _predict_crowd_from_sample_dict(sample, model, encoders)
        return jsonify({
            "crowd_percentage": crowd,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"Crowd prediction failed: {str(e)}"}), 500

@app.route("/retrain_crowd", methods=["POST", "OPTIONS"])
def retrain_crowd():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight"}), 200
    
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        new_csv = data.get("new_csv")
        old_csv = data.get("old_csv")
        test_size = float(data.get("test_size", 0.2))
        
        if not new_csv:
            return jsonify({"error": "new_csv parameter is required"}), 400
            
        if not os.path.exists(new_csv):
            return jsonify({"error": f"new_csv not found: {new_csv}"}), 400
            
        new_df = pd.read_csv(new_csv)
        if old_csv and os.path.exists(old_csv):
            full_df = pd.concat([pd.read_csv(old_csv), new_df], ignore_index=True)
        else:
            full_df = new_df
            
        encoders = {}
        encoders, full_df = _ensure_encoders(encoders, full_df)
        full_df = _add_derived_features(full_df)
        full_df["crowd_percentage"] = (full_df["current_passengers"] / full_df["bus_capacity"]) * 100
        
        X, y = full_df[CROWD_FEATURES], full_df["crowd_percentage"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        joblib.dump(model, CROWD_MODEL_PATH)
        joblib.dump(encoders, CROWD_ENC_PATH)
        
        return jsonify({
            "mae": mae,
            "rmse": rmse,
            "model_path": CROWD_MODEL_PATH,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": f"Crowd model retraining failed: {str(e)}"}), 500

@app.route("/count", methods=["POST", "OPTIONS"])
def count():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight"}), 200
    
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        video_path = data.get("video_path")
        if not video_path:
            return jsonify({"error": "video_path parameter is required"}), 400
            
        if not os.path.exists(video_path):
            return jsonify({"error": f"Video file not found: {video_path}"}), 400
            
        save_path = data.get("save_path", "annotated.mp4")
        log_path = data.get("log_path", "log.csv")
        line = tuple(map(tuple, data.get("line", [(100,100),(400,100)])))
        conf = float(data.get("conf", 0.35))
        swap = bool(data.get("swap", False))
        
        result = run_counter(
            video_path, 
            save_path=save_path, 
            log_path=log_path, 
            line=line,
            conf=conf,
            swap=swap
        )
        
        result.update({
            "status": "success",
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Counting failed: {str(e)}"}), 500

# ==============================
# Additional Utility Endpoints
# ==============================
@app.route("/status", methods=["GET"])
def status():
    eta_model_exists = os.path.exists(MODEL_PATH)
    crowd_model_exists = os.path.exists(CROWD_MODEL_PATH)
    
    return jsonify({
        "api_status": "running",
        "models": {
            "eta_model": {
                "exists": eta_model_exists,
                "path": MODEL_PATH
            },
            "crowd_model": {
                "exists": crowd_model_exists,
                "path": CROWD_MODEL_PATH
            }
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }), 200

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    print("Starting Bus API Server...")
    print("CORS enabled for all origins")
    print("Available endpoints:")
    print("  GET  /          - API information")
    print("  GET  /start     - Health check")
    print("  GET  /status    - Model status")
    print("  GET  /health    - Health check")
    print("  POST /predict   - Predict ETA")
    print("  POST /predict_crowd - Predict crowd percentage")
    print("  POST /count     - Count people in/out from video")
    print("  POST /retrain   - Retrain ETA model")
    print("  POST /retrain_crowd - Retrain crowd model")
    print("\nServer running on http://0.0.0.0:5000")
    
    app.run(host="0.0.0.0", port=5000, debug=True)