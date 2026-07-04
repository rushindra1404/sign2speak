import os
import threading
import time
from flask import Flask, jsonify, request, render_template
import numpy as np

from tensorflow import keras
import itertools
import copy
import string

from flask_cors import CORS

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Load the saved model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
    model = keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Map model output indices to '1'-'9' and 'A'-'Z'
alphabet = ['1','2','3','4','5','6','7','8','9']
alphabet += list(string.ascii_uppercase)

feedback_log = []

def calc_landmark_list_from_json(landmarks_json, image_width, image_height):
    landmark_point = []
    # Original backend pipeline used cv2.flip(image, 1) to mirror the raw webcam feed BEFORE 
    # MediaPipe extraction. Because JS MediaPipe extracts from the unmirrored raw <video> feed,
    # we MUST mathematically flip the X coordinates here so that the ML Model receives 
    # the exact mirrored shape it was trained on!
    for lm in landmarks_json:
        flipped_x = 1.0 - lm.get('x', 0)
        landmark_x = min(int(flipped_x * image_width), image_width - 1)
        landmark_y = min(int(lm.get('y', 0) * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value > 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

@app.route('/health')
def health():
    """Simple health check for frontend reconnection logic."""
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/app')
def detection():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for receiving client-side MediaPipe landmarks and running model inference."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Accept 'landmarks' as requested by the user, but fallback to 'hands' for backward compatibility
        hands_json = data.get('landmarks', data.get('hands', []))
        
        if not hands_json:
            return jsonify({'prediction': "", 'confidence': 0, 'hand_detected': False})
        image_width = data.get('width', 640)
        image_height = data.get('height', 480)

        best_label = "?"
        best_confidence = 0.0
        best_top3 = []
        hand_detected = True if len(hands_json) > 0 else False

        for landmarks_json in hands_json:
            # Ensure we have 21 landmarks for a single hand
            if len(landmarks_json) != 21:
                continue

            # Pre-process JSON landmarks identically to how OpenCV+MediaPipe did it
            landmark_list = calc_landmark_list_from_json(landmarks_json, image_width, image_height)
            pre_processed = pre_process_landmark(landmark_list)

            input_data = np.array(pre_processed, dtype=np.float32).reshape(1, -1)
            predictions = model.predict_on_batch(input_data)

            idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][idx])

            if idx < len(alphabet) and confidence > best_confidence:
                best_confidence = confidence
                best_label = alphabet[idx]
                probs = predictions[0]
                top3_idx = np.argsort(probs)[::-1][:3]
                best_top3 = [
                    {'label': alphabet[i], 'confidence': float(probs[i])}
                    for i in top3_idx if i < len(alphabet)
                ]

        return jsonify({
            'prediction': best_label if hand_detected else "",
            'confidence': best_confidence,
            'top_predictions': best_top3,
            'hand_detected': hand_detected
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Log user feedback on predictions for future model improvement."""
    global feedback_log
    data = request.get_json()
    if data:
        data['timestamp'] = time.time()
        feedback_log.append(data)
    return jsonify({'status': 'ok', 'total': len(feedback_log)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("SIGN2SPEAK SERVER STARTING...")
    print("URL: http://127.0.0.1:5000")
    print("Reloader disabled to prevent unsolicited restarts.")
    print("="*50 + "\n")
    
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000, use_reloader=False)
