import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import whisper
from sklearn.preprocessing import StandardScaler
import cv2
import time
from typing import List, Dict, Tuple, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


class VoiceItemLogger(nn.Module):
    """AI model for voice-controlled item logging with NLP"""

    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256):
        super(VoiceItemLogger, self).__init__()

        # Load Whisper for speech-to-text
        self.whisper_model = whisper.load_model("base")

        # NLP model for intent classification
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)

        # Classification heads
        self.intent_classifier = nn.Linear(
            hidden_dim * 2, 5
        )  # add, remove, query, list, clear
        self.item_extractor = nn.Linear(
            hidden_dim * 2, vocab_size
        )  # item name extraction
        self.quantity_extractor = nn.Linear(hidden_dim * 2, 20)  # quantity (1-20)

        # Item categories
        self.categories = {
            "clothing": [
                "shirt",
                "pants",
                "dress",
                "jacket",
                "shoes",
                "socks",
                "underwear",
            ],
            "electronics": [
                "laptop",
                "phone",
                "charger",
                "headphones",
                "camera",
                "tablet",
            ],
            "toiletries": ["toothbrush", "shampoo", "soap", "deodorant", "perfume"],
            "documents": ["passport", "ticket", "license", "insurance", "visa"],
            "accessories": ["watch", "jewelry", "belt", "hat", "sunglasses"],
        }

    def speech_to_text(self, audio_data):
        """Convert speech to text using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_data)
            return result["text"].lower().strip()
        except Exception as e:
            logging.error(f"Speech recognition error: {e}")
            return ""

    def forward(self, text_tokens):
        """Process tokenized text for intent and item extraction"""
        embedded = self.embedding(text_tokens)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)

        # Use last hidden state for classification
        last_hidden = lstm_out[:, -1, :]

        intent = F.softmax(self.intent_classifier(last_hidden), dim=1)
        item_probs = F.softmax(self.item_extractor(last_hidden), dim=1)
        quantity = F.softmax(self.quantity_extractor(last_hidden), dim=1)

        return intent, item_probs, quantity

    def process_voice_command(self, audio_data):
        """Complete pipeline for processing voice commands"""
        text = self.speech_to_text(audio_data)
        if not text:
            return {"error": "Could not understand audio"}

        # Simple NLP processing (in production, use proper tokenization)
        intent = self.classify_intent(text)
        item_info = self.extract_item_info(text)

        return {
            "text": text,
            "intent": intent,
            "item": item_info["item"],
            "quantity": item_info["quantity"],
            "category": item_info["category"],
        }

    def classify_intent(self, text):
        """Classify user intent from text"""
        text = text.lower()
        if any(word in text for word in ["add", "pack", "put", "include"]):
            return "add"
        elif any(word in text for word in ["remove", "take out", "delete"]):
            return "remove"
        elif any(word in text for word in ["find", "where", "search", "look for"]):
            return "query"
        elif any(word in text for word in ["list", "show", "what", "items"]):
            return "list"
        elif any(word in text for word in ["clear", "empty", "reset"]):
            return "clear"
        else:
            return "unknown"

    def extract_item_info(self, text):
        """Extract item name, quantity, and category from text"""
        text = text.lower()

        # Extract quantity
        quantity = 1
        for i in range(1, 21):
            if str(i) in text or self.number_to_word(i) in text:
                quantity = i
                break

        # Extract item and category
        item = ""
        category = "other"

        for cat, items in self.categories.items():
            for item_name in items:
                if item_name in text:
                    item = item_name
                    category = cat
                    break
            if item:
                break

        # If no predefined item found, extract the most likely noun
        if not item:
            words = text.split()
            # Simple heuristic: look for nouns (in production, use proper NLP)
            for word in words:
                if len(word) > 3 and word not in ["pack", "add", "put", "include"]:
                    item = word
                    break

        return {"item": item, "quantity": quantity, "category": category}

    def number_to_word(self, n):
        """Convert number to word for text matching"""
        numbers = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        return numbers.get(n, str(n))


class TheftDetectionAI(nn.Module):
    """AI model for theft detection using movement patterns and sensor data"""

    def __init__(self, input_features=10, hidden_dim=64):
        super(TheftDetectionAI, self).__init__()

        # Neural network for anomaly detection
        self.encoder = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_features),
        )

        # Classifier for theft probability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.scaler = StandardScaler()
        self.movement_threshold = 1.5  # meters
        self.time_window = 30  # seconds
        self.movement_history = []

    def forward(self, sensor_data):
        """Process sensor data for theft detection"""
        encoded = self.encoder(sensor_data)
        decoded = self.decoder(encoded)
        theft_prob = self.classifier(encoded)

        reconstruction_loss = F.mse_loss(decoded, sensor_data)
        return theft_prob, reconstruction_loss

    def process_sensor_data(
        self, gps_data, accelerometer_data, bluetooth_rssi, timestamp
    ):
        """Process real-time sensor data for theft detection"""
        # Calculate movement from GPS
        if len(self.movement_history) > 0:
            last_pos = self.movement_history[-1]["gps"]
            current_pos = gps_data
            distance = self.calculate_distance(last_pos, current_pos)
        else:
            distance = 0

        # Store movement history
        self.movement_history.append(
            {
                "gps": gps_data,
                "accelerometer": accelerometer_data,
                "bluetooth_rssi": bluetooth_rssi,
                "timestamp": timestamp,
                "distance": distance,
            }
        )

        # Keep only recent history
        cutoff_time = timestamp - self.time_window
        self.movement_history = [
            h for h in self.movement_history if h["timestamp"] > cutoff_time
        ]

        # Feature extraction
        features = self.extract_features()

        # Normalize features
        features_normalized = torch.FloatTensor(self.scaler.fit_transform([features]))[
            0
        ]

        # Predict theft probability
        with torch.no_grad():
            theft_prob, reconstruction_loss = self.forward(
                features_normalized.unsqueeze(0)
            )

        # Decision logic
        is_theft = self.detect_theft(theft_prob.item(), distance, bluetooth_rssi)

        return {
            "theft_probability": theft_prob.item(),
            "is_theft_detected": is_theft,
            "distance_moved": distance,
            "reconstruction_error": reconstruction_loss.item(),
        }

    def extract_features(self):
        """Extract features from movement history"""
        if len(self.movement_history) < 2:
            return [0] * 10

        recent_movements = self.movement_history[-10:]

        # Calculate statistics
        distances = [h["distance"] for h in recent_movements]
        rssi_values = [
            h["bluetooth_rssi"]
            for h in recent_movements
            if h["bluetooth_rssi"] is not None
        ]
        acc_magnitudes = [np.linalg.norm(h["accelerometer"]) for h in recent_movements]

        features = [
            np.mean(distances),  # avg distance
            np.std(distances),  # distance variance
            np.max(distances),  # max distance
            np.mean(rssi_values) if rssi_values else -100,  # avg RSSI
            np.std(rssi_values) if rssi_values else 0,  # RSSI variance
            np.mean(acc_magnitudes),  # avg acceleration
            np.std(acc_magnitudes),  # acceleration variance
            len(recent_movements),  # movement count
            sum(1 for d in distances if d > 0.1),  # significant movements
            sum(1 for r in rssi_values if r > -70)
            if rssi_values
            else 0,  # strong RSSI count
        ]

        return features

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two GPS coordinates"""
        lat1, lon1 = pos1
        lat2, lon2 = pos2

        # Haversine formula
        R = 6371000  # Earth's radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = (
            np.sin(delta_phi / 2) * 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) * 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def detect_theft(self, theft_prob, distance, bluetooth_rssi):
        """Determine if theft is occurring based on multiple factors"""
        # High probability from neural network
        if theft_prob > 0.7:
            return True

        # Distance threshold exceeded
        if distance > self.movement_threshold:
            return True

        # Bluetooth signal lost (owner too far)
        if bluetooth_rssi is None or bluetooth_rssi < -80:
            return True

        # Combination of factors
        if (
            theft_prob > 0.4
            and distance > 0.5
            and (bluetooth_rssi is None or bluetooth_rssi < -70)
        ):
            return True

        return False


class ObstacleDetectionAI(nn.Module):
    """AI model for obstacle detection and auto-braking"""

    def __init__(self):
        super(ObstacleDetectionAI, self).__init__()

        # Simple CNN for processing ultrasonic sensor data patterns
        self.obstacle_classifier = nn.Sequential(
            nn.Linear(8, 32),  # 8 ultrasonic sensors
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # no_obstacle, obstacle_ahead, obstacle_side
        )

        # Safety parameters
        self.safe_distance = 0.5  # meters
        self.warning_distance = 1.0  # meters
        self.emergency_stop_distance = 0.2  # meters

    def forward(self, ultrasonic_data):
        """Process ultrasonic sensor data"""
        obstacle_type = self.obstacle_classifier(ultrasonic_data)
        return F.softmax(obstacle_type, dim=1)

    def process_sensors(self, ultrasonic_readings, current_speed):
        """Process sensor readings and determine action"""
        # Convert to tensor
        sensor_tensor = torch.FloatTensor(ultrasonic_readings).unsqueeze(0)

        # Get obstacle classification
        with torch.no_grad():
            obstacle_probs = self.forward(sensor_tensor)

        # Find minimum distance and direction
        min_distance = min(ultrasonic_readings)
        min_index = ultrasonic_readings.index(min_distance)

        # Determine obstacle direction
        if min_index in [0, 1, 7]:  # front sensors
            direction = "front"
        elif min_index in [2, 3]:  # right sensors
            direction = "right"
        elif min_index in [4, 5]:  # back sensors
            direction = "back"
        else:  # left sensors
            direction = "left"

        # Determine action
        action = self.determine_action(min_distance, direction, current_speed)

        return {
            "min_distance": min_distance,
            "obstacle_direction": direction,
            "obstacle_probabilities": obstacle_probs.squeeze().tolist(),
            "action": action,
            "safe_to_move": min_distance > self.safe_distance,
        }

    def determine_action(self, distance, direction, speed):
        """Determine the appropriate action based on obstacle detection"""
        if distance < self.emergency_stop_distance:
            return "emergency_stop"
        elif distance < self.safe_distance:
            if direction == "front":
                return "stop"
            elif direction in ["right", "left"]:
                return "slow_down"
            else:
                return "continue"
        elif distance < self.warning_distance:
            return "caution"
        else:
            return "continue"


class FollowMeAI(nn.Module):
    """AI model for autonomous following using computer vision and BLE"""

    def __init__(self):
        super(FollowMeAI, self).__init__()

        # Simple person tracking network (in production, use YOLO or similar)
        self.person_detector = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),  # Assuming 224x224 input
            nn.ReLU(),
            nn.Linear(256, 4),  # x, y, width, height
        )

        # PID controller for smooth following
        self.pid_linear = PIDController(kp=1.0, ki=0.1, kd=0.05)
        self.pid_angular = PIDController(kp=2.0, ki=0.2, kd=0.1)

        self.target_distance = 2.0  # meters
        self.max_speed = 1.0  # m/s

    def forward(self, image):
        """Detect person in image"""
        return self.person_detector(image)

    def track_person(self, image, bluetooth_rssi):
        """Track person using computer vision and BLE signal strength"""
        # Preprocess image
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0

        # Detect person
        with torch.no_grad():
            bbox = self.forward(image_tensor)

        # Calculate person position relative to camera
        image_center_x = image.shape[1] // 2
        person_center_x = bbox[0] + bbox[2] // 2

        # Calculate angular error (person position relative to center)
        angular_error = (person_center_x - image_center_x) / image_center_x

        # Estimate distance using BLE RSSI (rough approximation)
        if bluetooth_rssi is not None:
            # RSSI to distance approximation (very rough)
            estimated_distance = max(0.1, 10 ** ((-69 - bluetooth_rssi) / 20))
        else:
            estimated_distance = self.target_distance

        distance_error = estimated_distance - self.target_distance

        linear_velocity = self.pid_linear.compute(distance_error)
        angular_velocity = self.pid_angular.compute(angular_error)

        linear_velocity = np.clip(linear_velocity, -self.max_speed, self.max_speed)
        angular_velocity = np.clip(angular_velocity, -2.0, 2.0)  # rad/s

        return {
            "person_detected": True,
            "person_bbox": bbox.tolist(),
            "estimated_distance": estimated_distance,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "following_active": abs(distance_error) < 0.5,
        }


class PIDController:
    """Simple PID controller for smooth following"""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        """Compute PID output"""
        self.integral += error
        derivative = error - self.prev_error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        return output


class SmartSuitcaseAI:
    """Main AI controller that integrates all models"""

    def __init__(self):
        self.voice_logger = VoiceItemLogger()
        self.theft_detector = TheftDetectionAI()
        self.obstacle_detector = ObstacleDetectionAI()
        self.follow_me = FollowMeAI()

        self.item_inventory = {}
        self.is_following = False
        self.current_location = None

    def process_voice_command(self, audio_data):
        """Process voice command for item logging"""
        return self.voice_logger.process_voice_command(audio_data)

    def update_inventory(self, command_result):
        """Update item inventory based on voice command"""
        if command_result.get("intent") == "add":
            item = command_result["item"]
            quantity = command_result["quantity"]
            if item in self.item_inventory:
                self.item_inventory[item] += quantity
            else:
                self.item_inventory[item] = quantity
        elif command_result.get("intent") == "remove":
            item = command_result["item"]
            quantity = command_result["quantity"]
            if item in self.item_inventory:
                self.item_inventory[item] = max(0, self.item_inventory[item] - quantity)
                if self.item_inventory[item] == 0:
                    del self.item_inventory[item]

        return self.item_inventory

    def check_security(self, gps_data, accelerometer_data, bluetooth_rssi, timestamp):
        """Check for theft and security issues"""
        return self.theft_detector.process_sensor_data(
            gps_data, accelerometer_data, bluetooth_rssi, timestamp
        )

    def check_obstacles(self, ultrasonic_readings, current_speed):
        """Check for obstacles and determine movement safety"""
        return self.obstacle_detector.process_sensors(
            ultrasonic_readings, current_speed
        )

    def follow_owner(self, camera_image, bluetooth_rssi):
        """Follow the owner using computer vision and BLE"""
        if self.is_following:
            return self.follow_me.track_person(camera_image, bluetooth_rssi)
        return None

    def get_system_status(self):
        """Get overall system status"""
        return {
            "inventory_count": len(self.item_inventory),
            "total_items": sum(self.item_inventory.values()),
            "following_mode": self.is_following,
            "current_location": self.current_location,
            "models_loaded": True,
        }



