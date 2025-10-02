
import cv2
import numpy as np
from ultralytics import YOLO

class FootballTeamClassifier:
    """Post-processor for team classification based on jersey colors"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.team_colors = {
            'team_a': [(0, 0, 100), (50, 50, 255)],      # Red range
            'team_b': [(100, 0, 0), (255, 50, 50)],      # Blue range
            'referee': [(0, 100, 100), (50, 255, 255)]   # Yellow range
        }
    
    def classify_team_by_color(self, image, bbox):
        """Classify team based on jersey color in bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Extract jersey region (upper part of person)
        jersey_height = int((y2 - y1) * 0.4)  # Top 40% of person
        jersey_region = image[y1:y1+jersey_height, x1:x2]
        
        if jersey_region.size == 0:
            return 'unknown'
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        
        # Calculate dominant color
        pixels = hsv.reshape(-1, 3)
        dominant_color = np.mean(pixels, axis=0)
        
        # Classify based on color ranges
        h, s, v = dominant_color
        
        if 0 <= h <= 10 or 170 <= h <= 180:  # Red range
            return 'team_a'
        elif 100 <= h <= 130:  # Blue range
            return 'team_b'
        elif 20 <= h <= 30:  # Yellow range
            return 'referee'
        else:
            return 'unknown'
    
    def process_detection(self, image, detections):
        """Process YOLO detections and add team classification"""
        results = []
        
        for detection in detections:
            if detection.cls == 0:  # Person detected
                bbox = detection.xyxy[0].cpu().numpy()
                team = self.classify_team_by_color(image, bbox)
                
                # Create enhanced detection
                enhanced_detection = {
                    'class': 'person',
                    'team': team,
                    'bbox': bbox,
                    'confidence': detection.conf
                }
                results.append(enhanced_detection)
            else:
                # Non-person detections (ball, goalpost)
                results.append({
                    'class': ['person', 'ball', 'goalpost', 'field'][int(detection.cls)],
                    'team': 'none',
                    'bbox': detection.xyxy[0].cpu().numpy(),
                    'confidence': detection.conf
                })
        
        return results

# Usage example
def test_practical_model():
    classifier = FootballTeamClassifier('models/godseye_practical_model.pt')
    
    # Test on video
    cap = cv2.VideoCapture('path/to/video.mp4')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = classifier.model(frame)
        
        # Process detections
        enhanced_results = classifier.process_detection(frame, results[0].boxes)
        
        # Draw results
        for detection in enhanced_results:
            bbox = detection['bbox']
            label = f"{detection['class']} ({detection['team']})"
            confidence = detection['confidence']
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", 
                       (int(bbox[0]), int(bbox[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Practical Football Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_practical_model()
