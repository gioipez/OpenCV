from flask import Flask, request, jsonify
import redis
import json
import threading
import uuid
import numpy as np
from utils.opencvLogger import logger
from SegmentAnything.SegmentAnything import read_image, get_image_masks, setup_device, load_model
import os

app = Flask(__name__)

# Initialize Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

CALIBRATED_IMAGE_DIR = "/home/gio/OpenCV/SegmentAnything/images/calibrated"

def segment_image(image_path, boxes):
    """Segment the image and return masks and scores."""
    logger.info("Getting image")
    image = read_image(image_path=image_path)

    logger.info("Converting boxes string to np")
    boxes_ = np.array(boxes)

    logger.info(f"Reshaping the boxes type: {type(boxes_)} - {boxes_}")
    boxes_coor = boxes_.reshape(-1, 4)  # Make sure boxes are reshaped correctly

    logger.info("Getting mask and score")
    try:
        masks, scores = get_image_masks(predictor, image, boxes_coor)

        # Check the shape of masks and ensure it's in the expected format
        logger.info(f"Masks shape: {masks.shape}, Scores: {scores}")

        if masks.ndim == 3:  # Example: (num_masks, height, width)
            masks = masks.astype(np.uint8)
        else:
            logger.error(f"Unexpected masks shape {masks.ndim}")

    except Exception as e:
        logger.error(f"Error predicting masks: {e}")
        return [], []

    return masks, scores


def long_running_task(task):
    """Handles long-running segmentation tasks."""
    task_id = task['task_id']
    try:
        flower_info = task['data']
        logger.info(f"Processing flower: {flower_info}")
        flower_name = flower_info['flower_name']
        
        local_image_path = os.path.join(CALIBRATED_IMAGE_DIR, flower_name)
        boxes = retrieve_boxes_from_redis(flower_name)
        
        masks, scores = segment_image(local_image_path, boxes)

        logger.info("Saving results to Redis")
        redis_client.hset(task_id, mapping={
            "status": "Processed",
            "flower_name": flower_name,
            "boxes": json.dumps(json.loads(flower_info['boxes'])),
            "masks": masks.size if masks is not None else "[]",
            "scores": json.dumps(scores.tolist()) if scores is not None else "[]",
        })
        redis_client.expire(task_id, 86400)  # Set expiration for the task result
        logger.info(f"Task completed: {task_id}")
    except Exception as e:
        logger.error(f"Error in long_running_task: {e}")
        redis_client.hset(task_id, mapping={
            "status": "failed",
            "error": str(e)
        })
        redis_client.expire(task_id, 86400)

def retrieve_boxes_from_redis(flower_name):
    """Retrieve bounding boxes for the given flower from Redis."""
    key = f"flower:{flower_name}"
    boxes_data = redis_client.hgetall(key)

    if boxes_data:
        # Get the boxes string and parse it into a list
        boxes = json.loads(boxes_data[b'boxes'])

        # Flatten the boxes and reshape them into (num_boxes, 4)
        # Each box is a pair of points, so we need to convert the list of pairs into an array
        flattened_boxes = []
        for box in boxes:
            # Each box has two points: [[x1, y1], [x2, y2]]
            flattened_boxes.append(box[0][0])  # x1
            flattened_boxes.append(box[0][1])  # y1
            flattened_boxes.append(box[1][0])  # x2
            flattened_boxes.append(box[1][1])  # y2

        # Convert to a NumPy array and reshape
        boxes_array = np.array(flattened_boxes).reshape(-1, 4)
        return boxes_array.tolist()  # or just return boxes_array if you prefer NumPy format

    return []

def worker():
    """Continuously checks for tasks to process."""
    while True:
        _, task_data = redis_client.brpop("task_queue")
        task = json.loads(task_data)
        long_running_task(task)

# Start the worker thread
threading.Thread(target=worker, daemon=True).start()

@app.route('/process', methods=['POST'])
def process():
    """Endpoint to queue a new image segmentation task."""
    data = request.json
    task_id = str(uuid.uuid4())
    task_data = {'task_id': task_id, 'data': data}
    redis_client.rpush("task_queue", json.dumps(task_data))
    return jsonify({"task_id": task_id, "message": "Task has been queued"}), 202

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """Retrieve the result of a long-running task."""
    result = redis_client.hgetall(task_id)
    if result:
        return jsonify({k.decode('utf-8'): json.loads(v) for k, v in result.items()}), 200
    else:
        return jsonify({"task_id": task_id, "status": "Pending or not found"}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "Running"}), 200

if __name__ == '__main__':
    device = setup_device()
    predictor = load_model(device)
    app.run(debug=True)
