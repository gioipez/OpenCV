import json
import redis

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

def load_json_to_redis(file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Store each entry in Redis
    for item in data:
        flower_name = item['flower_name']
        boxes = item['boxes']

        # Store in Redis with flower_name as the key
        redis_client.hset(f"flower:{flower_name}", mapping={
            "flower_name": flower_name,
            "boxes": json.dumps(boxes)  # Convert boxes to JSON string for storage
        })

    print("Data successfully written to Redis.")

if __name__ == "__main__":
    json_file_path = 'flowers_with_boxes.json'  # Replace with your JSON file path
    load_json_to_redis(json_file_path)
