from datetime import datetime
from pathlib import Path

import cv2

from clock_visualization import ClockVisualization


CLOCKS_DIR = Path(__file__).resolve().parent / "clocks"
LOG_FILE = (
    Path(__file__).resolve().parent
    / "logs"
    / f"clock_visualization_{datetime.now().astimezone():%Y%m%d_%H%M%S}.txt"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

#Opening logging session with a timestamp of when session started
def log_benchmark_start():
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(f"Benchmark started at {timestamp}\n")

#Loading clock images and their paths to an array for easier benchmarking
#Correct image naming: clock(number)-(hours)-(minutes).jpg
def load_clock_images(folder=CLOCKS_DIR):
    # Load supported clock images, keyed by filename without the extension.
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Clocks folder does not exist: {folder}")

    image_paths = sorted(
        (
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower()
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {folder}")

    images = {}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(f"Could not read image file: {image_path}")
        images[image_path.stem] = image

    return images

#Running clock time finding on each photo in an array and logging every process behind it
def run_clock_images(images=None):
    log_benchmark_start()

    if images is None:
        images = load_clock_images()

    visualizer = ClockVisualization(LOG_FILE)
    results = {}

    for name, image in images.items():
        print(f"Processing photo: {name}")
        try:
            results[name] = visualizer.run(image, image_name=name)
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            visualizer.log(f"Processing failed: {error}")
            results[name] = f"Error: {error}"
        visualizer.log(f"Result: {results[name]}")

    return results

#Points calculation based on how far clock time finding was from the answer
def clock_points(delta):
    if delta <= 2:
        return 1.0
    elif delta <= 10:
        return 0.8
    elif delta <= 30:
        return 0.5
    else:
        return 0

#Calculating difference between correct time and time given by clock time finding in minutes
def clock_delta_minutes(correct_hours, correct_minutes, received_hours, received_minutes):
    correct_time = (int(correct_hours) % 12) * 60 + int(correct_minutes)
    received_time = (int(received_hours) % 12) * 60 + int(received_minutes)
    difference = abs(correct_time - received_time)
    return min(difference, 12 * 60 - difference)

detected = 0    
correct = 0
points = 0

if __name__ == "__main__":
    clock_images = load_clock_images()
    print(f"Running {len(clock_images)} images from {CLOCKS_DIR}:")
    results = run_clock_images(clock_images)
    for name, result in results.items():
        delta = 0
        if len(result) == 5:
            detected +=1
            received_hours = result.split(":")[0]
            received_minutes = result.split(":")[1]
            correct_hours = name.split("-")[1]
            correct_minutes = name.split("-")[2]
            delta = clock_delta_minutes(
                correct_hours,
                correct_minutes,
                received_hours,
                received_minutes,
            )
            if delta == 0:
                correct +=1
                print(f"{name}: {result} -> Correct time! (points = 1)")
            else:
                print(f"{name}: {result} -> Incorrect time! (delta = {delta} minute(s); points = {clock_points(delta)})")
        else:
            print(f"{name}: {result}")
        points += clock_points(delta)

    print(f"Detected clocks: {detected}/{len(results)} ({detected/len(results)*100:.2f}%)")
    print(f"Correct times: {correct}/{len(results)} ({correct/len(results)*100:.2f}%)")
    print(f"Total points: {points}/{len(results)} ({points/len(results)*100:.2f}%)")
    print(f"Processing logs saved to {LOG_FILE}")
