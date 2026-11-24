from datetime import datetime
from pathlib import Path

import cv2

from clock_visualization import ClockVisualization


CLOCKS_DIR = Path(__file__).resolve().parent / "clocks"
LOG_FILE = Path(__file__).resolve().with_name("clock_visualization.txt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def log_benchmark_start():
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(f"Benchmark started at {timestamp}\n")


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


def run_clock_images(images=None):
    log_benchmark_start()

    if images is None:
        images = load_clock_images()

    visualizer = ClockVisualization()
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

correct = 0

if __name__ == "__main__":
    clock_images = load_clock_images()
    print(f"Running {len(clock_images)} images from {CLOCKS_DIR}:")
    results = run_clock_images(clock_images)
    for name, result in results.items():
        short_result = result.replace(":","")
        short_result = short_result.strip("\n")
        if len(result) == 5 and name.split("-")[1] == short_result:
            print(f"{name}: {result}")
            correct +=1
        elif len(result) == 5 and name.split("-")[1] != short_result:
            print(f"{name}: {result} Incorrect time!")
        else:
            print(f"{name}: {result}")
    print(f"Correct times: {correct}/{len(results)} times")
    print(f"Processing logs saved to {LOG_FILE}")
