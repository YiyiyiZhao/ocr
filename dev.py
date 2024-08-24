import argparse
import os
from datetime import datetime

import cv2
from cnocr import CnOcr
import pdb

def main(config):
    ocr = CnOcr(rec_model_name='densenet_lite_136-gru')
    video = cv2.VideoCapture(os.path.join("input", config['video_name']))
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vi_name=config['video_name'].split('.')[0]
    save_dir = f"output/{vi_name}_{config['threshold']}_{config['frame_interval']}_{current_time}/frames"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frame_count = 0
    number_list_final = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count %  config['frame_interval'] == config['frame_random']:
            try:
                out = ocr.ocr(frame)
                if out[0]["score"] >  config['threshold']:
                    number = int(out[0]["text"])
                    number_list_final.append(number)
                    frame_filename = f"{save_dir}/frame-{frame_count}.png"
                    cv2.imwrite(frame_filename, frame)
                    print(f"{number} and Frame saved: {frame_filename}")
            except:
                pass
        frame_count += 1
    video.release()
    print(number_list_final)
    processed_numbers = [x / 100 for x in number_list_final if x > config['min_value']]
    txt_filename = f"output/{vi_name}_{config['threshold']}_{config['frame_interval']}_{current_time}/numbers.txt"

    with open(txt_filename, 'w') as file:
        for number in processed_numbers:
            file.write(f"{number}\n")
    print(f"Data saved to {txt_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for getting values.")
    parser.add_argument('--video_name', type=str, default='file.MOV', help='Path to the video file')
    parser.add_argument('--threshold', type=float, default=0.85, help='Score threshold for OCR acceptance')
    parser.add_argument('--min_value', type=int, default=10000, help='Minimum value to consider for processing')
    parser.add_argument('--frame_interval', type=int, default=5, help='Interval of frames to process')
    parser.add_argument('--frame_random', type=int, default=1, help='Specific frame within the interval to process')
    args = parser.parse_args()

    # Create a configuration dictionary from parsed arguments
    config = {
        'video_name': args.video_name,
        'threshold': args.threshold,
        'min_value': args.min_value,
        'frame_interval': args.frame_interval,
        'frame_random': args.frame_random
    }

    main(config)
