import argparse
import os
from datetime import datetime

import cv2
from cnocr import CnOcr
import pandas as pd

def main(config):
    ocr = CnOcr(rec_model_name='densenet_lite_136-gru')
    video = cv2.VideoCapture(os.path.join("input", config['video_name']))
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vi_name=config['video_name'].split('.')[0]

    fps = video.get(cv2.CAP_PROP_FPS)

    save_dir = f"output/{vi_name}_{config['threshold']}_{config['frame_interval']}_{current_time}/frames"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frame_count = 0
    number_list_final = []
    null_list_final=[]
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % int(round(fps)) ==config['frame_random']:
            null_list_final.append({"index":frame_count, "value":"NULL"})
            frame_filename = f"{save_dir}/frame-{frame_count}.png"
            cv2.imwrite(frame_filename, frame)
            print(f"NULL and Frame saved: {frame_filename}")

        if frame_count %  config['frame_interval'] == config['frame_random']:
            try:
                out = ocr.ocr(frame)
                if out[0]["score"] >  config['threshold']:
                    number = int(out[0]["text"])
                    if number > config['min_value']:
                        number_list_final.append({"index":frame_count, "value": number/100 })
                        frame_filename = f"{save_dir}/frame-{frame_count}.png"
                        cv2.imwrite(frame_filename, frame)
                        print(f"{number} and Frame saved: {frame_filename}")
            except:
                pass
        frame_count += 1
    video.release()

    df_null = pd.DataFrame(null_list_final)
    df_number = pd.DataFrame(number_list_final)
    # 设置index列为索引
    df_null.set_index('index', inplace=True)
    df_number.set_index('index', inplace=True)

    # 合并，df_number的值优先于df_null的值
    df_combined = df_number.combine_first(df_null)

    # 重置索引以便查看
    df_combined.reset_index(inplace=True)
    df_sorted = df_combined.sort_values(by='index')
    csv_filename = f"output/{vi_name}_{config['threshold']}_{config['frame_interval']}_{current_time}/numbers.csv"
    df_sorted.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for getting values.")
    parser.add_argument('--video_name', type=str, default='file.MOV', help='Path to the video file')
    parser.add_argument('--threshold', type=float, default=0.7, help='Score threshold for OCR acceptance')
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
