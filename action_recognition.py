import torch
import torchvision
import cv2
import argparse
import time
import numpy as np
from tqdm import tqdm
import albumentations as A



                
def load_video(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError('Error while trying to read video. Please check path again')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    return cap, frame_width, frame_height

def setup_output_video_writer(input_video, frame_width, frame_height):
    save_name = input_video.split('/')[-1].split('.')[0]
    out = cv2.VideoWriter(
        f"outputs/{save_name}.avi", 
        cv2.VideoWriter_fourcc(*'XVID'), 
        30, 
        (frame_width, frame_height), 
        True
    )
    return out

def load_model(device):
    model = torchvision.models.video.r2plus1d_18(
        weights=torchvision.models.video.resnet.R2Plus1D_18_Weights.KINETICS400_V1, 
        progress=True
    )
    return model.eval().to(device)

def make_prediction(model, clips, device):
    with torch.no_grad():
        input_frames = np.array(clips)
        input_frames = np.expand_dims(input_frames, axis=0)
        input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
        input_frames = torch.tensor(input_frames, dtype=torch.float32).to(device)
        outputs = model(input_frames)
        _, preds = torch.max(outputs.data, 1)
    return preds

# define the transforms
transform = A.Compose([
    A.Resize(128, 171, always_apply=True),
    A.CenterCrop(112, 112, always_apply=True),
    A.Normalize(mean = [0.43216, 0.394666, 0.37645],
                std = [0.22803, 0.22145, 0.216989], 
                always_apply=True)
])



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_video = "input/joyce_DS.mp4"
    clip_len = 16
    print(f"Number of frames to consider for each prediction: {clip_len}")

    cap, frame_width, frame_height = load_video(input_video)
    out = setup_output_video_writer(input_video, frame_width, frame_height)
    model = load_model(device)

    frame_count = 0
    clips = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for _ in tqdm(range(total_frames), desc="Processing video", ncols=100):
        ret, frame = cap.read()
        if ret:
            image = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(image=frame)['image']
            clips.append(frame)
            if len(clips) == clip_len:
                preds = make_prediction(model, clips, device)
                label = "punchingperson(boxing)" if preds == 259 else "other"  #259 being the index of boxing activity

                frame_count += 1
                cv2.putText(image, label, (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2, 
                            lineType=cv2.LINE_AA)
                clips.pop(0)
                out.write(image)
        else:
            break

    print("\nDone processing video")
    cap.release()
    out.release()



if __name__ == "__main__":
    main()

