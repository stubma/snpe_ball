import time
import cv2
import onnxruntime
import os
import numpy as np


def onnx_infer_30(video_path, onnx_path1, frame_w, frame_h, goalnet_points):
    # 1. 30帧图片
    video_imgs = [os.path.join(video_path, file) for file in
                  os.listdir(video_path) if file.endswith(('.jpg', '.png', '.bmp'))]
    video_imgs = sorted(video_imgs, key=lambda x: int(os.path.basename(x).split('.')[0]))
    video_len = len(video_imgs)
    assert video_len==30, print("Video_len should be 30 frames: ", len(video_imgs))

    # 2. 30帧图片读取，裁剪，normalize
    frames = np.zeros((video_len, 3, frame_h, frame_w), dtype=np.float32)
    cx1, cy1 = (goalnet_points[2] + goalnet_points[3]) / 2
    cx1, cy1 = int(cx1), int(cy1)
    for idx, img_path in enumerate(video_imgs):
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        lx = min(max(cx1 - frame_w // 2, 0), w - frame_w)
        ly = min(max(cy1 - frame_h // 2, 0), h - frame_h)
        crop_img = image[ly:ly + frame_h, lx:lx + frame_w].copy()  # 根据球门裁剪图片
        crop_img = crop_img.astype(np.float32) / 255.0  # normalize
        input_frame = np.transpose(crop_img, (2, 0, 1))
        frames[idx] = input_frame

    # 3. 模型推理
    t1 = time.time()
    session1 = onnxruntime.InferenceSession(onnx_path1, providers=['CPUExecutionProvider'])  # CPUExecutionProvider
    inputs1 = {session1.get_inputs()[0].name: frames}
    output2 = session1.run(None, inputs1)   # 模型推理
    print(f'infer time: {time.time() - t1}')
    print("infer result: ", output2[0])
    print("ball confidence: ", output2[0][0, 1])


if __name__ == '__main__':
    input_width = 640
    input_height = 384
    onnx_path1 = f'./ballspotting_woGSM_part1.onnx'
    video_path = r"./frames"
    goalnet_points = np.array([[1167, 438], [1386, 426], [1168, 543], [1381, 521]])  #左边球门四个角点像素坐标
    onnx_infer_30(video_path, onnx_path1, input_width, input_height, goalnet_points)
