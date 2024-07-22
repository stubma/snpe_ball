import time
import cv2
import onnxruntime
import os
import numpy as np


class GoalNetModel(object):

    def __init__(self, onnx_path, input_width, input_height):
        super(GoalNetModel, self).__init__()
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_width = input_width
        self.input_height = input_height

    def infer(self, image):
        h, w = image.shape[:2]

        # 1. 图片进行前处理resize， normalize
        scale = min(self.input_height / h, self.input_width / w)
        i2d = np.array([
            [scale, 0, (-scale * w + self.input_width + scale - 1) * 0.5],
            [0, scale, (-scale * h + self.input_height + scale - 1) * 0.5]
        ], dtype=np.float32)
        d2i = cv2.invertAffineTransform(i2d)
        input_img = cv2.warpAffine(image, i2d, (self.input_width, self.input_height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = input_img.astype(np.float32) / 255.0

        # 2. 模型推理
        inputs = {self.session.get_inputs()[0].name: input_img[None]}
        output = self.session.run(None, inputs)

        # 3. 后处理解码球门坐标点
        heatmap = output[0]
        _, c, bh, bw = heatmap.shape
        pred_index = np.argmax(heatmap.reshape(c, bh*bw), axis=1)
        pred_points = np.concatenate([(pred_index % bw).reshape(-1, 1), (pred_index // bw).reshape(-1, 1)], axis=1)*4
        pred_points[:, 0] = pred_points[:, 0] * d2i[0, 0] + d2i[0, 2]
        pred_points[:, 1] = pred_points[:, 1] * d2i[1, 1] + d2i[1, 2]

        # # 4. 球门点可视化
        # print("pred_coord: ", pred_points)
        # for i in range(pred_points.shape[0]):
        #     cv2.circle(image, (int(pred_points[i, 0]), int(pred_points[i, 1])), 10, (0, 0, 255), 6)
        #     cv2.putText(image, str(i + 1), (int(pred_points[i, 0]) + 10, int(pred_points[i, 1]) + 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 255, 0), thickness=4)
        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img', 7600 // 4, 2160 // 4)
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return pred_points


def onnx_infer_30(video_path, onnx_path, frame_w, frame_h, goalnet_onnx_path):
    # 1. 30帧图片
    video_imgs = [os.path.join(video_path, file) for file in
                  os.listdir(video_path) if file.endswith(('.jpg', '.png', '.bmp'))]
    video_imgs = sorted(video_imgs, key=lambda x: int(os.path.basename(x).split('.')[0]))
    video_len = len(video_imgs)
    assert video_len==30, print("Video_len should be 30 frames: ", len(video_imgs))

    goalnet_model = GoalNetModel(goalnet_onnx_path, input_width=1600, input_height=480)
    # 2. 30帧图片读取，裁剪，normalize
    left_frames = np.zeros((video_len, 3, frame_h, frame_w), dtype=np.float32)
    right_frames = np.zeros((video_len, 3, frame_h, frame_w), dtype=np.float32)
    for idx, img_path in enumerate(video_imgs):
        image = cv2.imread(img_path)

        # 推理第一帧的左右球门位置 （如果相机固定，根据机器性能需求可以不用频繁推理）
        if idx == 0:
            h, w = image.shape[:2]
            goalnet_points = goalnet_model.infer(image)
            # 左边球门
            left_cx1, left_cy1 = (goalnet_points[0] + goalnet_points[3]) / 2
            left_cx1, left_cy1 = int(left_cx1), int(left_cy1)
            left_lx = min(max(left_cx1 - frame_w // 2, 0), w - frame_w)
            left_ly = min(max(left_cy1 - frame_h // 2, 0), h - frame_h)

            # # 右边球门
            right_cx1, right_cy1 = (goalnet_points[4] + goalnet_points[7]) / 2
            right_cx1, right_cy1 = int(right_cx1), int(right_cy1)
            right_lx = min(max(right_cx1 - frame_w // 2, 0), w - frame_w)
            right_ly = min(max(right_cy1 - frame_h // 2, 0), h - frame_h)

        left_img = image[left_ly:left_ly + frame_h, left_lx:left_lx + frame_w].copy()  # 根据球门裁剪图片
        left_img = left_img.astype(np.float32) / 255.0  # normalize
        left_frame = np.transpose(left_img, (2, 0, 1))
        left_frames[idx] = left_frame

        right_img = image[right_ly:right_ly + frame_h, right_lx:right_lx + frame_w].copy()  # 根据球门裁剪图片
        right_img = right_img.astype(np.float32) / 255.0  # normalize
        right_frame = np.transpose(right_img, (2, 0, 1))
        right_frames[idx] = right_frame

    # 3. 模型推理
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])  # CPUExecutionProvider
    # 左边进球
    t1 = time.time()
    left_inputs = {session.get_inputs()[0].name: left_frames}
    left_output = session.run(None, left_inputs)   # 模型推理
    print(f'left infer time: {time.time() - t1}')
    print("left infer result: ", left_output[0])
    print("left ball confidence: ", left_output[0][0, 1])
    print('*'*30)

    # 右边进球
    t1 = time.time()
    right_inputs = {session.get_inputs()[0].name: right_frames}
    right_output = session.run(None, right_inputs)  # 模型推理
    print(f'right infer time: {time.time() - t1}')
    print("right infer result: ", right_output[0])
    print("right ball confidence: ", right_output[0][0, 1])


if __name__ == '__main__':
    input_width = 640
    input_height = 384
    ballspotting_onnx_path = f'./ballspotting_woGSM_part1_20240722.onnx'
    goalnet_onnx_path = r"./goalnet_12points_20240625.onnx"
    video_path = r"./frames"
    onnx_infer_30(video_path, ballspotting_onnx_path, input_width, input_height, goalnet_onnx_path)
