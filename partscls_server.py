import torch
import yaml
import grpc
import base64
import cv2
import numpy as np
from concurrent import futures
import detect_pb2, detect_pb2_grpc
from nets import resnet
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from commons.augmentations import ScalePadding
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

test_transform = A.Compose([
    A.Resize(512, 512),
    A.CenterCrop(384, 384, p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

idx_name_map = {
0: '黑边',# 异常类别
1: '口咽部-正镜',
2: '食管-正镜',
3: '贲门-正镜',
4: '贲门-倒镜',
5: '胃底-倒镜',
6: '胃体中上部-正镜',
7: '胃体中上部-倒镜',
8: '胃体下部-正镜',
9: '胃角-正镜',
10: '胃角-倒镜',
11: '胃窦-正镜',
12: '幽门-正镜',
13: '十二指肠球部-正镜',
14: '十二指肠降部-正镜',
15: '十二指肠乳头-正镜',
17: '均值小于50',
18: '均值大于180',
19: '纯色',
}
index2n = {
0:             1,
1:             2,
2:             3,
3:             4,
4:       5,
5:       5,
6:       5,
7:       5,
8:     6,
9:     6,
10:     6,
11:     6,
12:   7,
13:   7,
14:   7,
15:   7,
16:   8,
17:   8,
18:   8,
19:   8,
20:            9,
21:      10,
22:      10,
23:      10,
24:   11,
25:   11,
26:   11,
27:   11,
28:            12,
29:            13,
30:            14,
31:            15,
}
#
# basic_transform = Compose([
#     Resize(size=512),
#     CenterCrop(size=384),
#     Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ToTensor()
# ])


def model_builder(cfg_path="./configs/part_cls.yaml",
                  weight_path="./weights/resnet_cls_9_best.pth"):
    with open(cfg_path, 'r') as rf:
        cfg_model = yaml.safe_load(rf)['model']
    model: torch.nn.Module = getattr(resnet, cfg_model['name'])(pretrained=False,
                                                                num_classes=cfg_model['num_classes'],
                                                                dropout=cfg_model['dropout'],
                                                                reduction=cfg_model['reduction'])
    model.load_state_dict(torch.load(weight_path,map_location="cpu"), strict=False)
    return model



class DetectorServer(detect_pb2_grpc.DetectService):
    def __init__(self, cfg_path="config/server.yml"):
        with open(cfg_path, 'r') as rf:
            self.cfg = yaml.safe_load(rf)
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:{:d}".format(self.cfg['gpu']))
        self.host = self.cfg['host2']
        self.port = self.cfg['port']
        self.max_workers = self.cfg['max_workers']
        self.reprocess = ScalePadding(self.cfg['img_size'])
        # self.model = model_builder().eval().to(self.device)
        self.model = torch.jit.load('./weights/swin88.jit').eval().to(self.device)
        print("init complete")


    def jdg_heibian(self, img):
        """
        :param img:
        :return: 1 表示图像有黑边
        """
        #     if np.std(img) < 23: return 0
        thre1 = 0.8
        thre2 = 0.03
        thre3 = 25
        h, w, c = img.shape
        cnt = 0
        for i in range(w):
            ic = np.sum((img[:, i, 0] < thre3).astype(int))
            if ic / h > thre1:
                cnt += 1
        if cnt / w > thre2: return 1
        cnt = 0
        for i in range(h):
            ic = np.sum((img[i, :, 0] < thre3).astype(int))
            if ic / w > thre1:
                cnt += 1
        if cnt / h > thre2: return 1
        return 0

    def jblur(self, img):
        img_blur_val = cv2.Laplacian(img, cv2.CV_64F).var()
        return 1 if img_blur_val > 400 else 0

    # def filter_im(self, im):
    #     fl =
    #
    #     if im is None: return 1
    #     if np.std(im) == 0.: return 1
    #     if self.jblur(im)

    def jg_yichang(self, im):
        """
        :param im:
        :return: 1 yichag, -1 ok
            # 16: '长宽不合理',
            # 17: '均值小于50',
            # 18: '均值大于180',
            # 19: '纯色',
        """
        if np.std(im) == 0: return 19
        if np.mean(im) > 190: return 18
        if np.mean(im) < 40: return 17
        return -1

    def response_ret(self, response, predict_idx, conf):
        print(predict_idx, idx_name_map[predict_idx])
        response.num = 1
        response.anchors.append(detect_pb2.Anchor(x1=0.01, x2=0.99, y1=0.05, y2=0.99, confidence=conf, type=predict_idx))
        return response

    def detect(self, request, context):
        response = detect_pb2.DetectResponse()
        w, h, c, img_str = request.width, request.height, request.channel, request.img
        b64_str = base64.b64decode(img_str)
        np_buf = np.frombuffer(b64_str, dtype=np.uint8)
        ori_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        print(image.shape)
        print(1111111)
        # 异常图片提前返回0 16
        predict_idx = 15
        if self.jdg_heibian(image) == 1:
            predict_idx = 0
        # elif self.jblur(image) == 1:
        #     predict_idx = 16

        if predict_idx == 0:
            conf = 1.
            return self.response_ret(response, predict_idx, conf)

        image_data = test_transform(image=image)['image'].contiguous().unsqueeze(dim=0).to(self.device)
        pred = self.model(image_data)[:, 1].reshape(-1, 35)
        pred = torch.nn.Sigmoid()(pred)[0]
        pred = pred.cpu().data.numpy()
        pred_sort = np.argsort(-pred)
        for i, x in enumerate(pred_sort):
            if x not in [32, 33, 34]:
                # if x == 0: # 口咽部
                #     if pred[x] < 0.5:
                #         i += 1
                #         continue
                predict_idx = index2n[x]
                #1
                if predict_idx in [9, 10]:  # 胃角
                    return self.response_ret(response, predict_idx, pred[x])
                #2
                if self.jg_yichang(image) in [17, 18, 19]:
                    return self.response_ret(response, self.jg_yichang(image), 1.)
                #3
                return self.response_ret(response, predict_idx, pred[x])



if __name__ == '__main__':
    server = DetectorServer()
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=server.max_workers))
    detect_pb2_grpc.add_DetectServiceServicer_to_server(server, grpc_server)
    grpc_server.add_insecure_port("{:s}:{:s}".format(str(server.host), str(server.port)))
    grpc_server.start()
    grpc_server.wait_for_termination()
