import grpc
import os
import base64
import detect_pb2_grpc, detect_pb2
import cv2 as cv


def draw_box(img, x1, y1, x2, y2, conf, med_type):
    h, w, _ = img.shape
    x1, x2 = int(x1 * w), int(x2 * w)
    y1, y2 = int(y1 * w), int(y2 * w)
    cv.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
    cv.putText(img, text="cancer:{:4.2f}%".format(conf * 100),
               org=(x1, y1 + 20),
               fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=0.6,
               color=(0, 120, 200),
               thickness=2)
    cv.putText(img, text="type:{:d}".format(med_type),
               org=(x1 + 10, y1 + 20),
               fontFace=cv.FONT_HERSHEY_SIMPLEX,
               fontScale=0.6,
               color=(0, 120, 200),
               thickness=2)


def client_func(img):
    conn = grpc.insecure_channel("{:s}:{:s}".format("10.0.99.106", "50002"))  # 监听频道
    h, w, c = img.shape
    img_arr = cv.imencode(".jpg", img)[1]
    b64_str = base64.b64encode(img_arr.tobytes())
    client = detect_pb2_grpc.DetectServiceStub(
        channel=conn)
    response = client.detect(detect_pb2.DetectRequest(
        width=w, height=h, channel=c, img=b64_str))
    ret_img = img.copy()
    # if response.num < 1:
    #     return ret_img

    # for i in range(response.num):
    #     anchor = response.anchors[i]
    #     draw_box(ret_img, anchor.x1, anchor.y1, anchor.x2, anchor.y2, anchor.confidence, anchor.mtype)
    return ret_img


def run():
    base_path = "/home/huffman/yang/up_part_imgs/原始数据/di/"
    for img_name in os.listdir(base_path):
        print(img_name)
        img_path = os.path.join(base_path, img_name)
        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        ret_img = client_func(img)


if __name__ == '__main__':
    run()
