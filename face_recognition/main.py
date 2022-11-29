import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


# 获得人脸特征向量
def load_known_faces(dstImgPath, mtcnn, resnet):
    aligned = []
    known_img = cv2.imread(dstImgPath)  # 读取图片
    face = mtcnn(known_img)  # 使用mtcnn检测人脸，返回【人脸数组】

    if face is not None:
        aligned.append(face[0])
    aligned = torch.stack(aligned).to(device)
    with torch.no_grad():
        known_faces_emb = resnet(aligned).detach().cpu()  # 使用resnet模型获取人脸对应的特征向量
    print("\n人脸对应的特征向量为：\n", known_faces_emb)
    return known_faces_emb, known_img


# 计算人脸特征向量间的欧氏距离，设置阈值，判断是否为同一个人脸
def match_faces(faces_emb, known_faces_emb, threshold):
    isExistDst = False
    distance = (known_faces_emb[0] - faces_emb[0]).norm().item()
    print("\n两张人脸的欧式距离为：%.2f" % distance)
    if distance < threshold:
        isExistDst = True
    return isExistDst


if __name__ == '__main__':
    # help(MTCNN)
    # help(InceptionResnetV1)
    # 获取设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # mtcnn模型加载【设置网络参数，进行人脸检测】
    mtcnn = MTCNN(min_face_size=12, thresholds=[0.2, 0.2, 0.3], keep_all=True, device=device)

    # InceptionResnetV1模型加载【用于获取人脸特征向量】
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    MatchThreshold = 0.8  # 人脸特征向量匹配阈值设置

    known_faces_emb, _ = load_known_faces('1.jpg', mtcnn, resnet)  # 已知人物图
    faces_emb, img = load_known_faces('images.jpg', mtcnn, resnet)  # 待检测人物图
    isExistDst = match_faces(faces_emb, known_faces_emb, MatchThreshold)  # 人脸匹配
    print("设置的人脸特征向量匹配阈值为：", MatchThreshold)

    if isExistDst:
        boxes, prob, landmarks = mtcnn.detect(img, landmarks=True)  # 返回人脸框，概率，5个人脸关键点
        print('由于欧氏距离小于匹配阈值，故匹配')
    else:
        print('由于欧氏距离大于匹配阈值，故不匹配')
