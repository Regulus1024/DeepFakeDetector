import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from training.utils.model import LRNet  # 确保模型的导入正确
import dlib
from tkinter import Tk, filedialog


def open_file_dialog():
    root = Tk()
    root.withdraw()

    file_paths = (filedialog.
                 askopenfilename(title="Select your image(s)",
                                 filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.gif")], initialdir="./images"))
    root.destroy()
    return file_paths


def visualize_prediction(image, title):
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(title)
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def load_model_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cuda')))
    model.eval()


def model_inference(model, input_image):
    with torch.no_grad():
        x = model.dropout_landmark(input_image)
        x = x.view(64, -1)
        linear_layer = torch.nn.Linear(x.size(-1), 64)
        x = linear_layer(x)
        output = model.dense(x)
        output = model.output(output)
    return output


# 定义LRNet模型，确保路径和模型名称正确
lrnet_model1 = LRNet(feature_size=136, lm_dropout_rate=0.1, rnn_unit=32,
                     num_layers=1, rnn_dropout_rate=0,
                     fc_dropout_rate=0.5, res_hidden=64)
lrnet_model2 = LRNet(feature_size=136, lm_dropout_rate=0.1, rnn_unit=32,
                     num_layers=1, rnn_dropout_rate=0,
                     fc_dropout_rate=0.5, res_hidden=64)
lrnet_model3 = LRNet(feature_size=136, lm_dropout_rate=0.1, rnn_unit=32,
                     num_layers=1, rnn_dropout_rate=0,
                     fc_dropout_rate=0.5, res_hidden=64)
lrnet_model4 = LRNet(feature_size=136, lm_dropout_rate=0.1, rnn_unit=32,
                     num_layers=1, rnn_dropout_rate=0,
                     fc_dropout_rate=0.5, res_hidden=64)

# 加载预训练的模型权重，确保路径和文件名正确
model_weights_path1 = r"training\weights\torch\g1_test.pth"
model_weights_path2 = r"training\weights\torch\g2_test.pth"
model_weights_path3 = r"training\weights\torch\g1.pth"
model_weights_path4 = r"training\weights\torch\g2.pth"

load_model_weights(lrnet_model1, model_weights_path1)
load_model_weights(lrnet_model2, model_weights_path2)
load_model_weights(lrnet_model3, model_weights_path3)
load_model_weights(lrnet_model4, model_weights_path4)
lrnet_model1.eval()
lrnet_model2.eval()
lrnet_model3.eval()
lrnet_model4.eval()
# 读取图像
image_path = open_file_dialog()
if not image_path:
    print("No image was selected. Now exiting.")
image = Image.open(image_path).convert('RGB')

# 加载dlib的人脸检测器
detector = dlib.get_frontal_face_detector()

# 转换为numpy数组
image_np = np.array(image)

# 使用dlib的人脸检测器检测人脸
faces = detector(image_np, 1)

# 如果检测到人脸
if len(faces) > 0:
    # 获取第一个人脸的坐标
    face_rect = faces[0]

    # 提取人脸区域
    face_image = image_np[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]

    # 将人脸区域转换为PIL图像
    face_pil = Image.fromarray(face_image)

    # 可视化原始图像和提取出的人脸
    visualize_prediction(transforms.ToTensor()(image).unsqueeze(0).squeeze(), title='Original Face')

    # 定义图像预处理转换
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 应用图像预处理
    input_image = data_transform(image)

    # 添加一个批次和时间维度
    input_image = input_image.unsqueeze(0)

    # 模型推理
    output1 = model_inference(lrnet_model1, input_image)
    output2 = model_inference(lrnet_model2, input_image)
    output3 = model_inference(lrnet_model3, input_image)
    output4 = model_inference(lrnet_model4, input_image)
    print(output1)
    # # 打印每个模型的输出结果
    prediction = torch.argmax(output1, dim=1)
    prediction2 = torch.argmax(output2, dim=1)
    prediction3 = torch.argmax(output3, dim=1)
    prediction4 = torch.argmax(output4, dim=1)
    print(prediction)
    print(prediction2)
    print(prediction3)
    print(prediction4)
    # # 计算每个模型的概率
    # predicted_class = torch.argmax(output1[:, 1]).item()
    #
    # # 可视化预测概率分布
    # plt.bar(range(output1.shape[0]), output1[:, 1].numpy(),
    #         tick_label=[f'Sample {i + 1}' for i in range(output1.shape[0])])
    # plt.xlabel('Sample')
    # plt.ylabel('Predicted Probability (Fake Face)')
    # plt.title('Predicted Probability Distribution')
    # plt.show()
    #
    # # 设置阈值，将预测概率高于阈值的样本判定为“Fake Face”
    # threshold = 0.5
    # predicted_labels = (output1[:, 1] > threshold).numpy()
    #
    # print(f"Predicted Class: {predicted_class}")
    # print(f"Predicted Labels: {predicted_labels}")
    # fake_face_probability1 = torch.mean(output1[:, 1])
    # fake_face_probability2 = torch.mean(output2[:, 1])
    # fake_face_probability3 = torch.mean(output3[:, 1])
    # fake_face_probability4 = torch.mean(output4[:, 1])
    fake_face_probability1 = output1[0, 0]
    fake_face_probability2 = output2[0, 0]
    fake_face_probability3 = output3[0, 0]
    fake_face_probability4 = output4[0, 0]

    # 设定阈值
    threshold = 0.5

    # 判断最终结果
    fake_count = sum(
        1 for prob in [fake_face_probability1, fake_face_probability2, fake_face_probability3, fake_face_probability4]
        if prob > threshold)

    # 可视化结果
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # 模型预测结果
    axs[0].imshow(np.transpose(input_image.squeeze().numpy(), (1, 2, 0)))
    axs[0].set_title(f'Model Predictions\nFake Count: {fake_count}')
    axs[0].axis('off')

    # 模型概率可视化
    axs[1].bar(['Model 1', 'Model 2', 'Model 3', 'Model 4'],
               [fake_face_probability1, fake_face_probability2, fake_face_probability3, fake_face_probability4],
               color=['blue', 'orange', 'green', 'red'])
    axs[1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    axs[1].set_title('Fake Face Probability Comparison')
    axs[1].set_ylabel('Probability')
    axs[1].legend()

    plt.show()
else:
    print("No faces detected in the image.")
