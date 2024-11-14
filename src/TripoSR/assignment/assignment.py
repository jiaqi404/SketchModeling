import torch
from PIL import Image
from ..tsr.models.tokenizers.image import DINOSingleImageTokenizer
from ..tsr.models.tokenizers.triplane import Triplane1DTokenizer

def load_image(image_path):
    image = Image.open(image_path)
    # 进行必要的预处理，例如调整大小、归一化等
    return image

def initialize_models():
    # 初始化图像特征提取模型-
    tokenizer_cfg = DINOSingleImageTokenizer.Config()
    tokenizer = DINOSingleImageTokenizer(tokenizer_cfg)
    
    # 初始化Triplane1DTokenizer
    triplane_cfg = Triplane1DTokenizer.Config(plane_size=64, num_channels=32)  # 示例参数
    triplane_tokenizer = Triplane1DTokenizer(triplane_cfg)
    
    return tokenizer, triplane_tokenizer

def generate_3d_model(image_path):
    image = load_image(image_path)
    tokenizer, triplane_tokenizer = initialize_models()

    # 将图像转换为张量并提取特征
    image_tensor = torch.tensor(image).unsqueeze(0)  # 添加批次维度
    features = tokenizer(image_tensor)

    # 使用特征生成3D模型
    batch_size = features.shape[0]
    triplane_features = triplane_tokenizer(batch_size)
    
    # 这里可以添加进一步处理triplane_features的代码
    return triplane_features

def main():
    image_path = r"..\examples\unicorn.png"
    model_3d = generate_3d_model(image_path)
    print("Generated 3D model:", model_3d)

if __name__ == "__main__":
    main()