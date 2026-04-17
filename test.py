import os
import argparse
import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch
from network.dual_model import ICFSeg, TeacherICFDSeg
from utils.utils import create_dir, seeding
from utils.utils import calculate_metrics



def process_mask(y_pred, threshold=0.5):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > threshold
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def create_comparison_image(original_img, true_mask, pred_mask, name):
    """
    创建原图、真实mask和预测结果的对比图像

    Args:
        original_img: 原始图像 (H, W, 3)
        true_mask: 真实掩码 (H, W, 3) - 已经是3通道
        pred_mask: 预测掩码 (H, W, 3) - 已经是3通道
        name: 图像名称

    Returns:
        comparison_img: 对比图像
    """
    h, w = original_img.shape[:2]

    if true_mask.shape[:2] != (h, w):
        true_mask = cv2.resize(true_mask, (w, h))
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h))

    
    comparison_img = np.hstack([original_img, true_mask, pred_mask])

    separator_width = 2
    separator_color = (255, 255, 255) 

  
    comparison_img[:, w:w + separator_width] = separator_color
    comparison_img[:, 2 * w + separator_width:2 * w + 2 * separator_width] = separator_color


    label_height = 40
    label_img = np.ones((label_height, comparison_img.shape[1], 3), dtype=np.uint8) * 0  # 黑色背景

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)  
    thickness = 2

 
    label_width = w + separator_width
    label_center_y = label_height // 2 + 5

    cv2.putText(label_img, "Original", (w // 2 - 30, label_center_y), font, font_scale, font_color, thickness)

    cv2.putText(label_img, "Ground Truth", (w + separator_width + w // 2 - 40, label_center_y), font, font_scale,
                font_color, thickness)
  
    cv2.putText(label_img, "Prediction", (2 * w + 2 * separator_width + w // 2 - 35, label_center_y), font, font_scale,
                font_color, thickness)

    final_img = np.vstack([comparison_img, label_img])

    return final_img


def print_score(metrics_score, test_length):
    jaccard = metrics_score[0] / test_length
    f1 = metrics_score[1] / test_length
    recall = metrics_score[2] / test_length
    precision = metrics_score[3] / test_length
    acc = metrics_score[4] / test_length
    f2 = metrics_score[5] / test_length

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")


def evaluate(model, model_type, save_path, test_x, test_y, size, device):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    test_length = len(test_x)

    comparison_dir = os.path.join(save_path, "comparison")
    create_dir(comparison_dir)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=test_length):
        name = os.path.basename(y).split(".")[0]
       
        if "_segmentation" in name:
            name = name.replace("_segmentation", "")

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        if image is None:
            print(f"警告: 无法读取图像 {x}，跳过")
            continue

        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"警告: 无法读取掩码 {y}，跳过")
            continue

        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
      
            outputs = model(image)

            if model_type == "baseline":
                
                mask_pred = outputs[0]
                threshold = 0.5  
            elif model_type == "sid_cdfa":
              
                mask_pred = outputs[0]
                threshold = 0.5
            elif model_type == "sid_cdfa_sa":
                
                mask_pred = outputs[0]
                threshold = 0.5
            elif model_type == "teacher":
                
                mask_pred = outputs[0]
                threshold = 0.5
            elif model_type == "full":
              
                mask_pred = (outputs[0] + outputs[1]) / 2.0
                threshold = 0.4  
            else:
                
                mask_pred = outputs[0]
                threshold = 0.5

          
            if i < 3:
                print(f"样本 {i + 1} ({model_type}): 输出范围=[{mask_pred.min():.4f}, {mask_pred.max():.4f}]")

              
                if model_type == "baseline":
                   
                    processed_pred = mask_pred
                else:
                    
                    processed_pred = torch.sigmoid(mask_pred)

                binary_pred = (processed_pred > threshold).float()
                foreground_ratio = binary_pred.mean().item()
                print(f"  处理后范围: [{processed_pred.min():.4f}, {processed_pred.max():.4f}]")
                print(f"  二值化后前景比例: {foreground_ratio:.4f}")

            p1 = mask_pred

            """ Evaluation metrics """
            score_1 = calculate_metrics(mask, p1)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            p1 = process_mask(p1, threshold)

           
            comparison_img = create_comparison_image(save_img, save_mask, p1, name)

            cv2.imwrite(f"{comparison_dir}/{name}_comparison.jpg", comparison_img)

        cv2.imwrite(f"{save_path}/mask/{name}.jpg", p1)

    print_score(metrics_score_1, test_length)

    with open(f"{save_path}/result.txt", "w") as file:
        file.write(f"Jaccard: {metrics_score_1[0] / test_length:1.4f}\n")
        file.write(f"F1: {metrics_score_1[1] / test_length:1.4f}\n")
        file.write(f"Recall: {metrics_score_1[2] / test_length:1.4f}\n")
        file.write(f"Precision: {metrics_score_1[3] / test_length:1.4f}\n")
        file.write(f"Acc: {metrics_score_1[4] / test_length:1.4f}\n")
        file.write(f"F2: {metrics_score_1[5] / test_length:1.4f}\n")


def detect_model_type(checkpoint_path):
    """根据checkpoint路径检测模型类型"""
    if "teacher_checkpoint" in checkpoint_path:
        return "teacher"
    elif "baseline" in checkpoint_path:
        return "baseline"
    elif "sid_cdfa_sa" in checkpoint_path:
        return "sid_cdfa_sa"
    elif "sid_cdfa" in checkpoint_path:
        return "sid_cdfa"
    else:
        return "full"  


def load_model(model_type, checkpoint_path, device):
    """根据模型类型加载相应的模型"""
    print(f"检测到模型类型: {model_type}")

    if model_type == "teacher":
        model = TeacherICFDSeg()
        print("使用教师模型 (TeacherICFDSeg)")
    elif model_type == "baseline":
        model = BaselineConDSeg()
        print("使用基线模型 (BaselineConDSeg)")
    elif model_type == "sid_cdfa":
        model = SIDCDFAConDSeg()
        print("使用SID+CDFA模型 (SIDCDFAConDSeg)")
    elif model_type == "sid_cdfa_sa":
        model = SIDCDFASAConDSeg()
        print("使用SID+CDFA+SA模型 (SIDCDFASAConDSeg)")
    else:
        model = ICFSeg()
        print("使用完整ICFSeg模型 (ICFSeg)")

    model = model.to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print(f"✓ {model_type}模型checkpoint加载成功")
    except Exception as e:
        print(f"✗ {model_type}模型checkpoint加载失败: {e}")
        print("尝试使用strict=False加载...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print(f"✓ {model_type}模型checkpoint加载成功（使用strict=False）")
        except Exception as e2:
            print(f"✗ {model_type}模型checkpoint加载完全失败: {e2}")
            raise e2

    model.eval()
    return model


def load_test_data(test_path, exclude_patterns=None):
    """
    直接从测试目录加载图像和掩码，不使用val.txt

    Args:
        test_path: 测试数据目录
        exclude_patterns: 要排除的文件名模式列表

    Returns:
        test_x: 测试图像路径列表
        test_y: 测试掩码路径列表
    """
    if exclude_patterns is None:
        exclude_patterns = []

    
    images_dir = os.path.join(test_path, "images")
    masks_dir = os.path.join(test_path, "masks")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")

    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"掩码目录不存在: {masks_dir}")

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    mask_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

  
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))

    
    filtered_image_files = []
    for img_path in image_files:
        if not any(pattern in img_path for pattern in exclude_patterns):
            filtered_image_files.append(img_path)

   
    filtered_image_files.sort()

 
    test_x = []
    test_y = []

    for img_path in filtered_image_files:
        img_basename = os.path.basename(img_path)
        img_name = os.path.splitext(img_basename)[0]

       
        mask_found = False

  
        for ext in mask_extensions:
            mask_pattern = os.path.join(masks_dir, f"{img_name}.*")
            matching_masks = glob.glob(mask_pattern)

            if matching_masks:
                mask_path = matching_masks[0]  
                test_x.append(img_path)
                test_y.append(mask_path)
                mask_found = True
                break

       
        if not mask_found:
            for ext in mask_extensions:
                mask_pattern = os.path.join(masks_dir, f"{img_name}_segmentation.*")
                matching_masks = glob.glob(mask_pattern)

                if matching_masks:
                    mask_path = matching_masks[0]  
                    test_x.append(img_path)
                    test_y.append(mask_path)
                    mask_found = True
                    print(f"找到带_segmentation后缀的掩码: {os.path.basename(mask_path)} 对应图像: {img_basename}")
                    break

        if not mask_found:
            print(f"警告: 图像 {img_basename} 没有找到对应的掩码文件")

    return test_x, test_y


if __name__ == "__main__":
    """ 添加命令行参数 """
    parser = argparse.ArgumentParser(description="测试ConDSeg模型的性能")
    parser.add_argument("--dataset", type=str, default="ISIC2018", help="数据集名称")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data_path", type=str, default="data/test", help="数据集根目录")
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存目录，默认为results/{dataset}/MyModel_dual_2018")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--test_dir", type=str, help="测试目录的直接路径，优先于根据dataset构建路径")
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    """ Seeding """
    dataset_name = args.dataset
    seeding(42)
    size = (256, 256)

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    checkpoint_path = args.checkpoint
    print(f"加载模型: {checkpoint_path}")

    model_type = detect_model_type(checkpoint_path)
    model = load_model(model_type, checkpoint_path, device)

    """ Test dataset """
    if args.test_dir:
        test_path = args.test_dir
    else:
        test_path = os.path.join(args.data_path, dataset_name)

    print(f"加载测试数据: {test_path}")

    exclude_patterns = ["_superpixels"]
    print(f"排除包含以下模式的文件: {exclude_patterns}")


    test_x, test_y = load_test_data(test_path, exclude_patterns)
    print(f"测试集样本数: {len(test_x)}")

    if len(test_x) == 0:
        print("错误: 未找到有效的测试数据!")
        exit(1)

    if args.output:
        save_path = args.output
    else:
        save_path = f"results/{dataset_name}/MyModel_dual_2018"

    print(f"结果将保存到: {save_path}")
    create_dir(f"{save_path}")
    create_dir(f"{save_path}/mask")
    create_dir(f"{save_path}/comparison")

 
    evaluate(model, model_type, save_path, test_x, test_y, size, device)




