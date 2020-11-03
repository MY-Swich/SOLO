from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import numpy as np
from scipy import ndimage


config_file = '../configs/solov2/solov2_r50_fpn_8gpu_1x___.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../work_dirs/solov2_12/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image
img = 'demo.jpg'
result = inference_detector(model, img)




def show_image_demo(img,
                    result,
                    class_names,
                    score_thr=0.3,
                    sort_by_density=False,
                    out_file=None):
    """Visualize the instance segmentation results on the image.

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The instance segmentation result.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the masks.
        sort_by_density (bool): sort the masks by their density.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """

    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape  # 获取照片的H和W,_是省略channel

    cur_result = result[0]  # 获取result
    seg_label = cur_result[0]  # result[0][0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)  # 转换成numpy数组
    cate_label = cur_result[1]  # result[0][1],这是类别
    cate_label = cate_label.cpu().numpy()  # 转换成numpy数组
    score = cur_result[2].cpu().numpy()  # result[0][2],并转换为numpy数组，这是阈值

    vis_inds = score > score_thr  # vis_inds是bool类型
    seg_label = seg_label[vis_inds]  # result[0][0][vis_inds],显示为真的mask数组
    num_mask = seg_label.shape[0]  # 统计mask的数量
    cate_label = cate_label[vis_inds]  # 可以显示的类别
    cate_score = score[vis_inds]  # 可以显示的类别的得分

    if sort_by_density:  # 根据mask密度排序
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = mmcv.imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    # 生成颜色不同的mask的颜色
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    for idx in range(num_mask):
        idx = -(idx + 1)  # idx是mask的id，为啥加-
        cur_mask = seg_label[idx, :, :]  # 选定一个mask
        cur_mask = mmcv.imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)  # 将bool值转换为int类型的0，1真值
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx][0]  # 选择颜色
        # cur_mask_bool = cur_mask.astype(np.bool)
        # img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        cur_cate = cate_label[idx]
        cur_score = cate_score[idx]

        b_boxs = np.argwhere(cur_mask == 1).T
        y, x = b_boxs
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        cv2.rectangle(img_show, (xmin, ymin), (xmax, ymax), (color_mask[0].item(), color_mask[1].item(), color_mask[2].item()), 2)

        label_text = class_names[cur_cate]
        # label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))  # 确定名称的位置
        cv2.putText(img_show, label_text, vis_pos,
                    cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green    在图片上写类别
    if out_file is None:
        return img
    else:
        mmcv.imwrite(img_show, out_file)

show_image_demo(img, result, model.CLASSES, score_thr=0.25, out_file="demo_out2.jpg")