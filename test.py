import sys
sys.path.insert(0, '.')
import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
import time
from argparse import ArgumentParser
from models.model import MSDIG_DTFENet

def ValidateSegmentationPrecision(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = MSDIG_DTFENet(3, 1)

    # 设置文件根目录
    if args.file_root == 'LIVER':
        args.file_root = 'D:\yanjiu\L-Net_L\LIVER'
    elif args.file_root == 'SYSU':
        args.file_root = 'D:\yanjiu\L-Net_L\SYSU'
    elif args.file_root == 'CDTH':
        args.file_root = 'D:\yanjiu\L-Net_L\S_CDTH'
    else:
        raise TypeError('%s has not defined' % args.file_root)


    if args.onGPU:
        model = model.cuda()

    # 设置均值和标准差
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # 数据预处理
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    # 加载数据
    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    # 加载模型
    model_file_name = args.weight
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        salEvalVal = ConfuseMatrixMeter(n_class=2)

        start = time.time()
        for iter, batched_inputs in enumerate(testLoader):
            img, target = batched_inputs
            pre_img = img[:, 0:3]
            post_img = img[:, 3:6]

            if args.onGPU == True:
                pre_img = pre_img.cuda()
                target = target.cuda()
                post_img = post_img.cuda()

            pre_img_var = torch.autograd.Variable(pre_img).float()
            post_img_var = torch.autograd.Variable(post_img).float()
            target_var = torch.autograd.Variable(target).float()

            # 模型推理
            output = model(pre_img_var, post_img_var)
            pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

            # 计算评估指标
            pr = pred[0, 0].cpu().numpy()
            gt = target_var[0, 0].cpu().numpy()

            f1 = salEvalVal.update_cm(pr, gt)

        print(time.time() - start)

        scores = salEvalVal.get_scores()
        print('Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}, IoU:{:.4f}, OA:{:.4f}'.format(
            scores['precision'], scores['recall'], scores['F1'], scores['Iou'], scores['OA']))

    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="CDTH", help='Data directory | LIVER | SYSU | GZCD | CDTH')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='D:\yanjiu\MDIG-DTENet\log\CDTH_iter_10000_lr_0.0001\Epoch127_0.6713029150092643.pth', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    ValidateSegmentationPrecision(args)


