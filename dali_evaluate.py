##git clone https://github.com/NVIDIA/apex
##cd apex
##pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
##pip install -v --no-cache-dir ./
## pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
import argparse
import os
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from models.ModelBuilder import*
from models.NetworkConfig import*
import json
import torch.optim as optim
import torch
from torch.optim.lr_scheduler import _LRScheduler
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

## color augment,epochs=240, lr=0.1, cosine lr+lr warmup, label smoothing


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', '--dataset', default="/dataset/imagenet2012", type=str, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--hardware', type=str, default='cpu', help='hardware type')
parser.add_argument('--num_classes',type=int,default=1000,help="number of classes")
parser.add_argument('--rootdir', type=str, default='logs/imagenet/retrain', help='experiment name')
parser.add_argument('--filename', type=str, default="testimagenet", help='gradient clipping')

cudnn.benchmark = True


## pytorch CyclicLR: https://github.com/TDehaene/blogposts/blob/master/learning_rates/notebooks/Model%20training.ipynb
## dali data augmentations: https://towardsdatascience.com/diving-into-dali-1c30c28731c0


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

best_prec1 = 0
args = parser.parse_args()



args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

# make apex optional
print('distributed',args.distributed)
if  args.distributed:
    try:
        from apex.parallel import DistributedDataParallel as DDP

    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def main():
    global best_prec1, args

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size




    # create model

    if os.path.isfile("architectures.json")==False:
        print("missing architectures.json!")
        return

    modelstr=json.load(open('architectures.json','r'))[args.hardware]
    print('current hardware:',args.hardware,'model:',modelstr)
    if os.path.isfile("pretrained/model_"+args.hardware+'.pth.tar')==False:
        print("missing pretrained model"+"pretrained/model_"+args.hardware+'.pth.tar')
        return
    args.resume="pretrained/model_"+args.hardware+'.pth.tar'
    mconfig=NetworkConfig(args.num_classes).build_modelconfig(modelstr.split(','))
    model=ModelBuilder(mconfig,args.num_classes)

    model = model.cuda()

    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()



    if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}' (epoch )"
                  .format(args.resume))
    else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code

    traindir = os.path.join(args.dataset, 'train')
    valdir = os.path.join(args.dataset, 'val')

    crop_size = 224
    val_size = 256

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir, crop=crop_size, size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))


    validate(val_loader, model, criterion)





def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
            val_loader_len = int(val_loader._size / args.batch_size)

            target = target.cuda(non_blocking=True)
            input_var = Variable(input)
            target_var = Variable(target)

            # compute output
            #with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {2:.3f} ({3:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, val_loader_len,
                       args.total_batch_size / batch_time.val,
                       args.total_batch_size / batch_time.avg,
                       batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_count += (pred == y).double().sum().data[0]
        #correct_count += (pred == y).sum().data[0]
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / float(batch_size)))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()
