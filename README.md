# Detección y seguimiento de objetos mediante aprendizaje profundo para conducción autónoma

## Descripción

El siguiente documento se basa en la implementacion de: [Complex YOLOv4](https://github.com/maudzung/Complex-YOLOv4-Pytorch)

Mas conjuntos de datos se pueden encontrar en: [KITTY Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php)

<figure class="image">
  <p align="center"> <img src="assets/test_bench.gif" alt></p>
  <figcaption align="center">Result Ouput</figcaption>
</figure>

## Estructura de carpeta

## Docker

## Local

## Repositorio Complex YOLOv4

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 531.29                 Driver Version: 531.29       CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                      TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060 L...  WDDM | 00000000:01:00.0  On |                  N/A |
| N/A   52C    P5               11W /  N/A|    546MiB /  6144MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

### 1. Clonar repositorio

```bash
git clone https://github.com/maudzung/Complex-YOLOv4-Pytorch.git
```

### 2. Configuracion del ambiente de trabajo



#### 2.1 Instalar Dependencias

### Utilizacion

```
usage: train.py [-h] [--seed SEED] [--saved_fn FN] [--working-dir PATH]
                [-a ARCH] [--cfgfile PATH] [--pretrained_path PATH]
                [--img_size IMG_SIZE] [--hflip_prob HFLIP_PROB]
                [--cutout_prob CUTOUT_PROB] [--cutout_nholes CUTOUT_NHOLES]
                [--cutout_ratio CUTOUT_RATIO]
                [--cutout_fill_value CUTOUT_FILL_VALUE]
                [--multiscale_training] [--mosaic] [--random-padding]
                [--no-val] [--num_samples NUM_SAMPLES]
                [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE]
                [--print_freq N] [--tensorboard_freq N] [--checkpoint_freq N]
                [--start_epoch N] [--num_epochs N] [--lr_type LR_TYPE]
                [--lr LR] [--minimum_lr MIN_LR] [--momentum M] [-wd WD]
                [--optimizer_type OPTIMIZER] [--burn_in N]
                [--steps [STEPS [STEPS ...]]] [--world-size N] [--rank N]
                [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                [--gpu_idx GPU_IDX] [--no_cuda]
                [--multiprocessing-distributed] [--evaluate]
                [--resume_path PATH] [--conf-thresh CONF_THRESH]
                [--nms-thresh NMS_THRESH] [--iou-thresh IOU_THRESH]

The Implementation of Complex YOLOv4

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           re-produce the results with seed random
  --saved_fn FN         The name using for saving logs, models,...
  --working-dir PATH    The ROOT working directory
  -a ARCH, --arch ARCH  The name of the model architecture
  --cfgfile PATH        The path for cfgfile (only for darknet)
  --pretrained_path PATH
                        the path of the pretrained checkpoint
  --img_size IMG_SIZE   the size of input image
  --hflip_prob HFLIP_PROB
                        The probability of horizontal flip
  --cutout_prob CUTOUT_PROB
                        The probability of cutout augmentation
  --cutout_nholes CUTOUT_NHOLES
                        The number of cutout area
  --cutout_ratio CUTOUT_RATIO
                        The max ratio of the cutout area
  --cutout_fill_value CUTOUT_FILL_VALUE
                        The fill value in the cut out area, default 0. (black)
  --multiscale_training
                        If true, use scaling data for training
  --mosaic              If true, compose training samples as mosaics
  --random-padding      If true, random padding if using mosaic augmentation
  --no-val              If true, dont evaluate the model on the val set
  --num_samples NUM_SAMPLES
                        Take a subset of the dataset to run and debug
  --num_workers NUM_WORKERS
                        Number of threads for loading data
  --batch_size BATCH_SIZE
                        mini-batch size (default: 4), this is the totalbatch
                        size of all GPUs on the current node when usingData
                        Parallel or Distributed Data Parallel
  --print_freq N        print frequency (default: 50)
  --tensorboard_freq N  frequency of saving tensorboard (default: 20)
  --checkpoint_freq N   frequency of saving checkpoints (default: 2)
  --start_epoch N       the starting epoch
  --num_epochs N        number of total epochs to run
  --lr_type LR_TYPE     the type of learning rate scheduler (cosin or
                        multi_step)
  --lr LR               initial learning rate
  --minimum_lr MIN_LR   minimum learning rate during training
  --momentum M          momentum
  -wd WD, --weight_decay WD
                        weight decay (default: 1e-6)
  --optimizer_type OPTIMIZER
                        the type of optimizer, it can be sgd or adam
  --burn_in N           number of burn in step
  --steps [STEPS [STEPS ...]]
                        number of burn in step
  --world-size N        number of nodes for distributed training
  --rank N              node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --gpu_idx GPU_IDX     GPU index to use.
  --no_cuda             If true, cuda is not used.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --evaluate            only evaluate the model, not training
  --resume_path PATH    the path of the resumed checkpoint
  --conf-thresh CONF_THRESH
                        for evaluation - the threshold for class conf
  --nms-thresh NMS_THRESH
                        for evaluation - the threshold for nms
  --iou-thresh IOU_THRESH
                        for evaluation - the threshold for IoU
```
