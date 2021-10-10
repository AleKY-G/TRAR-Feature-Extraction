## TRAR Extract Features
Extract grid features for our ICCV 2021 paper ["TRAR: Routing the Attention Spans in Transformers for Visual Question Answering"](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_TRAR_Routing_the_Attention_Spans_in_Transformer_for_Visual_Question_ICCV_2021_paper.pdf)

## Usage
### Installation
- Install Detectron 2 following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Since Detectron 2 is also being actively updated which can result in breaking behaviors, it is **highly recommended** to install via the following command:
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac'
```
Commits before or after `ffff8ac` might also work, but it could be risky.

- Clone this repo
```bash
git clone https://github.com/rentainhe/TRAR-Feature-Extraction.git
cd TRAR-Feature-Extraction
mkdir datasets
```

### Data
- Download COCO image data from [official download instructions](https://cocodataset.org/#download)

We use `coco_train_2014`, `coco_val_2014` for training and evaluation, `coco_test_2015` for online testing.

The file structure should look like:
```
datasets/coco/
├── annotations
│   ├── image_info_test2015.json
|   ├── instances_train2014.json
|   ├── instances_val2014.json
|   ├── ......
├── train2014
│   ├── COCO_train2014_000000000009.jpg
│   ├── COCO_train2014_000000000025.jpg
|   ├── ......
├── val2014
│   ├── COCO_train2014_000000000042.jpg
│   ├── COCO_train2014_000000000073.jpg
|   ├── ......
├── test2015
│   ├── COCO_train2014_000000000001.jpg
│   ├── COCO_train2014_000000000014.jpg
|   ├── ......
```

### Feature Extraction
Grid feature extraction can be done by simply running once the model is trained (or you can directly download our pre-trained models, see below):
```bash
python extract_trar_grid_feature.py --config-file configs/R-50-grid.yaml --dataset <dataset> --output_dir /path/to/save/features --weight_path /path/to/pretrained/model --feature_size 8
```
- `--dataset={'coco_train_2014', 'coco_val_2014', 'coco_test_2015'}`, e.g., to set the dataset to be extracted
- `--output_dir=str`, e.g., where to save the extracted features
- `--weight_path=str`, e.g., where to load the pretrained model weight
- `--feature_size=int`, default: `--feature_size=8`, e.g., set feature_size to 16 to get 16*16 features

and the code will load the model weight from `args.weight_path` (which one can override in command line) and start extracting features for `<dataset>` and save the extracted features to `args.output_dir`, we provide three options for the dataset: `coco_2014_train`, `coco_2014_val` and `coco_2015_test`, they correspond to `train`, `val` and `test` splits of the VQA dataset. The extracted features can be conveniently loaded in [Pythia](https://github.com/facebookresearch/pythia).

- For example: Extract `coco_2014_train` features using `pretrained ResNext152` Model
```
CUDA_VISIBLE_DEVICES=0 python extract_trar_grid_feature.py --config-file configs/X-152-grid.yaml --dataset coco_2014_train --weight_path ./weight/X-152.pth --output_dir ./data/
```
