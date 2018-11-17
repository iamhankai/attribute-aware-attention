## Attribute-Aware Attention Model
Code for paper: Attribute-Aware Attention Model for Fine-grained Representation Learning

![](./fig/a3m.png)

### Usage
Requires: Keras 1.2.1 ("image_data_format": "channels_first")

1. Download CUB-200-2011 dataset [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and unzip it to `$CUB`; Copy file `tools/processed_attributes.txt` in `$CUB`.

The `$CUB` dir should be like this:
![](./fig/dub-dir.png)

2. Change `data_dir` in `run.sh` to `$CUB`, run the scprit `sh run.sh` to obtain the result.

- Result on CUB dataset
![](./fig/result.png)

### Citation

Please use the following bibtex to cite our work:
```
@inproceedings{han2018attribute,
  title={Attribute-Aware Attention Model for Fine-grained Representation Learning},
  author={Han, Kai and Guo, Jianyuan and Zhang, Chao and Zhu, Mingjian},
  booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
  pages={2040--2048},
  year={2018},
  organization={ACM}
}
```
