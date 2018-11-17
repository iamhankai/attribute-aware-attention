## Attribute-Aware Attention Model
Code for paper: Attribute-Aware Attention Model for Fine-grained Representation Learning

![](./fig/a3m.png){:height="70%" width="70%"}

### Usage
Requires: Keras 1.2.1 ("image_data_format": "channels_first")

Run in two steps:

1. Download CUB-200-2011 dataset [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and unzip it to `$CUB`; Copy file `tools/processed_attributes.txt` in `$CUB`.

- The `$CUB` dir should be like this:
![](./fig/cub-dir.png)

2. Change `data_dir` in `run.sh` to `$CUB`, run the scprit `sh run.sh` to obtain the result.

- Result on CUB dataset
![](./fig/result.png){:height="50%" width="50%"}

### Citation

Please use the following bibtex to cite our work:
```
@inproceedings{han2018attribute,
  title={Attribute-Aware Attention Model for Fine-grained Representation Learning},
  author={Han, Kai and Guo, Jianyuan and Zhang, Chao and Zhu, Mingjian},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={2040--2048},
  year={2018},
  organization={ACM}
}
```
