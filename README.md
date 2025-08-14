<h2 align="center">
  <span>
    Unveiling the Mist over 3D Vision-Language Understanding:<br/>Object-centric Evaluation with Chain-of-Analysis
  </span>
</h2>

<h3 align="center">
CVPR 2025
</h3>

<div align="center" margin-bottom="6em">
<a target="_blank" href="https://huangjy-pku.github.io/">Jiangyong Huang<sup>✶</sup></a>,&nbsp;
<a target="_blank" href="https://buzz-beater.github.io/">Baoxiong Jia<sup>✶</sup></a>,&nbsp;
<a target="_blank" href="https://github.com/jetpackfirstme">Yan Wang</a>,&nbsp;
<a target="_blank" href="https://zhuziyu-edward.github.io/">Ziyu Zhu</a>,&nbsp;
<a target="_blank" href="https://github.com/Germany321">Xiongkun Linghu</a>,
<br/>
<a target="_blank" href="https://liqing-ustc.github.io/">Qing Li</a>,&nbsp;
<a target="_blank" href="http://www.stat.ucla.edu/~sczhu/">Song-Chun Zhu</a>,&nbsp;
<a target="_blank" href="https://siyuanhuang.com/">Siyuan Huang</a>

</div>
&nbsp;

<div align="center">
    <a href="https://arxiv.org/abs/2503.22420" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://beacon-3d.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/Page-Beacon3D-9cf" alt="Project Page"></a>
    <a href="https://youtu.be/8hiGFwCQMjk" target="_blank">
    <img src="https://img.shields.io/badge/Video-YouTube-9966ff" alt="Video"></a>
    <a href="https://github.com/beacon-3d/beacon-3d/blob/main/data" target="_blank">
    <img src="https://img.shields.io/badge/Data-Beacon3D-blue" alt="Data"></a>
    <a href="https://huggingface.co/spaces/huangjy-pku/Beacon3D-Demo" target="_blank">
    <img src="https://img.shields.io/badge/Demo-Huggingface-darkorange" alt="Demo"></a>
</div>
&nbsp;

We introduce **Beacon3D**, a novel benchmark and evaluation protocol for 3D vision-language (3D-VL) models. **Beacon3D** covers both 3D grounding and question answering (QA) tasks, featuring an *object-centric evaluation framework* and *chain analysis for studying task coherence*.

<div align="center">
<img src="assets/keypoints.png" width="100%" alt="Keypoints">
</div>

This repository provides the [test data](#data), [evaluation pipeline](#evaluation), and an up-to-date [leaderboard](#leaderboard).

**Note:** The released data has been meticulously refined and may differ from the initial version used in the paper. Please refer to the [leaderboard](#leaderboard) for the latest results. We welcome updates or pull requests for adding the evaluation results of new models to the leaderboard.

## Leaderboard
For object-centric models, we use GT object masks by default unless specified.
We have updated the data for ScanNet, and the ScanNet results here are slightly different from the results in paper.
Please refer to the table here for the latest ScanNet results.

### ScanNet: QA
| Model | Class | App. | Geo. | Spa. | Exi. | Overall (Case) | Overall (Obj.) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [3D-VisTA](https://3d-vista.github.io/) | 28.4 | 35.7 | 41.6 | 48.0 | 55.0 | 43.2 | 7.3 |
| [PQ3D](https://pq3d.github.io/) | 37.8 | 45.8 | 32.1 | 19.2 | 44.5 | 35.9 | 4.2 |
| [SceneVerse](https://scene-verse.github.io/) | 26.4 | 40.4 | 40.0 | 35.0 | 54.1 | 40.5 | 4.7 |
| [LEO](https://embodied-generalist.github.io/) | 16.4 | 39.8 | 47.6 | 52.8 | 54.3 | 45.2 | 7.5 |
| [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene) | 30.0 | 42.7 | 50.0 | 53.9 | 62.9 | 49.8 | 10.9 |
| [GPT-4o](https://openai.com/index/gpt-4o-system-card/) | 39.2 | 49.9 | 53.8 | 58.4 | 70.0 | 56.0 | 15.3 |

### ScanNet: Grounding
| Model | Class | App. | Geo. | Spa. | Overall (Case) | Overall (Obj.) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [ViL3DRel](https://cshizhe.github.io/projects/vil3dref.html) | 65.4 | 69.0 | 53.0 | 63.5 | 65.4 | 44.4 |
| [3D-VisTA](https://3d-vista.github.io/) | 71.9 | 65.6 | 57.5 | 70.9 | 71.9 | 49.1 |
| [PQ3D](https://pq3d.github.io/) | 74.4 | 75.5 | 62.1 | 76.8 | 74.4 | 60.0 |
| [SceneVerse](https://scene-verse.github.io/) | 73.4 | 65.3 | 61.6 | 73.0 | 73.4 | 51.4 |

### 3RScan: QA
| Model | Class | App. | Geo. | Spa. | Exi. | Overall (Case) | Overall (Obj.) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [3D-VisTA](https://3d-vista.github.io/) | 15.2 | 24.1 | 28.2 | 25.3 | 28.9 | 25.7 | 3.3 |
| [PQ3D](https://pq3d.github.io/) | 6.5 | 19.6 | 13.6 | 16.6 | 52.6 | 25.7 | 0.7 |
| [SceneVerse](https://scene-verse.github.io/) | 28.3 | 32.3 | 34.6 | 38.9 | 44.6 | 37.4 | 0.4 |
| [LEO](https://embodied-generalist.github.io/) | 23.9 | 36.4 | 53.2 | 49.5 | 45.5 | 44.0 | 1.5 |
| [GPT-4o](https://openai.com/index/gpt-4o-system-card/) | 34.8 | 38.2 | 40.0 | 45.4 | 60.7 | 46.1 | 11.0 |

### 3RScan: Grounding
| Model | Class | App. | Geo. | Spa. | Overall (Case) | Overall (Obj.) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [ViL3DRel](https://cshizhe.github.io/projects/vil3dref.html) | 41.5 | 44.9 | 37.4 | 37.3 | 41.5 | 18.4 |
| [3D-VisTA](https://3d-vista.github.io/) | 45.6 | 38.3 | 37.4 | 40.9 | 45.6 | 21.7 |
| [PQ3D](https://pq3d.github.io/) | 38.3 | 28.0 | 36.4 | 35.3 | 38.3 | 13.6 |
| [SceneVerse](https://scene-verse.github.io/) | 61.8 | 51.4 | 53.3 | 57.3 | 61.8 | 37.5 |

### MultiScan: QA
| Model | Class | App. | Geo. | Spa. | Exi. | Overall (Case) | Overall (Obj.) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [3D-VisTA](https://3d-vista.github.io/) | 6.5 | 22.6 | 16.7 | 13.2 | 28.8 | 19.1 | 0.0 |
| [PQ3D](https://pq3d.github.io/) | 21.0 | 16.8 | 16.7 | 9.6 | 39.0 | 20.8 | 0.6 |
| [SceneVerse](https://scene-verse.github.io/) | 16.2 | 32.1 | 12.5 | 26.5 | 38.1 | 28.9 | 3.1 |
| [LEO](https://embodied-generalist.github.io/) | 11.3 | 24.3 | 49.0 | 26.7 | 30.9 | 26.2 | 0.6 |
| [GPT-4o](https://openai.com/index/gpt-4o-system-card/) | 29.0 | 41.6 | 33.3 | 25.7 | 59.3 | 39.4 | 7.6 |

### MultiScan: Grounding
| Model | Class | App. | Geo. | Spa. | Overall (Case) | Overall (Obj.) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [ViL3DRel](https://cshizhe.github.io/projects/vil3dref.html) | 33.2 | 34.4 | 25.0 | 32.0 | 33.2 | 13.2 |
| [3D-VisTA](https://3d-vista.github.io/) | 40.8 | 30.5 | 28.1 | 38.0 | 40.8 | 18.9 |
| [PQ3D](https://pq3d.github.io/) | 56.3 | 53.9 | 37.5 | 52.8 | 56.3 | 34.0 |
| [SceneVerse](https://scene-verse.github.io/) | 59.5 | 54.6 | 53.1 | 56.6 | 59.5 | 35.9 |

## Get Started

1. Clone Github repo.
```shell
git clone git@github.com:beacon-3d/beacon-3d.git
cd beacon-3d
```
2. Setup environment. This step can be ignored since the code only involves `numpy`, `openai`, and `tqdm`.
3. Check out [data](#data) and [evaluation](#evaluation).

## Data
The test data is in `data/{domain}`, where `{domain}` includes scannet, 3rscan, and multiscan.

**Metadata.** The metadata records grounding chains and grounding-QA chains for each object.

**Format process.** We provide scripts to convert the metadata into ScanRefer format (for grounding) and ScanQA format (for QA). We also provide the json files after this process (without `metadata` prefix) that are ready to use.
```shell
cd data

# take scannet for example

# grounding
python grounding_to_scanrefer_format.py --domain scannet --src scannet/metadata_scannet_grounding.json --dst scannet/scannet_grounding.json

# QA
python qa_to_scanqa_format.py --domain scannet --src scannet/metadata_scannet_qa.json --dst scannet/scannet_qa.json

cd ..
```


## Evaluation
**TODO before running evaluation.** Implement the `extract_pred` function in `evaluate_grounding.py` and `evaluate_qa.py` to process the raw inference results from your model. Remember to setup your OpenAI API key before running `evaluate_qa.py`.

**Run evaluation.** Run `evaluate_grounding.py` and `evaluate_qa.py`. For Grounding-QA chain analysis, you need to add a path of the processed grounding results.
```shell
# take scannet for example

# grounding
python evaluate_grounding.py --infer ${inference_results_path} --output ${processed_results_path} --data data/scannet/scannet_grounding.json --metadata data/scannet/metadata_scannet_grounding.json

# QA
python evaluate_qa.py --infer ${inference_results_path} --output ${processed_results_path} --data data/scannet/scannet_qa.json --metadata data/scannet/metadata_scannet_qa.json

# QA (with GQA-Chain analysis)
python evaluate_qa.py --infer ${inference_results_path} --output ${processed_results_path} --data data/scannet/scannet_qa.json --metadata data/scannet/metadata_scannet_qa.json --grounding ${processed_grounding_results_path}
```


## BibTex
```bibtex
@inproceedings{huang2025unveiling,
  title={Unveiling the Mist over 3D Vision-Language Understanding: Object-centric Evaluation with Chain-of-Analysis},
  author={Huang, Jiangyong and Jia, Baoxiong and Wang, Yan and Zhu, Ziyu and Linghu, Xiongkun and Li, Qing and Zhu, Song-Chun and Huang, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
