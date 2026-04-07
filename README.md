# UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting

![UniRelight](assets/unirelight_teaser.jpeg)

[Kai He](https://www.cs.toronto.edu/~hekai/), [Ruofan Liang](https://www.cs.toronto.edu/~ruofan/), [Jacob Munkberg](https://research.nvidia.com/person/jacob-munkberg), [Jon Hasselgren](https://research.nvidia.com/person/jon-hasselgren), [Nandita Vijaykumar](https://www.cs.toronto.edu/~nandita/), [Alexander Keller](https://research.nvidia.com/person/alex-keller), [Sanja Fidler](https://www.cs.toronto.edu/~fidler/), [Igor Gilitschenski<sup>†</sup>](https://www.gilitschenski.org/igor/), [Zan Gojcic<sup>†</sup>](https://zgojcic.github.io/), [Zian Wang<sup>†</sup>](https://www.cs.toronto.edu/~zianwang/)

<sup>†</sup> indicates joint advising

**[Paper](https://arxiv.org/abs/2506.15673) | [Project Page](https://research.nvidia.com/labs/toronto-ai/UniRelight/)**

**Overview.**
UniRelight is a relighting framework that jointly models the distribution of scene intrinsics and illumination. It enables high-quality relighting and intrinsic decomposition from a single input image or video, producing temporally consistent shadows, reflections, and transparency, and outperforms state-of-the-art methods.

## Installation

### Conda environment 

The below commands create the `cosmos-predict1` conda environment and install the dependencies for inference:
```bash
# Create the cosmos-predict1 conda environment.
conda env create --file cosmos-predict1.yaml
# Activate the cosmos-predict1 conda environment.
conda activate cosmos-predict1
# Install the dependencies.
pip install -r requirements.txt
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0
# Install Apex for inferencing with bfloat16.
git clone https://github.com/NVIDIA/apex
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex
```

If the [dependency](https://github.com/NVlabs/nvdiffrast/blob/main/docker/Dockerfile) is well taken care of, install `nvdiffrast` with:
```bash
# Patch dependency for nvdiffrast 
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/triton/backends/nvidia/include/crt $CONDA_PREFIX/include/
pip install git+https://github.com/NVlabs/nvdiffrast.git
```
For platforms other than ubuntu, check [nvdiffrast official documentation](https://nvlabs.github.io/nvdiffrast/) and their [Dockerfile](https://github.com/NVlabs/nvdiffrast/blob/main/docker/Dockerfile). 

### Download model weights

The model weights are available on [Hugging Face](https://huggingface.co/nvidia/UniRelight).

1. Generate a [Hugging Face](https://huggingface.co/settings/tokens) access token (if you haven't done so already). Set the access token to `Read` permission (default is `Fine-grained`).

2. Log in to Hugging Face with the access token:
   ```bash
   huggingface-cli login
   ```

3. Download the UniRelight model weights from [Hugging Face](https://huggingface.co/nvidia/UniRelight):
   ```bash
   CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_unirelight_checkpoints.py --checkpoint_dir checkpoints
   ```

## Running inference  

By default, our inference script groups image frames in the same folder as one video sample.
For example, one possible way to organize the videos can be
```
inference_input_dir/
├── video_1/
│   ├── 00000.rgb.jpg
│   ├── 00001.rgb.jpg
│   ├── 00002.rgb.jpg
│   ├── ...
│   └── 00056.rgb.jpg
│
├── video_2/...
...
└── video_n/...
```

The inference script can run with the following command:
```
python -m cosmos_predict1.diffusion.inference.single_gpu_relight \
        --config_file ${CFG1} \
        --config ${CFG2} \
        --ckpt_path ${CHECKPOINT} \
        --dataset_name ${INPUT_DIR} \
        --output_path ${OUT_DIR} \
        --seed ${SEED} \
        --sample_n_frames ${N_FRAMES} \
        --env_light_path ${ENV_LIGHT_DIR} \
        --rotate_light ${ROTATE_LIGHT} \
```
- `${CFG1}`: the path to config file.
- `${CFG2}`: the path to the experiment config file.
- `${CHECKPOINT}`: the path to the checkpoint.
- `${INPUT_DIR}`: the path to input directory.
- `${OUT_DIR}`: the path to the output directory.
- `${SEED}`: the number of random seed.
- `${N_FRAMES}`: the number of inference frames.
- `${ENV_LIGHT_DIR}`: the path to environment maps (HDRIs) directory.
- `${ROTATE_LIGHT}`: true/false indicates if we rotate the lights.

### Video examples:

#### Extract frames from videos
```
python scripts/dataproc_extract_frames_from_video.py \
        --input_folder assets/examples/video_examples/ \
        --output_folder assets/examples/video_frames_examples/ \
        --frame_rate 24 --resize 1280x704 --max_frames=57
```

#### Relighting of videos
```
python -m cosmos_predict1.diffusion.inference.single_gpu_relight \
        --config_file cosmos_predict1/diffusion/training/config/config_relight.py \
        --config unirelight_I2V_f57_480p \
        --ckpt_path checkpoints/UniRelight/model.pt \
        --dataset_name assets/examples/video_frames_examples/ \
        --output_path ./outputs/video_relighting \
        --sample_n_frames 57 \
        --env_light_path assets/examples/hdri_examples
```

We can also use a static frame and show relighting with a rotating environment light by specifying `--rotate_light true --use_fixed_frame_ind true`.
```
python -m cosmos_predict1.diffusion.inference.single_gpu_relight \
        --config_file cosmos_predict1/diffusion/training/config/config_relight.py \
        --config unirelight_I2V_f57_480p \
        --ckpt_path checkpoints/UniRelight/model.pt \
        --dataset_name assets/examples/video_frames_examples/ \
        --output_path ./outputs/video_relighting_rotation \
        --sample_n_frames 57 \
        --env_light_path assets/examples/hdri_examples \
        --rotate_light true \
        --use_fixed_frame_ind true
```

## License

This project is licensed under NVIDIA OneWay Noncommercial License. See the [LICENSE](LICENSE) file for details.

## Acknowledgment

This project is built upon [NVIDIA Cosmos World Foundation Models](https://www.nvidia.com/en-us/ai/cosmos/). Some codes are adopted from [DiffusionRenderer](https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/). We thank all the authors for their impressive works.

## Citation

If you find this work useful, please consider citing:
```bibtex
@article{he2025unirelight,
  title={UniRelight: Learning Joint Decomposition and Synthesis for Video Relighting},
  author={He, Kai and Liang, Ruofan and Munkberg, Jacob and Hasselgren, Jon and Vijaykumar, Nandita and Keller, Alexander and Fidler, Sanja and Gilitschenski, Igor and Gojcic, Zan and Wang, Zian},
  journal={arXiv preprint arXiv:2506.15673},
  year={2025}
}
```