# Transformers are Sample Efficient World Models (IRIS)

[Transformers are Sample Efficient World Models](https://arxiv.org/abs/2209.00588) <br>
[Vincent Micheli](https://vmicheli.github.io)\*, [Eloi Alonso](https://eloialonso.github.io)\*, [François Fleuret](https://fleuret.org/francois/) <br>
\* Denotes equal contribution


<div align='center'>
  IRIS agent after 100k environment steps, i.e. two hours of real-time experience
  <img alt="IRIS playing on Asterix, Boxing, Breakout, Demon Attack, Freeway, Gopher, Kung Fu Master, Pong" src="assets/iris.gif">
</div>

**tl;dr**

- IRIS is a data-efficient agent trained over millions of imagined trajectories in a world model.
- The world model is composed of a discrete autoencoder and an autoregressive Transformer.
- Our approach casts dynamic learning as a sequence modelling problem, where the autoencoder builds a language of image tokens and the Transformer composes that language over time.



## BibTeX

If you find this code or paper useful, please use the following reference:

```
@article{iris2022,
  title={Transformers are Sample Efficient World Models},
  author={Micheli, Vincent and Alonso, Eloi and Fleuret, François},
  journal={arXiv preprint arXiv:2209.00588},
  year={2022}
}
```

## Setup

- Install [PyTorch](https://pytorch.org/get-started/locally/) (torch and torchvision). Code developed with torch==1.11.0 and torchvision==0.12.0.
- Install [other dependencies](requirements.txt): `pip install -r requirements.txt`
- Warning: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

## Launch a training run

```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.device=cuda:0 wandb.mode=online
```

By default, the logs are synced to [weights & biases](https://wandb.ai), set `wandb.mode=disabled` to turn it off.

## Configuration

- All configuration files are located in `config/`, the main configuration file is `config/trainer.yaml`.
- The simplest way to customize the configuration is to edit these files directly.
- Please refer to [Hydra](https://github.com/facebookresearch/hydra) for more details regarding configuration management.

## Run folder

Each new run is located at `outputs/YYYY-MM-DD/hh-mm-ss/`. This folder is structured as:

```txt
outputs/YYYY-MM-DD/hh-mm-ss/
│
└─── checkpoints
│   │   last.pt
|   |   optimizer.pt
|   |   ...
│   │
│   └─── dataset
│       │   0.pt
│       │   1.pt
│       │   ...
│
└─── config
│   |   trainer.yaml
|
└─── media
│   │
│   └─── episodes
│   |   │   ...
│   │
│   └─── reconstructions
│   |   │   ...
│
└─── scripts
|   |   eval.py
│   │   play.sh
│   │   resume.sh
|   |   ...
|
└─── src
|   |   ...
|
└─── wandb
    |   ...
```

- `checkpoints`: contains the last checkpoint of the model, its optimizer and the dataset.
- `media`:
  - `episodes`: contains train / test / imagination episodes for visualization purposes.
  - `reconstructions`: contains original frames alongside their reconstructions with the autoencoder.
- `scripts`: **from the run folder**, you can use the following three scripts.
  - `eval.py`: Launch `python ./scripts/eval.py` to evaluate the run.
  - `resume.sh`: Launch `./scripts/resume.sh` to resume a training that crashed.
  - `play.sh`: Tool to visualize some interesting aspects of the run.
    - Launch `./scripts/play.sh -a` to watch the agent play live in the environment. The left panel displays the original environment, and the right panel shows what the agent actually sees through its discrete autoencoder.
    - Launch `./scripts/play.sh -w` to unroll live trajectories with your keyboard inputs (i.e. to play in the world model). Note that for faster interaction, the memory of the Transformer is flushed every 20 frames.
    - Launch `./scripts/play.sh` to visualize the episodes contained in `media/episodes`.

## Results notebook

The folder `results/data/` contains raw scores (for each game, and for each training run) for IRIS and the baselines.

Use the notebook `results/results_iris.ipynb` to reproduce the figures from the paper.

## Credits

- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
- [https://github.com/karpathy/minGPT](https://github.com/karpathy/minGPT)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
