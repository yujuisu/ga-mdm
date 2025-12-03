## Environment
I set the environment to run with GATr (requires docker or a pc with ubuntu 20.04). Otherwise you just need kingdon and torch.
1. Clone GATr.

```bash
git clone https://github.com/Qualcomm-AI-research/geometric-algebra-transformer
```
2. Clone ga_mdm as a submodule
```bash
cd geometric-algebra-transformer
git submodule add https://github.com/yujuisu/ga-mdm.git
```
3. Build the Docker image (using [devcontainer](https://code.visualstudio.com/docs/devcontainers/devcontainer-cli))
```bash
cp -r ga-mdm/.devcontainer .devcontainer
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash
```
4. Download [/test_data](https://drive.google.com/drive/folders/1oodEd9QcgN8sVvtXP13Vr6PtaXrcHrZh?usp=sharing) 
Put it under /ga_mdm
5. Upzip texts.zip
6. Test training:
```bash
python train_ga_mdm.py
```


## Data Processing
Download the [SMPL model](https://drive.google.com/file/d/13gsD8FCqZNsA-0YHNRYnJBixjV7uSThP/view?usp=sharing)
Follow the steps of the [notebook](process_data/amass_motor_bivector_263.ipynb)
## Training