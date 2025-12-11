## Getting Start
I set the environment to run with GATr (requires docker or a pc with ubuntu 20.04). Otherwise you just need Kingdon, Torch and Transformers.
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

And download the [SMPL model](https://drive.google.com/file/d/13gsD8FCqZNsA-0YHNRYnJBixjV7uSThP/view?usp=sharing)
Put it under /serialized

5. Upzip texts.zip
6. Test training:
```bash
python train_ga_mdm.py
```

## Data Processing
Follow the steps of the [notebook](process_data/amass_motor_bivector_263.ipynb)
## Training
To monitor gradient on tensorboard:
```bash
export MONITOR_GRADS=1
```
To try the autoregressive version:
```bash
export AUTOREG=1
```


## Directions
1. Build CGENN's geometric product layer and replace the linear layer to catch multilinearity and nonlinearity of powers.
2. Replace the transformer with GATr
3. Instead of GATr's full 16-dim multivector, use the 8-dim even subalgebra.
4. Interaction with geometric objects (force as lines, floor as a plane, and left/right as a mirroring)
5. Prove the graph positional encoding helps carry informations of the kinematic chain