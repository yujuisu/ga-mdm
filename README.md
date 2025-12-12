# Geometric Algebra Infused Motion Diffusion Model
The original goal is to beat the MDM model. But I lose interest after realizing they train on the inverse-kinematic motion. I want to train on real motion (with orientation), to have more potential on robotics, dynamics, and interaction with geometric objects.

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
4. Download [/test_data](https://drive.google.com/drive/folders/1oodEd9QcgN8sVvtXP13Vr6PtaXrcHrZh?usp=sharing) to /ga_mdm

And download the [SMPL model](https://drive.google.com/file/d/13gsD8FCqZNsA-0YHNRYnJBixjV7uSThP/view?usp=sharing) to /serialized

5. Upzip texts.zip
6. Test training:
```bash
python train_ga_mdm.py
```

## Data Processing
Follow the steps of [notebook](process_data/amass_motor_bivector_263.ipynb)

## Visualizing
Download a [trained model](https://drive.google.com/file/d/1oGA4b4rh-ofxifyV2vlHaznhTbAhe43F/view?usp=sharing)

Follow the steps of [notebook](result_visualization.ipynb)

## Training
To monitor gradient on tensorboard:
```bash
export MONITOR_GRADS=1
```
To try the autoregressive version (yet finish):
```bash
export AUTOREG=1
```
## Tutorials of Geometric Algebra on the SMPL model
Download an amass motion [npz file](https://drive.google.com/file/d/1ekkGu1YAFpPnX_MiL9nWE3UZ2yH3mitp/view?usp=sharing) to /serialized
1. [SMPL model](tutorials/amass_motor.ipynb)
2. [Blending and Animation](tutorials/amass_instaneous_lie_alg.ipynb)
3. [Torch implementations](torch_motor_utils.py) of logarithms/exponentials are carefully treated to have non-singular gradients.

## Getting Further
1. Build CGENN's geometric product layer and replace the linear layer to catch multilinearity and nonlinearity of powers.
2. Replace the transformer with GATr
3. Instead of GATr's full 16-dim multivector, use the 8-dim even subalgebra.
4. Interaction with geometric objects (force as lines, floor as a plane, and left/right as a mirroring)
5. Prove the graph positional encoding helps carry informations of the kinematic chain