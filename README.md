## Environment
I set the environment to run with GATr smoothly.
1. Clone GATr.

```bash
git clone https://github.com/Qualcomm-AI-research/geometric-algebra-transformer
```
2. Clone ga_mdm as a submodule
```bash
cd geometric-algebra-transformer
git submodule add https://github.com/yujuisu/ga-mdm.git
```
3. Build the Docker image (Here I use vscode devcontainer)
```bash
cp -r ga-mdm/.devcontainer .devcontainer
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash
```
4. 

## Data Processing
## Training