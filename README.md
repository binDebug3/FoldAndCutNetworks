# Origami Network

Origami Network is a novel neural network architecture designed to transform high-dimensional data with efficient learning and reduced complexity. Inspired by principles of geometric folding, this model employs a custom **fold layer** to reshape data through learned hyperplanes, allowing for non-linear transformations that improve prediction speed and convergence.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Research & Development](#research--development)
- [Contributing](#contributing)
- [License](#license)
- [Workflow](#workflow)

## Overview

The Origami Network architecture introduces a data transformation process similar to folding in origami, where input data is sequentially mapped across high-dimensional hyperplanes. This process enables:
- A reduced need for large, high-parameter layers in standard architectures.
- Accelerated prediction through **hyperplane-based data folding**.
- Simplified optimization, using fewer trainable parameters to capture complex data patterns.

## Features
- **Custom Fold Layer**: Each layer learns a set of hyperplanes to reshape the data dynamically, reducing training time.
- **Optimized Performance**: Through reduced parameterization, Origami Networks achieve lower latency.
- **Configurable Optimizers**: Support for various optimizers to best match the model's fold-based architecture.
- **Improved Memory Efficiency**: Fewer parameters make Origami Networks more memory and parameter efficient.
- **Interpretability**: Fold operations provide a more intuitive and interpretable alternative to traditional deep learning models.

## Architecture
The Origami Network builds on the idea of **soft folds** and **hard folds**, which involve using hyperplanes to partition and reshape the input data iteratively. The fold layers are inspired by origami and the Fold and Cut Theorem to emulate a ReLU function but add the capability to fold data into more separable forms in fewer steps.

## Installation
To install and set up the Origami Network repository, ensure you have the following prerequisites:

```bash
git clone https://github.com/yourusername/FoldAndCutNetworks.git
cd FoldAndCutNetworks
pip install -r requirements.txt
```

## Usage
To train the Origami Network, specify the model and dataset parameters as follows:

```python
from models.model_bank import OrigamiFold4
from models.training import *
# define x_train, y_train, x_test, y_test
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = load_data(x_train, y_train)
val_loader = load_data(x_test, y_test)
model = OrigamiFold4(x_train.shape[1])
train(model, optimizer, train_loader, val_loader, epochs=200, verbose=1)
```
Refer to the ipynb files titled 'd-name.ipynb' for more usage and training examples.

## Research and Development
The Origami Network architecture is under active research, focusing on:
- Experimenting with different fold depth and width configurations.
- Testing efficiency gains in prediction for natural language processing, computer vision, and other domains.
- Identifying optimizers that best support fold-layer dynamics.
- Experimenting with folds in higher dimensions.
- Developing a fold version of convolutional neural networks.

## Contributing
Contributions to improve the Origami Network, fix bugs, or add features are welcome! Please open an issue or submit a pull request.
Current Contributors:

Sam Layton

[![LinkedIn][linkedin-icon]][linkedin-url2]
[![GitHub][github-icon]][github-url2]
[![Email][email-icon]][email-url2]

Dallin Stewart - dallinpstewart@gmail.com

[![LinkedIn][linkedin-icon]][linkedin-url1]
[![GitHub][github-icon]][github-url1]
[![Email][email-icon]][email-url1]

Jeddy Bennett

[![LinkedIn][linkedin-icon]][linkedin-url3]
[![GitHub][github-icon]][github-url3]
[![Email][email-icon]][email-url3]

## License
This project is licensed under the MIT License. See the LICENSE file for details.


## Workflow

This is to make sure that the main repo has all the necessary changes and you continually get all of the updates on your end.

##### Sync your Fork with the Source

Open command prompt (or git bash) and cd into your repository folder.
Run `git branch` to check your current branch.
If a star appears next to `main`, you are on the default branch, called main.

```bash
git pull upstream main               # Get updates from the source repo.
git push origin main                 # Push updates to your fork.
```
##### Make Edits

1. Create a new branch for editing.
```bash
git checkout -b newbranch               # Make a new branch and switch to it. Pick a good branch name.
```
**Only make new branches from the `develop` branch** (when you make a new branch with `git branch`, it "branches off" of the current branch).
To switch between branches, use `git checkout <branchname>`.

2. Make edits to the labs, saving your progress at reasonable segments.
```bash
git add filethatyouchanged
git commit -m "<a DESCRIPTIVE commit message>"
```
3. Push your working branch to your fork once you're done making edits.
```bash
git push origin newbranch               # Make sure the branch name matches your current branch
```
4. Create a pull request.
Go to the page for this repository.
Click the green **New Pull Request** button.

##### Clean Up

After your pull request is merged, you need to get those changes (and any other changes from other contributors) into your `develop` branch and delete your working branch.
If you continue to work on the same branch without deleting it, you are risking major merge conflicts.

1. Update the `main` branch.
```bash
git checkout main               # Switch to main.
git pull origin main            # Pull changes from the source repo.
```
2. Delete your working branch. **Always do this after (and only after) your pull request is merged.**
```bash
git checkout main               # Switch back to develop.
git branch -d newbranch         # Delete the working branch.
git push origin :newbranch      # Tell your fork to delete the example branch.
```


[linkedIn-icon]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedIn-url1]: https://www.linkedin.com/in/dallinstewart/
[linkedIn-url2]: https://www.linkedin.com/in/
[linkedIn-url3]: https://www.linkedin.com/in/jeddy-bennett/


[github-icon]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[github-url1]: https://github.com/binDebug3
[github-url2]: https://github.com/
[github-url3]: https://github.com/jeddybennett

[Email-icon]: https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white
[Email-url1]: mailto:dallinpstewart@gmail.com
[Email-url2]: mailto:dallinpstewart@gmail.com
[Email-url3]: mailto:jeddybennett01@gmail.com
