# FoldAndCutNetworks
A new deep learning layer and architecture based on the intersection of the fold and cut theorem and ReLu activation functions

### Workflow

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


