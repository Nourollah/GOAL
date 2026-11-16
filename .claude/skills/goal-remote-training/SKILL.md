---
name: goal-remote-training
description: Runs GOAL training, evaluation and MD jobs on a remote GPU workstation or cluster over SSH. Reads host details from a local file outside the repository, syncs code without environments, data or logs, installs the pixi environment remotely, picks a free GPU, launches detached runs, monitors them, fetches results, and applies the precision rule (float64 only on datacenter GPUs). Use when the user asks to run, check or collect a job on their GPU machine, a remote host or a cluster.
license: MIT
---

# Remote GPU runs

## Host details stay out of the repository

Read `~/.config/goal/hosts.yaml`. If it is missing, ask the user for each host's SSH alias (from `~/.ssh/config`), GOAL path, GPUs and pixi environment, and offer to write the file. Never write host names, addresses, user names or home paths into the repository, logs or commit messages.

```yaml
# ~/.config/goal/hosts.yaml (local only, never committed)
hosts:
  - alias: workstation          # a Host entry in ~/.ssh/config
    goal_path: ~/Projects/GOAL
    gpus: [RTX 6000 Ada 48GB]
    pixi: pixi                  # or a full path such as ~/.pixi/bin/pixi
    env: cuda
  - alias: gpu-box
    goal_path: ~/Projects/GOAL
    gpus: [RTX 5090 32GB, RTX 5090 32GB]
    pixi: pixi
    env: cuda
fp64_hosts: []                  # aliases of A100, H100 or H200 machines
```

## Sync

- Preferred: commit, push a branch, and `git pull` on the host.
- Otherwise rsync without environments, data or outputs, dry run first:

```bash
rsync -azn --exclude .pixi/ --exclude logs/ --exclude outputs/ --exclude data/ --exclude wandb/ ./ <alias>:<goal_path>/
rsync -az  --exclude .pixi/ --exclude logs/ --exclude outputs/ --exclude data/ --exclude wandb/ ./ <alias>:<goal_path>/
```

  Do not add `--delete` unless the user asks for it.
- First use of a host: `ssh <alias> 'cd <goal_path> && <pixi> install -e <env>'`.

## Launch

```bash
ssh <alias> nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv    # pick a free GPU
ssh <alias> "cd <goal_path> && mkdir -p logs && tmux new -d -s <run_name> \
  'CUDA_VISIBLE_DEVICES=<gpu> <pixi> run -e <env> goal-train --config-name <experiment> run_name=<run_name> seed=<seed> 2>&1 | tee logs/<run_name>.log'"
```

- Without tmux, use `nohup ... > logs/<run_name>.log 2>&1 &`.
- One run per GPU for seeds and ablation arms; DDP (`++strategy.name=ddp trainer.devices=2`; see `goal-train-model`) only when one run needs both GPUs.
- Names must show the stage, the arm and the seed.

## Monitor and collect

```bash
ssh <alias> "tail -n 50 <goal_path>/logs/<run_name>.log"
ssh <alias> "cd <goal_path> && <pixi> run -e <env> goal-inspect logs/<task>/runs/<run>/checkpoints/"
rsync -az <alias>:<goal_path>/logs/<task>/runs/<run>/ logs/<task>/runs/<run>/
```

Fetch only what is needed (configs, metrics, the best archive); full checkpoint pools are large.

## Precision

Consumer and workstation GPUs (RTX 5090, RTX 6000 Ada) run float64 at about 1/64 of their float32 rate. Run float64 training and L-BFGS polishing only on `fp64_hosts` (A100, H100, H200).

## Safety

- Never delete or overwrite remote run directories, and never kill processes you did not start, without asking.
- On shared machines, check `nvidia-smi` for other users' jobs before launching.

## Done when

- [ ] The run appears in `tmux ls` (or the scheduler queue) and its log shows training steps.
- [ ] The results needed locally have been fetched.
