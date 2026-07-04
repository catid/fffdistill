# Cluster Inventory

Generated: 2026-07-04T18:00:01.610058+00:00

## Current Caveats

- Strict cluster verification sees all 12 configured GPUs and project venvs.
- GitHub SSH from `ripper`, `foureyes`, and `ai` still fails with `publickey`
  errors, so remote repo updates use `scripts/sync_repo_remote.sh` rsync fallback
  from `work` after commits are pushed.
- Before launching jobs, scheduler preflight must require the expected local git
  commit and record expected/local/remote commit hashes. Stale remote workdirs
  fail as infrastructure errors, and detached launch scripts re-check
  `git rev-parse HEAD` immediately before running the job command so a drifted
  rsync target cannot silently enter an experiment.
- `ai` has 2x RTX 5090 GPUs with about 32 GiB each, so batch-size caps should be
  separate from the 95 GiB Pro6000 class.

| machine | host | role | workdir | expected GPUs | detected GPUs | available | GPU summary |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| work | localhost | local | /home/catid/fff | 2 | 2 | True | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, 97288 MiB<br>NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, 97288 MiB |
| ripper | ripper | remote | /home/catid/fffdistill | 4 | 4 | True | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97288 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97288 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97288 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97288 MiB |
| foureyes | foureyes | remote | /home/catid/fffdistill | 4 | 4 | True | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97288 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97288 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97288 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97288 MiB |
| ai | ai | remote | /home/catid/fffdistill | 2 | 2 | True | NVIDIA GeForce RTX 5090, 32607 MiB, 32146 MiB<br>NVIDIA GeForce RTX 5090, 32607 MiB, 32146 MiB |
