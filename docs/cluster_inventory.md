# Cluster Inventory

Generated: 2026-07-04T04:40:59.489382+00:00

## Current Caveats

- Strict cluster verification sees all 12 configured GPUs and project venvs, but detached scheduler preflight found `ripper`, `foureyes`, and `ai` workdirs still at commit `eb67a76a3a2831c70641d76d7ff04d8ffe53bb0d` while `work` expected `6f35d8c1eb7efde0ad1712d32a690743f35045d8`. Sync remote workdirs to the latest pushed commit before launching remote teacher jobs.
- A 12-slot metadata scheduler attempt before preflight hardening launched successfully on local `work` slots only; remotes rejected `--smoke-mode metadata` from stale `train_teacher.py`. The scheduler now classifies this as `failed_infra` during preflight instead of launching stale remote code.
- `foureyes` GPUs 2 and 3 report about 15 GiB free, so memory-heavy real jobs should treat those as busy until a fresh inventory shows they have cleared. `ai` has 2x RTX 5090 GPUs with about 32 GiB each, so batch-size caps should be separate from the 95 GiB Pro6000 class.
- Earlier remote setup verified official Mamba3/Muon/fastfeedforward environments on `ripper`, `foureyes`, and `ai`. GitHub SSH from the remote hosts previously failed with `publickey` errors, so remote repo sync may need rsync from `work` or an explicitly authorized git reset after credentials are fixed.

| machine | host | role | workdir | expected GPUs | detected GPUs | available | GPU summary |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| work | localhost | local | /home/catid/fff | 2 | 2 | True | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, 97247 MiB<br>NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, 97247 MiB |
| ripper | ripper | remote | /home/catid/fffdistill | 4 | 4 | True | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97247 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97247 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97247 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97247 MiB |
| foureyes | foureyes | remote | /home/catid/fffdistill | 4 | 4 | True | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97249 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97249 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 15029 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 15029 MiB |
| ai | ai | remote | /home/catid/fffdistill | 2 | 2 | True | NVIDIA GeForce RTX 5090, 32607 MiB, 32146 MiB<br>NVIDIA GeForce RTX 5090, 32607 MiB, 32146 MiB |
