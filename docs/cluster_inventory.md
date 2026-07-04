# Cluster Inventory

Generated: 2026-07-04T02:34:44.856490+00:00

Strict cluster verification passed after remote setup. `scripts/verify_env.py --quick-smoke true`
previously passed on `ripper`, `foureyes`, and `ai`, including official Mamba3 MIMO TileLang BF16
forward/backward, exact `mamba-ssm`/`muon-optimizer` source commit checks, official Muon
optimizer smoke, and `fastfeedforward.FFF` smoke. GitHub SSH clone from the remote hosts failed
with `publickey` authentication errors, so the repo was synced to `/home/catid/fffdistill` by
`rsync` from `work`.

Current T16 occupancy check: `foureyes` GPUs 2 and 3 reported about 15 GiB free and 100%
utilization, so they are treated as occupied for smoke/profiling launches until that clears.

| machine | host | role | workdir | expected GPUs | detected GPUs | available | GPU summary |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| work | localhost | local | /home/catid/fff | 2 | 2 | True | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, 97247 MiB<br>NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, 97247 MiB |
| ripper | ripper | remote | /home/catid/fffdistill | 4 | 4 | True | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97247 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97247 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97247 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97247 MiB |
| foureyes | foureyes | remote | /home/catid/fffdistill | 4 | 4 | True | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97249 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 97249 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 15029 MiB<br>NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB, 15029 MiB |
| ai | ai | remote | /home/catid/fffdistill | 2 | 2 | True | NVIDIA GeForce RTX 5090, 32607 MiB, 32146 MiB<br>NVIDIA GeForce RTX 5090, 32607 MiB, 32146 MiB |
