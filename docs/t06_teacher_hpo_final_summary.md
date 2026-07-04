# T06 Teacher HPO Final Summary

Stage B selected an official Mamba-3 CIFAR-10 teacher from validation metrics and
then ran CIFAR-10 test evaluation exactly once for that selected checkpoint.

Selection cutoff:

- Selected checkpoint: `ripper:/home/catid/fffdistill/outputs/scheduler_hpo/teacher_hpo_wave1_20260704_0618/ripper/0/trials/trial_000000/teacher_best.pt`
- Local ignored copy for downstream distillation: `checkpoints/teacher/ripper0_val9418_test9399_teacher_best.pt`
- Checkpoint SHA-256: `f52c4774f4fc423e820b5cd018707d98a962867c49c6403895b4148dbf6af509`
- Final-eval metrics SHA-256 on `ripper`: `385ea2c8a23ecb2d0ba5b96ec16c38bf471e014370796769d71861b87f05b726`
- Test-eval output dir: `ripper:/home/catid/fffdistill/outputs/teacher_final/ripper0_val9418_20260704_0839`

Selected model/config:

- Official Mamba-3, no fallback architecture.
- Parameters: `9,053,258`.
- Shape: `d_model=192`, `depth=16`, `patch_size=2`, `d_state=64`, `headdim=64`, `is_mimo=true`, `mimo_rank=2`, `bidirectional=true`, `drop_path=0.05`.
- Training: BF16 autocast, official `SingleDeviceMuonWithAuxAdam` plus AdamW fallback groups.
- HPO overrides: `batch_size_per_gpu=512`, `lr_muon=0.009843051871459752`, `lr_adamw=0.0024129099253894473`, `weight_decay_muon=0.05`, `weight_decay_adamw=0.05`, `label_smoothing=0.1`, `mixup=0.4`, `cutmix=1.0`, `schedule=cosine`.

Validation result:

- Best validation accuracy: `0.9418`.
- Best validation epoch: `175`.
- Validation loss at selected checkpoint: `0.32254487290382383`.
- HPO trial status: `succeeded`.
- Training elapsed: `8,367.938245694852` seconds.
- Train-only throughput: `1,097.2164745917225` images/sec.
- Peak CUDA memory reserved: `25,461,522,432` bytes.

Final CIFAR-10 test result:

- Command:
  `PYTHONPATH=src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m cifar_mamba_fff.evaluate_teacher --checkpoint outputs/scheduler_hpo/teacher_hpo_wave1_20260704_0618/ripper/0/trials/trial_000000/teacher_best.pt --output-dir outputs/teacher_final/ripper0_val9418_20260704_0839 --batch-size 1024 --num-workers 8 --min-selected-val-accuracy 0.90`
- Final test accuracy: `0.9399`.
- Final test loss: `0.3235518706321716`.
- Full test steps: `10`.
- Partial evaluation: `false`.
- Test access: `true`, only after validation selection.

Notes:

- The selected checkpoint satisfies the T06 >=90% final-test target.
- The CIFAR-10 test set was not used by HPO, smoke tests, or validation selection.
- Several teacher HPO/refill jobs were still running when this checkpoint was selected; after final evaluation they should be cancelled or allowed to finish only if they do not delay downstream T13 distillation.
