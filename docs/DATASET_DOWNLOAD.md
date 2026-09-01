# Bearing-UAV-90K download and conversion

The dataset is **not included in this Git repository**. Download the original
Bearing-UAV-90K files from the authors' official Hugging Face dataset page:

- Dataset: https://huggingface.co/datasets/HaoyZhou/bearinguav/tree/main
- Official code and dataset description: https://github.com/liukejia121/bearinguav

Follow the license and usage terms published by the dataset authors. After
extracting the archive, the input directory must be named `Bearing_UAV_90K`
and contain `citya`, `cityb`, `cityc`, `cityd`, and `city_rsi`.

## Convert Bearing-UAV-90K to UAV90K

From this repository root on Windows:

```powershell
python scripts/build_uav90k.py `
  --source D:\datasets\Bearing_UAV_90K `
  --output D:\datasets\UAV90K `
  --link-mode hardlink `
  --seed 42

python scripts/validate_uav90k.py --dataset D:\datasets\UAV90K
python scripts/fingerprint_dataset.py --dataset D:\datasets\UAV90K
```

On Linux, use `--link-mode symlink` or `--link-mode copy` if hard links are not
appropriate. The conversion is non-destructive and idempotent.

The expected dataset fingerprint for the frozen seed-42 split is:

```text
fdaf3f74fed5e8fa953152a7d4c7bfbd02f0becdf1ff89a99dace52c3ad4001a
```

Expected counts are 90,000 queries, 1,024 unique satellite tiles, and a
76,500 / 4,500 / 9,000 train/validation/test split. The generated images,
maps, `samples.csv`, and split text files remain ignored by Git.
