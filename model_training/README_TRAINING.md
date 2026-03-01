# DeepLabV3 - FUSeg (Foot Ulcer Segmentation Challenge)

## Lancer l'entraînement
python train_deeplabv3_fuseg.py --data_dir "data/Foot Ulcer Segmentation Challenge" --epochs 50 --batch_size 8 --lr 1e-4 --img_size 512 --device auto --amp --num_workers 4

## Sorties
- checkpoints : download_dummy_weights/deeplabv3_fuseg_best.pth (meilleur, clé = dice puis precision), deeplabv3_fuseg_last.pth
- logs : outputs/training_log.csv
- visuels : outputs/vis/epochXXX_idxY.overlay.png (vert=GT, rouge=prediction), gt/pred bruts
- résumé : outputs/metrics_summary.json

## Structure attendue du dataset
data/Foot Ulcer Segmentation Challenge/
  train/images/*.png
  train/labels/*.png   (0/1 ou 0/255)
  validation/images/*.png
  validation/labels/*.png
  test/images/*.png    (labels optionnels)

## Notes
- Les métriques sont calculées en global sur tous les pixels (TP/FP/FN agrégés).
- Les masques sont binarisés >127.
- `--save_every` pour sauvegarder régulièrement, `--vis_every` pour la fréquence des visuels.
