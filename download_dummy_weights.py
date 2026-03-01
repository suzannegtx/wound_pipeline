from pathlib import Path
import torch

def main() -> None:
    weights = {
        "wound_deeplabv3_r50.pth": {},
        "tissue_seg.pth": {},
        "wound_classifier.pth": {},
    }
    Path("weights").mkdir(exist_ok=True)
    for name, state in weights.items():
        path = Path("weights") / name
        torch.save(state, path)
        print(f"Created dummy weight: {path}")

if __name__ == "__main__":
    main()
