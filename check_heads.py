import os
import torch

base = r"D:\Birds-classification\Birds-classification"

files = [
    r"myapp\swin_b_birds_224.pth",
    r"myapp\swin_model_best.pth",
    r"training_output\best_model.pth",
    r"training_output\last_model.pth",
    r"training_output_smoke\last_model.pth",
    r"training_output_smoke\swin_b_birds_224.pth",
]

for relative in files:
    path = os.path.join(base, relative)
    print("\n" + "=" * 60)
    print(relative)

    if not os.path.isfile(path):
        print("NOT FOUND")
        continue

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)

        if isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and (
                    key.endswith("head.fc.weight")
                    or key.endswith("head.fc.bias")
                    or key.endswith("classifier.weight")
                    or key.endswith("classifier.bias")
                    or key.endswith("head.weight")
                    or key.endswith("head.bias")
                ):
                    print(key, tuple(value.shape))
        else:
            print("Unexpected checkpoint format:", type(state).__name__)

    except Exception as error:
        print("Could not inspect:", error)
