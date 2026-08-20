import torch
import torch.nn as nn


class VehicleCNN(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        # ====================================================
        # FEATURE EXTRACTION
        # ====================================================

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(p=0.4),

            nn.Linear(
                in_features=256,
                out_features=num_classes,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        # IMPORTANT:
        # Do NOT apply sigmoid here.
        #
        # BCEWithLogitsLoss expects raw logits.

        return x


# ============================================================
# MODEL SUMMARY / FORWARD PASS TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("VEHICLE CNN TEST")
    print("=" * 60)

    # Create model
    model = VehicleCNN(
        num_classes=4
    )

    print("\nMODEL")
    print("-" * 60)

    print(model)

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nPARAMETERS")
    print("-" * 60)

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    dummy_input = torch.randn(
        8,
        3,
        224,
        224
    )

    output = model(
        dummy_input
    )

    print("\nFORWARD PASS")
    print("-" * 60)

    print(
        f"Input shape  : "
        f"{dummy_input.shape}"
    )

    print(
        f"Output shape : "
        f"{output.shape}"
    )

    print(
        f"Output dtype : "
        f"{output.dtype}"
    )

    # --------------------------------------------------------
    # Verify expected shape
    # --------------------------------------------------------

    assert output.shape == (
        8,
        4
    ), (
        f"Unexpected output shape: "
        f"{output.shape}"
    )

    print(
        "\n✓ Forward pass successful"
    )

    print(
        "✓ Output contains 4 logits"
    )

    print(
        "✓ Model is ready for BCEWithLogitsLoss"
    )

    print("\n" + "=" * 60)