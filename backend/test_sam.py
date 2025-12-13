import torch
from models.sam_engine import SAMSegmentationEngine

print("=" * 50)
print("🧪 SAM Installation Test")
print("=" * 50)

# Check CUDA
print(f"\n1️⃣ CUDA Status:")
print(f"   Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Initialize SAM
print(f"\n2️⃣ Initializing SAM...")
try:
    engine = SAMSegmentationEngine(model_type="vit_b", device="cpu  ")
    print("   ✅ SAM Engine Ready!")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 50)
print("✅ Installation test complete!")
print("=" * 50)

