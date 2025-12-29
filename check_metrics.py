import torch
import json

payload = torch.load('torch_models/parallel_genre_classifier_torch.pt', map_location='cpu')
meta = payload.get('meta', {})

print('=' * 60)
print('MODEL TRAINING METRICS')
print('=' * 60)

# Check for train accuracy
train_acc = meta.get('train_accuracy', None)
val_acc = meta.get('val_accuracy', 0)
test_acc = meta.get('test_accuracy', 0)

if train_acc is not None:
    print(f"Train Accuracy: {train_acc:.2%}" if isinstance(train_acc, float) else f"Train Accuracy: {train_acc}")
else:
    print("Train Accuracy: NOT STORED IN CHECKPOINT")

print(f"Validation Accuracy: {val_acc:.2%}" if isinstance(val_acc, float) else f"Validation Accuracy: {val_acc}")
print(f"Test Accuracy: {test_acc:.2%}" if isinstance(test_acc, float) else f"Test Accuracy: {test_acc}")
print(f"Feature Type: {meta.get('feature_type', 'N/A')}")

mapping = payload.get('mapping', {})
if isinstance(mapping, dict):
    classes = ', '.join(mapping.values())
    print(f"Num Classes: {len(mapping)}")
elif isinstance(mapping, list):
    classes = ', '.join(mapping)
    print(f"Num Classes: {len(mapping)}")
else:
    classes = str(mapping)
    print(f"Num Classes: Unknown")

print(f"\nClasses: {classes}")

# Check all available metadata keys
print('\n' + '=' * 60)
print('ALL METADATA KEYS IN CHECKPOINT')
print('=' * 60)
print(f"Available keys: {', '.join(meta.keys())}")

# Check fine-tune stats
fine_tune = meta.get('fine_tune_stats', {})
if fine_tune:
    print('\n' + '=' * 60)
    print('FINE-TUNE STATS')
    print('=' * 60)
    for k, v in fine_tune.items():
        print(f"{k}: {v}")

# Pretty print all metadata
print('\n' + '=' * 60)
print('COMPLETE METADATA DUMP')
print('=' * 60)
print(json.dumps(meta, indent=2, default=str))
