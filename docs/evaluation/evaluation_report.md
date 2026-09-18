# Independent test evaluation

- Model: `models/best.pt`
- Dataset: `data/fire.yaml`
- Split: `test`
- Images: 1,704
- Device: CUDA device 0, NVIDIA GeForce RTX 5060 Ti
- Error-example confidence: 0.25
- Error matching: same class and IoU >= 0.50
- Source dataset unchanged during the completed evaluation: yes

## Validation and independent test comparison

| Metric | Validation | Test | Test - validation |
|---|---:|---:|---:|
| Precision | 0.557 | 0.486 | -0.071 |
| Recall | 0.540 | 0.642 | +0.102 |
| mAP50 | 0.524 | 0.555 | +0.031 |
| mAP50-95 | 0.242 | 0.278 | +0.036 |
| fire mAP50-95 | 0.223 | 0.289 | +0.066 |
| smoke mAP50-95 | 0.260 | 0.267 | +0.007 |

## Test class metrics

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| fire | 0.485 | 0.576 | 0.573 | 0.289 |
| smoke | 0.486 | 0.708 | 0.537 | 0.267 |

## Error analysis at confidence 0.25

| Item | Count |
|---|---:|
| Ground-truth objects | 949 |
| Predicted objects | 1,112 |
| False-positive objects | 539 |
| False-negative objects | 376 |
| Images with at least one false positive | 391 |
| Images with at least one false negative | 318 |
| Negative images | 942 |
| Negative images with no detection | 941 |

The saved examples are deterministic qualifying test images, not random copies. Cyan boxes prefixed with `GT` are ground truth; model predictions include the class name and confidence. The black header records the ground-truth, prediction, false-positive, and false-negative counts.

## Artifacts

- `metrics.json`: complete aggregate and class metrics
- `metrics/`: confusion matrices, PR/F1/precision/recall curves, and prediction batches
- `samples/`: test prediction examples
- `false_positives/`: actual prediction/ground-truth comparison examples
- `false_negatives/`: actual prediction/ground-truth comparison examples
- `undetected_negatives/`: ground-truth-negative images with no model detection
- `example_manifest.json`: source and count provenance for every saved example
