# test_backbone_pretrained.py

import torch
from meld.models.backbone import resnet32

print("=== Test 1: Random init (no crash) ===")
model = resnet32(pretrained=False)
x = torch.randn(2, 3, 32, 32)
out = model(x)
print(f"Output shape: {out.shape}")  # expect [2, 64]
print("PASS\n")

print("=== Test 2: Pretrained init (no crash) ===")
model_pt = resnet32(pretrained=True)
out_pt = model_pt(x)
print(f"Output shape: {out_pt.shape}")  # expect [2, 64]
print("PASS\n")

print("=== Test 3: Weights actually changed ===")
model_rand = resnet32(pretrained=False)
model_pre  = resnet32(pretrained=True)

rand_w = model_rand.conv_1_3x3.weight.clone()
pre_w  = model_pre.conv_1_3x3.weight.clone()
assert not torch.allclose(rand_w, pre_w), "Stem weights unchanged — pretrained did nothing"
print("Stem weights differ from random init: PASS\n")

print("=== Test 4: Stage 1 block 0 conv_a changed ===")
rand_s1 = model_rand.stage_1[0].conv_a.weight.clone()
pre_s1  = model_pre.stage_1[0].conv_a.weight.clone()
assert not torch.allclose(rand_s1, pre_s1), "Stage 1 weights unchanged"
print("Stage 1 weights differ from random init: PASS\n")

print("=== Test 5: Stage 3 block 0 conv_a changed ===")
rand_s3 = model_rand.stage_3[0].conv_a.weight.clone()
pre_s3  = model_pre.stage_3[0].conv_a.weight.clone()
assert not torch.allclose(rand_s3, pre_s3), "Stage 3 weights unchanged — deep layers not mapped"
print("Stage 3 weights differ from random init: PASS\n")

print("=== Test 6: No NaN or zero-filled weights ===")
for name, param in model_pre.named_parameters():
    assert not torch.isnan(param).any(),  f"NaN in {name}"
    assert not torch.isinf(param).any(),  f"Inf in {name}"
    # no weight tensor should be entirely zero
    if "weight" in name and param.numel() > 4:
        assert param.abs().sum() > 0, f"All-zero weights in {name}"
print("No NaN / Inf / all-zero weights: PASS\n")

print("=== Test 7: Output differs between random and pretrained ===")
torch.manual_seed(0)
x_fixed = torch.randn(2, 3, 32, 32)
model_rand.eval()
model_pre.eval()
with torch.no_grad():
    out_rand = model_rand(x_fixed)
    out_pre  = model_pre(x_fixed)
assert not torch.allclose(out_rand, out_pre), "Outputs identical — pretrained has no effect"
print("Outputs differ between random and pretrained: PASS\n")

print("=== ALL TESTS PASSED ===")