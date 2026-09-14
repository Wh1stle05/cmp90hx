#!/usr/bin/env python3
"""生成对比图（LoRA 强度/训练轮次/底模）与 4K 预览，供研究报告与视频使用"""
import os
from PIL import Image, ImageDraw, ImageFont

IMG = "/data3/dual-90hx-research/results/images"
OUT = "/data3/dual-90hx-research/results/previews"
os.makedirs(OUT, exist_ok=True)

F = {
    "no_lora": "t2i_1024_no_lora_t2i_nolora_00001_.png",
    "w0.6": "lora_w0.6_lora_w06_00001_.png",
    "w1.0": "lora_w1.0_lora_w10_00001_.png",
    "w1.3": "t2i_1024_chisa_lora_t2i_lora_00001_.png",
    "epoch2": "lora_epoch2_lora_e2_00001_.png",
    "epoch6": "lora_epoch6_lora_e6_00001_.png",
    "epoch12": "t2i_1024_chisa_lora_t2i_lora_00001_.png",
    "illustrious_lora": "t2i_1024_chisa_lora_t2i_lora_00001_.png",
    "animagine": "t2i_1024_animagine_t2i_anima_00001_.png",
    "hires_src": "hires_4k_hires_src_00001_.png",
    "hires_4k": "hires_4k_hires_4k_00001_.png",
    "direct_4k": "direct_4k_direct4k_00001_.png",
    "img2img": "img2img_1mp_img2img_00001_.png",
    "inpaint": "inpaint_28_inpaint_00001_.png",
    "batch1": "batch4_1024_chisa_batch4_00001_.png",
}


def font(sz):
    for p in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def strip(keys, labels, out, cell=512, title=None):
    cells = []
    for i, k in enumerate(keys):
        im = Image.open(os.path.join(IMG, F[k])).convert("RGB")
        im.thumbnail((cell, cell), Image.LANCZOS)
        cells.append(im)
    body_h = max(im.height for im in cells)
    w = cell * len(cells)
    top = 44 if title else 0
    canvas = Image.new("RGB", (w, body_h + 34 + top), "white")
    d = ImageDraw.Draw(canvas)
    for i, im in enumerate(cells):
        canvas.paste(im, (i * cell + (cell - im.width) // 2, top))
        d.text((i * cell + 6, top + body_h + 4), labels[i], fill="black", font=font(20))
    if title:
        d.text((8, 8), title, fill="black", font=font(28))
    p = os.path.join(OUT, out)
    canvas.save(p, quality=92)
    print("saved", p, canvas.size)


def preview(key, out, width=1920):
    im = Image.open(os.path.join(IMG, F[key])).convert("RGB")
    orig = im.size
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    p = os.path.join(OUT, out)
    im.save(p, quality=92)
    print("saved", p, f"(orig {orig[0]}x{orig[1]} -> {im.size[0]}x{im.size[1]})")


strip(["no_lora", "w0.6", "w1.0", "w1.3"],
      ["no LoRA", "LoRA w=0.6", "LoRA w=1.0", "LoRA w=1.3 (workflow 默认)"],
      "compare_lora_strength.jpg", title="chisa LoRA 强度对比（同种子 1234567890 / 24步 / 1024²）")

strip(["epoch2", "epoch6", "epoch12"],
      ["epoch 2/12", "epoch 6/12", "epoch 12/12 (最终)"],
      "compare_lora_epoch.jpg", title="LoRA 训练过程对比（同种子 / 24步 / 1024²）")

strip(["illustrious_lora", "animagine"],
      ["Illustrious-XL-v2.0 + chisa LoRA", "animagine-xl-4.0（同提示词/种子）"],
      "compare_checkpoint.jpg", cell=640, title="底模对比")

strip(["hires_src", "hires_4k"], ["1344×768 原生（30步）", "→ 4x-UltraSharp + lanczos 3840×2160"],
      "compare_hires_4k.jpg", cell=768, title="高清 4K 流程（workflow: 角色_chisa_高清4K）")

preview("direct_4k", "preview_direct_4k.jpg", 1920)
preview("hires_4k", "preview_hires_4k.jpg", 1920)
preview("img2img", "preview_img2img.jpg", 900)
preview("inpaint", "preview_inpaint.jpg", 900)
print("done")
