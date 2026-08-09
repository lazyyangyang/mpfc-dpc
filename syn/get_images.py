import os
from PIL import Image

# 三个学习模式文件夹
base_dirs = [
    "c10-ipc10-cosine",
    "c10-ipc10-linear",
    "c10-ipc10-step"
]

out_dir = "merged_results"
os.makedirs(out_dir, exist_ok=True)

rows = []

for base_dir in base_dirs:
    row_imgs = []
    gen_results_dir = os.path.join(base_dir, "generated_results")
    
    if not os.path.exists(gen_results_dir):
        print(f"Directory not found: {gen_results_dir}")
        continue
    
    # 遍历10个类别目录 new000 ~ new009
    for class_idx in range(10):
        sub_dir = f"new{str(class_idx).zfill(3)}"
        sub_path = os.path.join(gen_results_dir, sub_dir)
        if not os.path.isdir(sub_path):
            print(f"Missing category folder: {sub_path}")
            continue
        
        # 只取 id000 的图片
        filename = f"class{str(class_idx).zfill(3)}_id034.jpg"
        img_path = os.path.join(sub_path, filename)
        if os.path.exists(img_path):
            row_imgs.append(Image.open(img_path))
        else:
            print(f"Missing image: {img_path}")
    
    # 拼接成一行
    if row_imgs:
        width, height = row_imgs[0].size
        row_im = Image.new('RGB', (width * len(row_imgs), height))
        for i, im in enumerate(row_imgs):
            row_im.paste(im, (i * width, 0))
        rows.append(row_im)

# 拼接三行成最终大图
if rows:
    row_width, row_height = rows[0].size
    final_im = Image.new('RGB', (row_width, row_height * len(rows)))
    for i, row in enumerate(rows):
        final_im.paste(row, (0, i * row_height))

    out_path = os.path.join(out_dir, "merged_all.jpg")
    final_im.save(out_path)
    print(f"Saved final merged image: {out_path}")
else:
    print("No images found to merge.")



# import os
# from PIL import Image

# # 你的三类目录
# base_dirs = [
#     "c10-ipc10-cosine",
#     "c10-ipc10-linear",
#     "c10-ipc10-step"
# ]

# out_dir = "merged_results"
# os.makedirs(out_dir, exist_ok=True)

# rows = []

# for base_dir in base_dirs:
#     row_imgs = []
#     gen_results_dir = os.path.join(base_dir, "generated_results")
    
#     if not os.path.exists(gen_results_dir):
#         print(f"Directory not found: {gen_results_dir}")
#         continue
    
#     # 遍历 generated_results 下所有子文件夹
#     for sub_dir in sorted(os.listdir(gen_results_dir)):
#         sub_path = os.path.join(gen_results_dir, sub_dir)
#         if not os.path.isdir(sub_path):
#             continue
        
#         # 遍历子文件夹下的所有 png 图片
#         for filename in sorted(os.listdir(sub_path)):
#             if filename.lower().endswith(".jpg"):
#                 img_path = os.path.join(sub_path, filename)
#                 try:
#                     row_imgs.append(Image.open(img_path))
#                 except Exception as e:
#                     print(f"Failed to open {img_path}: {e}")
    
#     if row_imgs:
#         # 假设所有图片大小相同
#         width, height = row_imgs[0].size
#         row_im = Image.new('RGB', (width * len(row_imgs), height))
#         for i, im in enumerate(row_imgs):
#             row_im.paste(im, (i * width, 0))
#         rows.append(row_im)

# # 拼接多行
# if rows:
#     row_width, row_height = rows[0].size
#     final_im = Image.new('RGB', (row_width, row_height * len(rows)))
#     for i, row in enumerate(rows):
#         final_im.paste(row, (0, i * row_height))

#     out_path = os.path.join(out_dir, "merged_all.jpg")
#     final_im.save(out_path)
#     print(f"Saved final merged image: {out_path}")
# else:
#     print("No images found to merge.")

