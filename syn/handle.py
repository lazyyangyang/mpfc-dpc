from PIL import Image, ImageDraw, ImageFont
import os

# 图片路径列表
img_dir = "merged_results"  # 替换为你的图片所在目录
img_files = ["3_1.jpg", "3_2.jpg", "3_3.jpg", "3_4.jpg"]  # 这里你可以扩展到12行
img_paths = [os.path.join(img_dir, f) for f in img_files]

# 打开图片
images = [Image.open(p) for p in img_paths]

# 假设所有图片宽度一致，如果宽度不一致可统一缩放
widths, heights = zip(*(i.size for i in images))
max_width = max(widths)
total_height = sum(heights)

# 创建空白大图
final_im = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))

# 设置字体（可选，使用默认字体）
try:
    font = ImageFont.truetype("arial.ttf", 30)  # 可以修改大小
except:
    font = ImageFont.load_default()

# 拼接并加标号
y_offset = 0
labels = ["(a)", "(b)", "(c)", "(d)"]  # 每3行标一个

for idx, im in enumerate(images):
    final_im.paste(im, (0, y_offset))
    
    # 每三行添加一个标号
    if idx % 3 == 0:
        draw = ImageDraw.Draw(final_im)
        label_text = labels[idx // 3]
        
        # 使用 textbbox 获取文字尺寸
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 将标号放在三行图像的中间
        middle_y = y_offset + sum(heights[idx:idx+3]) // 2 - text_height // 2
        x = (max_width - text_width) // 2
        y = middle_y
        
        draw.text((x, y), label_text, fill=(255, 0, 0), font=font)
    
    y_offset += im.height

# 保存结果
out_dir = "merged_results"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "merged_c100_4.jpg")
final_im.save(out_path)
print(f"Saved merged image with labels: {out_path}")

# from PIL import Image
# import os

# # 图片路径列表
# img_dir = "merged_results"  # 替换为你的图片所在目录
# img_files = ["1_1.jpg", "1_2.jpg", "1_3.jpg", "1_4.jpg"]
# img_paths = [os.path.join(img_dir, f) for f in img_files]

# # 打开图片
# images = [Image.open(p) for p in img_paths]

# # 假设所有图片宽度一致，如果宽度不一致可统一缩放
# widths, heights = zip(*(i.size for i in images))
# max_width = max(widths)
# total_height = sum(heights)

# # 创建空白大图
# final_im = Image.new('RGB', (max_width, total_height))

# # 竖向拼接
# y_offset = 0
# for im in images:
#     final_im.paste(im, (0, y_offset))
#     y_offset += im.height

# # 保存结果
# out_dir = "merged_results"
# os.makedirs(out_dir, exist_ok=True)
# out_path = os.path.join(out_dir, "merged_vertical.jpg")
# final_im.save(out_path)
# print(f"Saved merged image: {out_path}")
# import os
# from PIL import Image

# # 图片所在目录
# img_dir = "merged_results"
# out_dir = "final_results"
# os.makedirs(out_dir, exist_ok=True)

# # 遍历目录，收集所有图片
# img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]

# # 按类别号分组
# from collections import defaultdict
# categories = defaultdict(list)
# for f in img_files:
#     # 假设命名格式: 类别号_类型号.jpg
#     name, ext = os.path.splitext(f)
#     cat_id = name.split("_")[0]  # 类别号
#     categories[cat_id].append(f)

# # 生成每类拼接图
# rows = []
# for cat_id in sorted(categories.keys(), key=lambda x: int(x)):
#     files = sorted(categories[cat_id], key=lambda x: int(os.path.splitext(x)[0].split("_")[1]))  # 类型号排序
#     images = [Image.open(os.path.join(img_dir, f)) for f in files]

#     # 横向拼接这一行
#     width, height = images[0].size
#     row_im = Image.new('RGB', (width * len(images), height))
#     for i, im in enumerate(images):
#         row_im.paste(im, (i * width, 0))
#     rows.append(row_im)

# # 拼接所有行（按类别顺序）
# if rows:
#     row_width, row_height = rows[0].size
#     final_im = Image.new('RGB', (row_width, row_height * len(rows)))
#     for i, row in enumerate(rows):
#         final_im.paste(row, (0, i * row_height))
    
#     out_path = os.path.join(out_dir, "merged_by_category.jpg")
#     final_im.save(out_path)
#     print(f"Saved merged image: {out_path}")
# else:
#     print("No images found.")