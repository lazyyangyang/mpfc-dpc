import os

def print_image_filenames(root_dir):
    # 指定images文件夹路径
    img_root_dir = os.path.join(root_dir, 'images')
    
    if not os.path.exists(img_root_dir):
        print(f"Error: {img_root_dir} does not exist.")
        return

    # 遍历文件夹并打印文件名
    print(f"Listing image files in: {img_root_dir}")
    for root, dirs, files in os.walk(img_root_dir):
        for file in files:
            if file.endswith(".JPEG"):
                print(file)

if __name__ == "__main__":
    root_dir = "./data/test"  # 修改为你实际的数据路径
    print_image_filenames(root_dir)
