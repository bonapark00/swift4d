import os
import shutil

# base_dir = "/ssd_data1/users/jypark/data/n3d/video_data/cut_roasted_beef"  # 여기를 실제 경로로 바꿔줘
# base_dir = "/ssd_data1/users/jypark/data/n3d/video_data/coffee_martini"  # 여기를 실제 경로로 바꿔줘
base_dir = "/ssd_data1/users/jypark/data/n3d/video_data/cook_spinach"  # 여기를 실제 경로로 바꿔줘

for cam_id in range(21):
    cam_folder = os.path.join(base_dir, f"cam{cam_id:02d}")
    images_folder = os.path.join(cam_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    png_files = sorted([
        f for f in os.listdir(cam_folder)
        if f.endswith(".png") and os.path.isfile(os.path.join(cam_folder, f))
    ])

    for idx, filename in enumerate(png_files):
        src_path = os.path.join(cam_folder, filename)
        dst_path = os.path.join(images_folder, f"{idx:04d}.png")
        # shutil.move(src_path, dst_path)
        shutil.copyfile(src_path, dst_path)
    print("cam",cam_id, " Done!")
print("변환 완료!")