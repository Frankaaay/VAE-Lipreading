import kagglehub
import shutil
import os

print("Downloading dataset via kagglehub... (This might take a while for 15GB)")
# Download latest version
cache_path = kagglehub.dataset_download("mohamedbentalb/lipreading-dataset")

print(f"Dataset successfully downloaded to Kaggle cache: {cache_path}")

# Optional: If you want to move it into your project's data/ folder securely
project_data_dir = os.path.join(os.getcwd(), "data")

if not os.path.exists(project_data_dir):
    print(f"Copying files to local {project_data_dir} folder...")
    shutil.copytree(cache_path, project_data_dir)
    print("Done!")
else:
    print(f"Data folder already exists at {project_data_dir}.")
