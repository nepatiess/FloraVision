import tarfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

archive_path = project_root / "data" / "102flowers.tgz"
extract_path = project_root / "data"

print("Dataset çikartiliyor...")

with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=extract_path)

print("Dataset başariyla çikartildi.")