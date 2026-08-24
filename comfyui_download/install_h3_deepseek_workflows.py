from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "comfyui_download" / "workflows"
DEFAULT_TARGET = ROOT / "ComfyUI_windows_portable" / "ComfyUI" / "user" / "default" / "workflows"


def main():
    target = Path(input(f"ComfyUI workflows directory [{DEFAULT_TARGET}]: ").strip() or DEFAULT_TARGET)
    target.mkdir(parents=True, exist_ok=True)
    files = sorted(SOURCE.glob("H3_DeepSeek_*.json"))
    for source in files:
        destination = target / source.name
        shutil.copy2(source, destination)
        print(f"Installed {destination}")
    print(f"Installed {len(files)} H3 DeepSeek workflows. Restart ComfyUI to refresh the workflow list.")


if __name__ == "__main__":
    main()
