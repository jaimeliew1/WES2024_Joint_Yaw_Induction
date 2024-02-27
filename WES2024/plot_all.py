from pathlib import Path
import importlib

# Assuming the modules are in the same directory as the script
folder_path = Path(__file__).parent

# Get a list of module filenames in the folder
module_files = [file.stem for file in folder_path.glob("Fig*.py")]

# Import all modules
for module_name in module_files:
    module = importlib.import_module(module_name)

    # If needed, you can access objects from the imported module, for example:
    # figure_instance = module.FigureClass()

    print(f"Module {module_name} imported successfully.")
    module.main()
