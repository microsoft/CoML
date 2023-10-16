import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

dotenv_dir = Path.home() / ".coml"
dotenv_path = (dotenv_dir / ".env").resolve()

if not os.path.exists(dotenv_dir):
    os.makedirs(dotenv_dir, exist_ok=True)
if not os.path.exists(dotenv_path):
    # copy the default .env file
    shutil.copyfile(Path(__file__).parent / ".env.template", dotenv_path)

# Load the users .env file into environment variables
load_dotenv(dotenv_path, verbose=True, override=False)

del load_dotenv
