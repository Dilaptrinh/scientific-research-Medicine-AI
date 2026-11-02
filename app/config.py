from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep real credentials local while allowing explicit shell variables to win.
load_dotenv(PROJECT_ROOT / ".env", override=False)
