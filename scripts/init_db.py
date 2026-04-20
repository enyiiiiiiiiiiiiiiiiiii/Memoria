from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carebridge import db
from carebridge.seed import seed_if_needed


def main() -> None:
    db.init_db()
    seed_if_needed()
    print("Database initialised and demo data seeded.")


if __name__ == "__main__":
    main()
