from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carebridge import db


def main() -> None:
    db.execute(
        "INSERT INTO messages (sender_user_id, recipient_user_id, body) VALUES (3, 4, 'Checking in today. Ready for our activity?')"
    )
    print('Demo message inserted.')


if __name__ == '__main__':
    main()
