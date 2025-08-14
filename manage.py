#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'iiscc_site.settings')
    # Ensure stray SQLite file is not present
    try:
        from pathlib import Path
        base_dir = Path(__file__).resolve().parent
        db_file = base_dir / "db.sqlite3"
        if db_file.exists():
            db_file.unlink()
    except Exception:
        pass
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
