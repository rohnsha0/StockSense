#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stocksense_api.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    
    # For production, you might want to use gunicorn instead
    if len(sys.argv) == 1:
        sys.argv.append('runserver')
        sys.argv.append('0.0.0.0:8000')
    
    execute_from_command_line(sys.argv)