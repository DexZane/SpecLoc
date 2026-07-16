"""Allow ``python -m specloc`` to behave like the console command."""

from specloc.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
