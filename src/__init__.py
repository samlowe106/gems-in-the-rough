# Intentionally empty. Re-exporting `main` here made `python -m src.main` import
# the module twice (runpy warns about it), and pulled torch, transformers, and Qt
# into every `import src`. Import from the submodules instead.
