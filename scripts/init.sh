# one-time
pixi init        # creates pixi.toml
git init

# add pyproject.toml manually (above)
uv venv .venv
pixi run setup   # venv + uv sync + pre-commit

# dev loop
pixi run all
pixi run run
pixi run cov

# release
uv build
uv publish

