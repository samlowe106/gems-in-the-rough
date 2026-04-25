uv python pin 3.13.9 # tensorflow isn't compatible with 3.14 yet
uv init

# only installing top level dependencies -- numpy, pandas, matplotlib are included
# Additional dependencies should be added below!
uv add httpx notebook geopandas geopy scikit-learn seaborn tensorflow tensorflow-datasets torch pre-commit

pre-commit autoupdate
pre-commit install

rm setup.sh
