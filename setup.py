from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name="ArcClimate",
    version="1.0.0",
    license="MIT",
    description='約 5km メッシュの格子点データ郡 から気象データなどを読み取り、基準地域メッシュ（３次メッシ ュ 約１km メッシュ）ごとの気象データを補完計算可能なプログラムを作成することを目的とする',
    author="ArcClimate Development Team",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={'': ['LICENSE.txt']},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt'),
    entry_points= {
        'console_scripts': ['arcclimate=arcclimate.arcclimate:main']
    }
)