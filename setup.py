# ============================
# ========= SETUP ============
# ============================

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='MLOPS-PROJECT-1',
    version='1.0',
    description='An End-to-end MLOPS `Hotel Reservation Prediction` project on customer booking, target makrketing and predictions',
    author='Rohit Dusane',
    author_email='stat.data247@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.12',
    license="MIT",
)
