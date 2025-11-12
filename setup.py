from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='aoemtl',
    version='1.0.0',
    description='Autonomy of Experts (AoE-MTL): A gate-free expert-based multi-task learning framework built on LibMTL',
    author='Anand Mogul',
    url='https://github.com/anand-amon/aoemtl',
    packages=find_packages(),
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'torch>=2.3.0',
        'torchvision>=0.18.0',
        'numpy>=1.26',
        'scipy>=1.13',
    ],
    python_requires='>=3.8',
)
