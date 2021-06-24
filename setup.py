from setuptools import setup, find_packages

setup(
  name = 'transformer-memorization',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='GPL-3',
  description = 'Transformer Memorization Experiments',
  author = 'Charles Foster',
  author_email = 'cfoster0@alumni.stanford.edu',
  url = 'https://github.com/cfoster0/transformer-memorization',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3',
    'hydra-core',
    'wandb',
    'simple-parallel-transformer @ git+ssh://git@github.com/cfoster0/simple-parallel-transformer.git@main'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
