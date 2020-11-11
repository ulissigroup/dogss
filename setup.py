
from setuptools import setup, find_packages
from distutils.command.install import INSTALL_SCHEMES


for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

setup(name='DOGSS',
      version='0.0.1',
      description='Differentiable Optimization for the prediction of Ground State Structure',
      url='https://github.com/ulissigroup/dogss',
      author='Junwoong (Jun) Yoon, Zachary Ulissi',
      author_email='junwoony@andrew.cmu.edu',
      license='GPL',
      platforms=[],
      packages=find_packages(),
      scripts=[],
      include_package_data=False,
      install_requires=[
			'ase>=3.19.1',
			'numpy',
			'matplotlib',
            'seaborn',
            'pymatgen',
          'torch',
          'pyyaml',
          'tqdm',
          'seaborn',
          'skorch==0.4.0',      
          'sklearn',
      ],
)
