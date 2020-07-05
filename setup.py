from setuptools import setup, find_packages

setup(
  name = 'mogrifier',
  packages = find_packages(),
  version = '0.0.3',
  license='MIT',
  description = 'Implementation of Mogrifier circuit from Deepmind',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/mogrifier',
  keywords = ['artificial intelligence', 'natural language processing'],
  install_requires=[
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)
