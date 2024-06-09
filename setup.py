from setuptools import setup, find_packages

setup(
    name = 'mogrifier',
    packages = find_packages(),
    version = '0.0.5',
    license='MIT',
    description = 'Implementation of Mogrifier circuit from Deepmind',
    long_description_content_type = 'text/markdown',
    author = 'Phil Wang',
    author_email = 'lucidrains@gmail.com',
    url = 'https://github.com/lucidrains/mogrifier',
    keywords = [
        'artificial intelligence',
        'natural language processing',
        'improved conditioning'
    ],
    install_requires=[
        'einops>=0.8',
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
