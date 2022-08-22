from setuptools import setup

setup(
    name='raw2rgb',
    version='0.2.0',
    description='A python tool to convert .tiff image file to .png file. GitHub link: https://github.com/vilktor370/raw2rgb',
    author='Haochen Yu',
    zip_safe=False,
    install_requires=[
        'Pillow>=9.2.0',
        'scipy>=1.9.0',
        'numpy>=1.14.5',
        'matplotlib>=3.5.2'        
    ],
)