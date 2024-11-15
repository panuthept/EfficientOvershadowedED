from setuptools import setup, find_packages


with open("requirements.txt") as f:
    dependencies = [line for line in f]

setup(
    name='efficient_overshadowed_ed',
    version='0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    license='Apache-2.0 License',
    author='Panuthep Tasawong',
    author_email='panuthep.t_s20@vistec.ac.th',
    description='Efficient Overshadowed ED',
    # python_requires='==3.11.4',
    install_requires=dependencies
)