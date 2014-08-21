from setuptools import setup

setup(name='rainbow',
    version='0',
    packages = ['rainbow','rainbow.utils'],
    package_data = {'rainbow': ['data_files/*']},
    description = "Fit multi-colour transit data",
    author = "S. Littlefair",
    url = "https://github.com/StuartLittlefair/rainbow",
    author_email = "s.littlefair@shef.ac.uk",
    install_requires=[
        'numpy',
        'triangle_plot'
    ],
    scripts = ['scripts/transitModel.py','scripts/limbdark.py']
    )
    