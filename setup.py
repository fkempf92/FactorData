import setuptools

setuptools.setup(
    name='factordata',
    version='1.0',
    author='Felix Kempf',
    author_email='felix.kempf@kcl.ac.uk',
    description='Constructs 103 US firm-level firm characteristics and corresponding L/S portfolios',
    url='https://github.com/fkempf92/FactorData',
    project_urls = {
        "Bug Tracker": "https://github.com/fkempf92/FactorData/issues"
    },
    license='MIT',
    packages=['factordata'],
    install_requires=['bls >= 0.3.0',
                      'pandas >= 1.0.5',
                      'numpy >= 1.18.5',
                      'wrds >= 3.1.1',
                      'scipy >= 1.5.0', 
                      'statsmodels >= 0.11.1',
                      'fredapi >= 0.5.0'],
)
