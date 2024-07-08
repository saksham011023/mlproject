from setuptools import find_packages,setup
from typing import List

Hyphon_e_dot='-e .'
def get_requirements(file_path:str)->list:

    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open(file_path ) as file_obj:
        requirements=file_obj.readlines()
        requirements= [req.replace('\n','') for req in requirements]

        if Hyphon_e_dot in requirements:
            requirements.remove(Hyphon_e_dot)
        
    return requirements

setup(
name='mlproject',
version='0.0.1',
author='saksham',
author_email='msaksham1016@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)