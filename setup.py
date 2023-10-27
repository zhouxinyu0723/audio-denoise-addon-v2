from setuptools import setup

setup(name='ZENNet',
      version='0.1',
      description='Build, train and test ZENNet',
      url='http://github.com/',
      author='Xinyu Zhou',
      author_email='abc@def.com',
      license='apache license v.2.0',
      packages=['ZENNet'],
			install_requires=[
					"librosa==0.10.1",
					"matplotlib==3.8.0",
					"pyqt5==5.15.10",
					"pytest==7.4.2",
					"sounddevice==0.4.6",
					"speechbrain==0.5.15",
			],
      zip_safe=False)
