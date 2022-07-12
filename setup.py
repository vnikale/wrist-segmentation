from distutils.core import setup


setup(name='wrist_segmentation',
      version='0.9',
      description='Segmentation of a wrist cartilage on MRI',
      author='Nikita Vlaidmirov',
      author_email='vladimnikita@gmail.com',
      packages=['wrist_segmentation',
                'wrist_segmentation.data',
                'wrist_segmentation.data.loader',
                'wrist_segmentation.data.preprocess',
                'wrist_segmentation.models',
                'wrist_segmentation.utils'],
     )