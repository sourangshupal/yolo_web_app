import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yolo_object_detector",
    version="0.0.1",
    author="Sourangshu Pal",
    author_email="paul.visionai@gmail.com",
    description="A Streamlit app for YOLOv11 object detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/yolo_object_detector",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
    install_requires=[
        'streamlit',
        'opencv-python',
        'Pillow',
        'numpy',
        'torch',
        'torchvision',
        'ultralytics',
        'requests',
        'tqdm',
        'python-json-logger',  # Added for structured logging
    ],
)
