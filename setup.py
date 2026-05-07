from setuptools import setup, find_packages

setup(
    name="afriaware",
    version="0.1.0",
    description="Digital Wellbeing Intelligence for African Languages",
    author="Your Name",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "gradio>=4.0.0",
    ],
)
