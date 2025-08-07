#setup script used to package and distribute a Python project using setuptools.

# library used to facilitate packaging Python projects. 
# It helps build, install, and distribute Python code in a standardized way.
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


# Package Metadata

# current version of your package.
__version__ = "0.0.0"
# name of your GitHub repo.
REPO_NAME = "Agentic_RAG"
AUTHOR_USER_NAME = "RVA-TheCoder"
AUTHOR_EMAIL = "aakash.sharma00004@gmail.com"
# the folder name inside 'src' directory that contains your actual Python package(s).
SRC_REPO = "agentic_rag"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for Agentic RAG app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)

"""
Explanation of above code :

(a) name: name of the package (agentic_rag).
(b)version, author, author_email: metadata.

(c)description: a short description for your package.
(d)long_description: from README.md, used on PyPI.
(e)long_description_content="text/markdown": specifies the format of README.md.

(f)url: link to your GitHub repo.
(g)project_urls: additional URLs (e.g., Bug Tracker).
(h)package_dir={"": "src"}: Tells setuptools to look in the src/ directory for your package modules.

(i) packages=setuptools.find_packages(where="src"): Finds all packages inside the src/ directory to include in the distribution.

This script is typically saved as setup.py, and is used to:

- Build your Python package.
- Upload it to PyPI.
- Install it via pip (pip install . or pip install <your-package-name> once published).

"""


