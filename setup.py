from setuptools import setup, Extension, find_packages

setup(
    name="phoc_ext",
    version="0.1",
    packages=find_packages(include=["utils", "utils.*"]),
    ext_modules=[
        Extension(
            "utils.phoc.cphoc_vn",
            sources=["utils/phoc/src/cphoc_vn.c"],
            extra_compile_args=["-std=c99", "-O3"],
        ),
        Extension(
            "utils.phoc.cphoc",
            sources=["utils/phoc/src/cphoc.c"],
            extra_compile_args=["-std=c99", "-O3"],
        )
    ]
)
