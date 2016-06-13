

from setuptools import setup, find_packages
from distutils.extension import Extension

import os
import sys
import glob
import string
import shutil
import subprocess


def main():
  extra_compile_args = []
  extra_link_args = []

  setup(
    name = "simplemining",
    version = '1.0.0',
    namespace_packages=[],
		packages = ['simplemining'],
		package_dir = { 'simplemining' : 'simplemining' },

		data_files = [],

		description = "python machine learning utils",
		author = "elan_gu",
		author_email = "elan_gu@quantone.com",

		license = "",
		keywords = (),
		platforms = "Independant",
		url = "",

		cmdclass = {},
		ext_modules = []
	)

if __name__ == '__main__':
	main()

