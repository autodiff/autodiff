from autodiff import autodiff4py

def autodiff_number_format(self, spec):
    return format(self.val(), spec)

autodiff4py.real1st.__format__ = autodiff_number_format
autodiff4py.real2nd.__format__ = autodiff_number_format
autodiff4py.real3rd.__format__ = autodiff_number_format
autodiff4py.real4th.__format__ = autodiff_number_format

from autodiff.autodiff4py import *
# from autodiff._extensions.some_module import *
