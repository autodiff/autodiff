from autodiff import autodiff4py

# List of all autodiff C++ number types exported to Python. Extend this as other
# number types are exported!
_autodiff_number_types = [
    autodiff4py.real1st,
    autodiff4py.real2nd,
    autodiff4py.real3rd,
    autodiff4py.real4th,
]

# -------------------------------------------------------------------------------------
# Define the __format__ methods for all autodiff number types. This is needed so
# that we can write formatted strings such as f"{x:.3f}" and avoid a Python
# runtime error.
def _autodiff_number_format(self, spec):
    return format(self.val(), spec)

for numbertype in _autodiff_number_types:
    numbertype.__format__ = _autodiff_number_format


from autodiff.autodiff4py import *
# from autodiff._extensions.some_module import *
