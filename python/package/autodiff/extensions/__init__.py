import autodiff.autodiff4py as autodiff4py


# List of all autodiff C++ number types exported to Python. Extend this as other
# number types are exported!
_autodiff_number_types = [
    autodiff4py.real1st,
    autodiff4py.real2nd,
    autodiff4py.real3rd,
    autodiff4py.real4th,
]

# Define the __format__ methods for all autodiff number types. This is needed so
# that we can write formatted strings such as f"{x:.3f}" and avoid a Python
# runtime error.
def _autodiff_number_format(self, spec):
    return format(self.val(), spec)

for numbertype in _autodiff_number_types:
    numbertype.__format__ = _autodiff_number_format

# -------------------------------------------------------------------------------------
# The code below is needed so that autodiff numbers can be used in plotly figures.
# It teaches PlotlyJSONEncoder on how to encode an autodiff number to JSON.
try:
    from plotly.utils import PlotlyJSONEncoder
    plotly_default = PlotlyJSONEncoder.default

    def is_autodiff_number_type(o):
        return any (t for t in _autodiff_number_types if isinstance(o, t))

    def plotly_default_new(self, o):
        if is_autodiff_number_type(o):
            return o.val()
        else:
            return plotly_default(self, o)

    PlotlyJSONEncoder.default = plotly_default_new
except:
    pass
