#ifdef __clang__
#define CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS // Prevents error: 'uncaught_exceptions' is unavailable: introduced in macOS 10.12 (see discussion at https://github.com/catchorg/Catch2/issues/1218)
#endif
