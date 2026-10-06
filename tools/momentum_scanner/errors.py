class ScannerError(Exception):
    """Base class for momentum-scanner errors."""


class ScannerNetworkError(ScannerError):
    """A network/HTTP call failed, returned a non-200 status, or returned a body
    that could not be parsed."""
