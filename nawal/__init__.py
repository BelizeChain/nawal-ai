"""Nawal package namespace — redirects imports to the flat repo layout."""
import os as _os

__path__ = [_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))]
