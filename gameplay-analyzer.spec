# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec.

One file, no console window suppression (this is a CLI), and an aggressive
exclude list. The excludes are not cosmetic: a naive bundle of anything
importing NumPy drags in matplotlib, tkinter, and test suites, roughly tripling
the download for code that never runs.

SciPy is absent from the runtime entirely — the one function this project used
from it, signal.welch, is reimplemented in analyzer/psd.py and verified against
SciPy to 1e-16 in tests/test_psd.py. That alone is ~112 MB off the bundle.

Build:  pyinstaller gameplay-analyzer.spec --noconfirm
"""

block_cipher = None

datas = [
    ('profiles', 'profiles'),                       # per-game ROIs, loaded at runtime
    ('schemas', 'schemas'),                         # record schema, for validation
]

# Modules that get pulled in transitively and are never used at runtime.
excludes = [
    'scipy',                # replaced by analyzer/psd.py
    'matplotlib', 'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'IPython', 'jupyter', 'notebook', 'pytest', '_pytest',
    'pandas', 'sqlalchemy', 'setuptools', 'pip', 'wheel',
    'numpy.distutils', 'numpy.f2py', 'numpy.testing',
    'PIL', 'lib2to3', 'pydoc_data', 'doctest', 'unittest',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=['yaml', 'analyzer', 'analyzer.cli', 'analyzer.pipeline',
                   'analyzer.motion', 'analyzer.metrics', 'analyzer.events',
                   'analyzer.overlay', 'analyzer.synthetic', 'analyzer.psd',
                   'analyzer.profiles', 'analyzer.resources'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='gameplay-analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-packed binaries are a common false positive for AV
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
