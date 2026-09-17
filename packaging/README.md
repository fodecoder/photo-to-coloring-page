# Packaging as a standalone executable

These are instructions, not a committed binary — do not commit build
output (`build/`, `dist/`, `*.spec`) to the repository; it is already
excluded via `.gitignore`.

## Building with PyInstaller

Install PyInstaller into your dev environment (it is not a project
dependency, since it is only needed to produce a binary):

```bash
pip install pyinstaller
```

Build a one-file executable:

```bash
pyinstaller --onefile --name coloring-page src/coloring_page/__main__.py
```

The resulting binary is written to `dist/coloring-page` (or
`dist\coloring-page.exe` on Windows). Run it directly, e.g.:

```bash
dist/coloring-page photo.jpg output.png --style canny
```

## Important limitations

- **Builds are platform-specific.** PyInstaller does not cross-compile —
  a Windows `.exe` must be built on Windows, a macOS binary on macOS, and
  a Linux binary on Linux. Building on one platform never produces a
  binary usable on another.
- **Antivirus false positives are common.** One-file PyInstaller binaries
  are frequently flagged by antivirus software (they unpack a bundled
  Python runtime at startup, a pattern shared with some malware droppers).
  This is a known, widely-reported PyInstaller limitation, not a defect in
  this project. If distributing a build, expect to need to document this
  for recipients or sign the binary.
- **No binary is bundled with this repository.** Build the executable
  locally using the command above whenever you need one.
