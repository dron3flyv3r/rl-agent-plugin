# Building the Native C++ GDExtension

The plugin ships with a prebuilt Linux binary (`native/bin/librl_cnn.so`).
If you need to rebuild it — or produce a binary for **Windows** or **macOS** — follow this guide.

---

## What This Extension Does

`rl_cnn` is the CPU-side CNN encoder used for image observations.
When Vulkan is available, a Godot-native GPU path takes over during PPO training, but the CPU encoder (this library) is always required for rollout inference.

---

## Prerequisites

### All platforms

| Tool | Minimum version | Notes |
|------|----------------|-------|
| CMake | 3.22 | [cmake.org/download](https://cmake.org/download/) |
| Git | any recent | needed to fetch the `godot-cpp` submodule |
| C++ compiler | C++17 support | see per-platform section below |

### Linux

- GCC 9+ or Clang 10+ (most distros already ship a suitable version)
- Install via package manager if needed:

  ```bash
  # Debian / Ubuntu
  sudo apt install build-essential cmake git

  # Fedora / RHEL
  sudo dnf install gcc-c++ cmake git
  ```

### macOS

- Xcode Command Line Tools (provides Apple Clang):

  ```bash
  xcode-select --install
  ```

- CMake via Homebrew (optional but convenient):

  ```bash
  brew install cmake
  ```

### Windows

- Visual Studio 2019 or 2022 with the **Desktop development with C++** workload, **or**
- MSYS2 + MinGW-w64 (for a GCC-based build)
- CMake 3.22+ (the Visual Studio installer can install it, or download from cmake.org)

---

## Step 1 — Initialize the `godot-cpp` Submodule

The build requires `godot-cpp` at `native/godot-cpp`.
If you cloned the plugin with `--recursive` this is already done.
Otherwise, from the plugin root:

```bash
git submodule update --init --recursive
```

Verify the directory exists and is not empty:

```bash
ls native/godot-cpp/CMakeLists.txt
```

---

## Step 2 — Configure With CMake

Run all commands from the **plugin root** (`addons/rl-agent-plugin/` inside your game project, or the root of the cloned plugin repository).

### Linux

```bash
cmake -S native -B native/build \
      -DCMAKE_BUILD_TYPE=Release
```

### macOS

```bash
cmake -S native -B native/build \
      -DCMAKE_BUILD_TYPE=Release
```

To produce a universal binary (x86_64 + arm64) for distribution:

```bash
cmake -S native -B native/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
```

### Windows (Visual Studio)

Open a **Developer Command Prompt** or run from a normal terminal with CMake on `PATH`:

```bat
cmake -S native -B native\build -G "Visual Studio 17 2022" -A x64
```

Replace `"Visual Studio 17 2022"` with `"Visual Studio 16 2019"` if you use VS 2019.

### Windows (MinGW / MSYS2)

From the MSYS2 MinGW64 shell:

```bash
cmake -S native -B native/build \
      -G "MinGW Makefiles" \
      -DCMAKE_BUILD_TYPE=Release
```

---

## Step 3 — Build

### Linux / macOS

```bash
cmake --build native/build --config Release -- -j$(nproc)
```

macOS alternative using `sysctl` for core count:

```bash
cmake --build native/build --config Release -- -j$(sysctl -n hw.logicalcpu)
```

### Windows (Visual Studio)

```bat
cmake --build native\build --config Release
```

### Windows (MinGW)

```bash
cmake --build native/build --config Release
```

---

## Step 4 — Verify The Output

After a successful build the library lands in `native/bin/`:

| Platform | File |
|----------|------|
| Linux | `native/bin/librl_cnn.so` |
| macOS | `native/bin/librl_cnn.dylib` |
| Windows | `native/bin/librl_cnn.dll` |

These exact paths are what `native/rl_cnn.gdextension` references, so no extra copy step is needed.

---

## AVX2 / FMA (Optional)

The CMake build automatically detects and enables AVX2 + FMA on x86-64 hosts.
To disable this (e.g., for a generic/portable build):

```bash
cmake -S native -B native/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DDISABLE_AVX2=ON
```

---

## Debug Build

Replace `Release` with `Debug` in all commands above.
Debug and release outputs go to the same `native/bin/` directory; the `.gdextension` file maps both configurations to the same binary name, so only one can be present at a time.

---

## Cross-Compilation Notes

The CMakeLists does not include a cross-compilation toolchain file.
If you need to cross-compile (e.g., building a Linux `.so` on Windows), supply your own CMake toolchain file via `-DCMAKE_TOOLCHAIN_FILE=<path>` and ensure `godot-cpp` is compiled for the target platform.
For most users, building natively on the target OS is simpler.

---

## After Building

1. Confirm `native/bin/` contains the correct library for your OS.
2. Open Godot and rebuild the project once (`Alt+B`) so the editor reloads the extension.
3. If the extension fails to load, check the Godot console for `GDExtension` error messages and confirm your platform and architecture match what the `.gdextension` file declares.

---

## Related Docs

- [GPU CNN Training](gpu-cnn.md) — how the CPU encoder integrates with the optional GPU training path
- [Sensors](sensors.md) — camera and image observation setup that triggers the CNN encoder
- [Architecture](architecture.md) — overall plugin structure
