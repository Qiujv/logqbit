"""Command-line interface for logqbit utilities."""

import argparse
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path


def copy_template(template_name: str, output_path: Path | None = None) -> int:
    """Copy a template script to the current directory or specified path.
    
    Args:
        template_name: Name of the template (without .py extension)
        output_path: Optional output path. If None, uses current directory.
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    template_file = f"{template_name}.py"
    
    try:
        # Get template from package
        template_path = files("logqbit") / "templates" / template_file
        
        if not template_path.is_file():
            print(f"Error: Template '{template_name}' not found.", file=sys.stderr)
            print("\nAvailable templates:", file=sys.stderr)
            templates_dir = files("logqbit") / "templates"
            for item in templates_dir.iterdir():
                if item.name.endswith(".py"):
                    print(f"  - {item.name[:-3]}", file=sys.stderr)
            return 1
        
        # Determine output path
        if output_path is None:
            output_path = Path.cwd() / template_file
        elif output_path.is_dir():
            output_path = output_path / template_file
        
        # Check if file already exists
        if output_path.exists():
            response = input(f"File '{output_path}' already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return 0
        
        # Copy template
        with template_path.open('rb') as src:
            output_path.write_bytes(src.read())
        
        print(f"✓ Template copied to: {output_path}")
        print(f"\nNext steps:")
        print(f"  1. Edit the file to configure your paths")
        print(f"  2. Run: python {output_path.name}")
        
        return 0
        
    except Exception as exc:
        print(f"Error copying template: {exc}", file=sys.stderr)
        return 1


def create_shortcuts(output_dir: Path | None = None) -> int:
    """Create desktop shortcuts for logqbit-browser and logqbit-live-plotter.
    
    Creates .lnk files with custom icons on the desktop or specified directory.
    If ICO files don't exist, they will be automatically generated from SVG files.
    
    Args:
        output_dir: Directory to create shortcuts in. If None, uses desktop.
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Find executable paths
        browser_exe = shutil.which("logqbit-browser")
        plotter_exe = shutil.which("logqbit-live-plotter")
        
        if not browser_exe:
            print("Error: logqbit-browser.exe not found in PATH", file=sys.stderr)
            print("Please ensure logqbit is installed.", file=sys.stderr)
            return 1
        
        if not plotter_exe:
            print("Error: logqbit-live-plotter.exe not found in PATH", file=sys.stderr)
            print("Please ensure logqbit is installed.", file=sys.stderr)
            return 1
        
        # Get icon paths from package
        assets_dir = files("logqbit") / "assets"
        browser_svg = assets_dir / "browser.svg"
        plotter_svg = assets_dir / "live_plotter.svg"
        browser_ico = assets_dir / "browser.ico"
        plotter_ico = assets_dir / "live_plotter.ico"
        
        # Generate ICO files if they don't exist
        icons_to_generate = []
        if not browser_ico.is_file():
            if not browser_svg.is_file():
                print(f"Error: SVG file not found: {browser_svg}", file=sys.stderr)
                return 1
            icons_to_generate.append(("browser", browser_svg, browser_ico))
        
        if not plotter_ico.is_file():
            if not plotter_svg.is_file():
                print(f"Error: SVG file not found: {plotter_svg}", file=sys.stderr)
                return 1
            icons_to_generate.append(("live_plotter", plotter_svg, plotter_ico))
        
        # Generate missing ICO files
        if icons_to_generate:
            print("Generating ICO files from SVG...")
            try:
                from logqbit.misc.svg2ico import svg_to_ico
                
                for name, svg_path, ico_path in icons_to_generate:
                    print(f"  Converting {name}.svg to {name}.ico...")
                    # Convert to writable path if needed
                    svg_str = str(svg_path)
                    ico_str = str(ico_path)
                    
                    # If package path is read-only, use temp location and inform user
                    try:
                        svg_to_ico(svg_str, ico_str)
                        print(f"  ✓ Created: {ico_path}")
                    except (PermissionError, OSError) as e:
                        # Package directory might be read-only, try user's local data dir
                        import tempfile
                        temp_ico = Path(tempfile.gettempdir()) / f"logqbit_{name}.ico"
                        svg_to_ico(svg_str, str(temp_ico))
                        print(f"  ✓ Created: {temp_ico} (package dir not writable)")
                        # Update ico path to use temp location
                        if name == "browser":
                            browser_ico = temp_ico
                        else:
                            plotter_ico = temp_ico
                
            except ImportError as exc:
                print(f"Error: Could not import svg2ico: {exc}", file=sys.stderr)
                print("Please ensure PySide6 is installed.", file=sys.stderr)
                return 1
            except Exception as exc:
                print(f"Error generating ICO files: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return 1
        
        # Determine output directory
        if output_dir is None:
            # Use PowerShell to get the actual desktop path
            result = subprocess.run(
                ["powershell", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                print("Error: Could not determine Desktop path", file=sys.stderr)
                return 1
            
            desktop_path = Path(result.stdout.strip())
            if not desktop_path.exists():
                print(f"Error: Desktop directory does not exist: {desktop_path}", file=sys.stderr)
                return 1
            
            output_dir = desktop_path
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create shortcuts using PowerShell
        shortcuts = [
            {
                "name": "LogQbit Browser",
                "target": browser_exe,
                "icon": str(browser_ico),
                "output": output_dir / "LogQbit Browser.lnk"
            },
            {
                "name": "LogQbit Live Plotter",
                "target": plotter_exe,
                "icon": str(plotter_ico),
                "output": output_dir / "LogQbit Live Plotter.lnk"
            }
        ]
        
        for sc in shortcuts:
            # PowerShell script to create shortcut with icon
            ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{sc['output']}")
$Shortcut.TargetPath = "{sc['target']}"
$Shortcut.IconLocation = "{sc['icon']}"
$Shortcut.Save()
"""
            
            # Run PowerShell script
            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Error creating shortcut: {sc['name']}", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                return 1
            
            print(f"✓ Created: {sc['output']}")
        
        print(f"\n✓ Shortcuts created in: {output_dir}")
        return 0
        
    except Exception as exc:
        print(f"Error creating shortcuts: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def create_example_data() -> int:
    """Create example data folders and launch browser.
    
    Creates:
        - logqbit_example/ directory with 3 example log folders (0, 1, 2)
        - Launches the browser automatically
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        import numpy as np

        from logqbit.logfolder import LogFolder
        
        cwd = Path.cwd()
        example_dir = cwd / "logqbit_example"
        
        # Check if directory already exists
        if example_dir.exists():
            response = input(
                f"Directory '{example_dir}' already exists. Overwrite? (y/N): "
            )
            if response.lower() != 'y':
                print("Cancelled.")
                return 0
            # Remove existing directory
            import shutil
            shutil.rmtree(example_dir)
        
        # Create example directory
        example_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating example data in: {example_dir}")
        
        # Example 1: Simple linear data
        print("  Creating example 0: Linear relationship...")
        log0_path = example_dir / "0"
        log0 = LogFolder(log0_path, create=True)
        log0.meta.title = "Linear Relationship Example"
        log0.meta.star = 1
        log0.meta.plot_axes = ["x"]
        x_vals = np.linspace(0, 10, 50)
        for x in x_vals:
            log0.add_row(x=x, y=2 * x + 1, z=x**2)
        log0.const["description"] = "y = 2x + 1, z = x^2"
        log0.const["experiment_type"] = "simulation"
        log0.flush()
        
        # Example 2: Sinusoidal data with noise
        print("  Creating example 1: Sinusoidal with noise...")
        log1_path = example_dir / "1"
        log1 = LogFolder(log1_path, create=True)
        log1.meta.title = "Noisy Sinusoidal Signal"
        log1.meta.star = 2
        log1.meta.plot_axes = ["time"]
        time_vals = np.linspace(0, 4 * np.pi, 100)
        for t in time_vals:
            signal = np.sin(t)
            noise = np.random.normal(0, 0.1)
            log1.add_row(time=t, signal=signal, noisy=signal + noise)
        log1.const["description"] = "sin(t) with Gaussian noise"
        log1.const["frequency"] = "1 Hz"
        log1.const["noise_level"] = 0.1
        log1.flush()
        
        # Example 3: 2D scan data
        print("  Creating example 2: 2D parameter scan...")
        log2_path = example_dir / "2"
        log2 = LogFolder(log2_path, create=True)
        log2.meta.title = "2D Parameter Scan"
        log2.meta.star = 3
        log2.meta.plot_axes = ["voltage", "frequency"]
        voltages = np.linspace(-1, 1, 20)
        frequencies = np.linspace(1, 10, 20)
        for v in voltages:
            for f in frequencies:
                # Simulate some resonance-like behavior
                response = np.exp(-((f - 5.5)**2) / 2) * np.exp(-((v - 0.2)**2) / 0.5)
                response += np.random.normal(0, 0.05)
                log2.add_row(voltage=v, frequency=f, response=response)
        log2.const["description"] = "Simulated resonance scan"
        log2.const["voltage_unit"] = "V"
        log2.const["frequency_unit"] = "GHz"
        log2.flush()
        
        print(f"\n✓ Created 3 example log folders in: {example_dir}")
        print("\nLaunching browser...")
        
        # Launch browser directly
        from logqbit.browser import main as browser_main
        return browser_main([str(example_dir)])
        
    except ImportError as exc:
        print(f"Error: Missing required dependency: {exc}", file=sys.stderr)
        print("Please ensure logqbit is properly installed.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error creating example data: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="logqbit",
        description="Logqbit command-line utilities",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # copy-template command
    copy_parser = subparsers.add_parser(
        "copy-template",
        help="Copy a template script to your working directory",
    )
    copy_parser.add_argument(
        "template",
        help="Template name (e.g., 'move_from_labrad')",
    )
    copy_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path (default: current directory)",
    )
    
    # browser command (for convenience)
    browser_parser = subparsers.add_parser(
        "browser",
        help="Launch the log browser GUI",
    )
    browser_parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Directory to open (default: current directory)",
    )
    
    # browser-demo command
    demo_parser = subparsers.add_parser(
        "browser-demo",
        help="Create example data and launch browser",
    )
    
    # shortcuts command
    shortcuts_parser = subparsers.add_parser(
        "shortcuts",
        help="Create desktop shortcuts with custom icons",
    )
    shortcuts_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory (default: Desktop)",
    )
    
    args = parser.parse_args()
    
    if args.command == "copy-template":
        return copy_template(args.template, args.output)
    
    elif args.command == "browser":
        from logqbit.browser import main as browser_main
        directory = str(args.directory) if args.directory else None
        browser_args = [directory] if directory else []
        return browser_main(browser_args)
    
    elif args.command == "browser-demo":
        return create_example_data()
    
    elif args.command == "shortcuts":
        return create_shortcuts(args.output)
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
